#!/usr/bin/env python3

import math
import os
import signal
import subprocess
import time
import numpy as np
import rospy
import rosgraph
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from nav_msgs.msg import Odometry

from planners import (
    create_gazebo_env_grid,
    create_collision_env_grid,
    grid_to_world,
    run_astar,
    run_dijkstra,
    run_greedy,
    world_to_grid,
)

# Etat robot lu depuis /odom.
current_x, current_y, current_theta = 0.0, -4.0, 0.0


def odom_callback(msg):
    global current_x, current_y, current_theta
    current_x = msg.pose.pose.position.x
    current_y = msg.pose.pose.position.y

    # Quaternion -> yaw.
    q = msg.pose.pose.orientation
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    current_theta = math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def ensure_master_running():
    # Demarre roscore automatiquement si besoin.
    if rosgraph.is_master_online():
        return None

    print("ROS master non detecte, lancement de roscore...")
    master_proc = subprocess.Popen(["roscore"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for _ in range(60):
        if rosgraph.is_master_online():
            print("ROS master pret.")
            return master_proc
        time.sleep(0.2)

    return master_proc


def start_gazebo_if_needed():
    # Si Gazebo n'est pas deja actif, on lance le monde.
    if rosgraph.is_master_online():
        try:
            rospy.wait_for_service("/gazebo/set_model_state", timeout=1.0)
            return None
        except Exception:
            pass

    print("Lancement de Gazebo (circuit.launch)...")
    gazebo_proc = subprocess.Popen(["roslaunch", "circuit.launch"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(8)
    return gazebo_proc


def reset_robot_to_start(x=0.0, y=-4.0, z=0.05):
    # Replace explicitement le robot au point de depart.
    try:
        rospy.wait_for_service("/gazebo/set_model_state", timeout=12.0)
        set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        state = ModelState()
        state.model_name = "turtlebot3_burger"
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        state.pose.orientation.x = 0.0
        state.pose.orientation.y = 0.0
        state.pose.orientation.z = 0.0
        state.pose.orientation.w = 1.0
        state.reference_frame = "world"

        set_state(state)
        print("Robot repositionne au depart.")
    except Exception as e:
        print(f"Reset robot ignore ({e}).")


def simulate(x, y, theta, v, w, dt=0.2):
    # Meme modele cinematique que le TP.
    x_new = x + v * math.cos(theta) * dt
    y_new = y + v * math.sin(theta) * dt
    theta_new = theta + w * dt
    return x_new, y_new, theta_new


def distance_to_path(x, y, path):
    # Distance au chemin (coordonnees monde).
    return min(math.hypot(x - px, y - py) for px, py in path)


def in_collision(x, y, grid):
    r, c = world_to_grid(x, y)
    return grid[r, c] == 1


def nearest_path_index(x, y, path):
    best_i = 0
    best_d = 1e9
    for i, (px, py) in enumerate(path):
        d = math.hypot(x - px, y - py)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def dwa_step(x, y, theta, v_current, w_current, path, grid):
    # Parametres TP.
    dt = 0.2
    horizon = 5

    # Parametres vitesse (adaptes au TurtleBot en sim).
    v_max = 0.22
    w_max = 1.0
    a_v = 0.5
    a_w = 1.0

    # Fenetre dynamique.
    v_min_dw = max(0.0, v_current - a_v * dt)
    v_max_dw = min(v_max, v_current + a_v * dt)
    w_min_dw = max(-w_max, w_current - a_w * dt)
    w_max_dw = min(w_max, w_current + a_w * dt)

    v_samples = np.linspace(v_min_dw, v_max_dw, 5)
    w_samples = np.linspace(w_min_dw, w_max_dw, 7)

    best_cost = 1e9
    best_u = (0.0, 0.0)
    best_state = (x, y, theta)
    best_track_err = 1e9

    # Cible locale: quelques points devant le point le plus proche.
    closest_idx = nearest_path_index(x, y, path)
    target_idx = min(len(path) - 1, closest_idx + 3)
    goal_pt = path[target_idx]

    for v in v_samples:
        for w in w_samples:
            xn, yn, thetan = x, y, theta
            collision = False

            for _ in range(horizon):
                xn, yn, thetan = simulate(xn, yn, thetan, v, w, dt)
                if in_collision(xn, yn, grid):
                    collision = True
                    break

            if collision:
                continue

            dist = distance_to_path(xn, yn, path)
            goal_theta = math.atan2(goal_pt[1] - yn, goal_pt[0] - xn)
            heading = abs(wrap_angle(goal_theta - thetan))
            velocity_cost = -v

            # Meme structure de cout que ton exemple.
            J = 1.0 * dist + 1.0 * heading + 1.5 * velocity_cost

            if J < best_cost:
                best_cost = J
                best_u = (float(v), float(w))
                best_state = (xn, yn, thetan)
                best_track_err = dist

    return best_u, best_state, best_track_err


if __name__ == "__main__":
    master_proc = ensure_master_running()

    # planning_grid: inflation pour calculer le chemin
    # collision_grid: murs reels pour verifier les collisions en mouvement
    planning_grid = create_gazebo_env_grid()
    collision_grid = create_collision_env_grid()

    start_grid = world_to_grid(0.0, -4.0)
    goal_grid = world_to_grid(0.0, 4.0)

    # Degager depart/arrivee pour eviter les blocages numeriques.
    r_s, c_s = start_grid
    r_g, c_g = goal_grid
    planning_grid[max(0, r_s - 1):min(planning_grid.shape[0], r_s + 2), max(0, c_s - 1):min(planning_grid.shape[1], c_s + 2)] = 0
    planning_grid[max(0, r_g - 1):min(planning_grid.shape[0], r_g + 2), max(0, c_g - 1):min(planning_grid.shape[1], c_g + 2)] = 0

    # On degage uniquement le depart dans la grille collision
    # pour eviter un faux blocage au spawn.
    collision_grid[max(0, r_s - 1):min(collision_grid.shape[0], r_s + 2), max(0, c_s - 1):min(collision_grid.shape[1], c_s + 2)] = 0

    print("Choix de l'algorithme global:")
    print("1 - A*")
    print("2 - Dijkstra")
    print("3 - Greedy")
    algo = input("Votre choix (1/2/3): ").strip()

    print("Calcul du chemin global...")
    if algo == "2":
        path_grid, _, _ = run_dijkstra(planning_grid, start_grid, goal_grid)
    elif algo == "3":
        path_grid, _, _ = run_greedy(planning_grid, start_grid, goal_grid)
    else:
        path_grid, _, _ = run_astar(planning_grid, start_grid, goal_grid)

    if not path_grid:
        print("Erreur: aucun chemin global trouve.")
        raise SystemExit(1)

    global_path = [grid_to_world(r, c) for r, c in path_grid]
    print(f"Chemin global calcule: {len(global_path)} points.")

    rospy.init_node("controller_dwa", anonymous=True)

    gazebo_proc = start_gazebo_if_needed()
    reset_robot_to_start(0.0, -4.0, 0.05)

    rospy.Subscriber("/odom", Odometry, odom_callback)
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    rate = rospy.Rate(15)

    v_current, w_current = 0.0, 0.0
    tracking_errors = []
    cmd_history = []
    step = 0

    goal_x, goal_y = global_path[-1]

    try:
        while not rospy.is_shutdown():
            goal_dist = math.hypot(goal_x - current_x, goal_y - current_y)
            if goal_dist < 0.20:
                break

            (v_cmd, w_cmd), _, e_track = dwa_step(
                current_x,
                current_y,
                current_theta,
                v_current,
                w_current,
                global_path,
                collision_grid,
            )

            cmd = Twist()
            cmd.linear.x = v_cmd
            cmd.angular.z = w_cmd
            pub.publish(cmd)

            v_current, w_current = v_cmd, w_cmd
            tracking_errors.append(e_track)
            cmd_history.append((v_cmd, w_cmd))

            step += 1
            if step % 20 == 0:
                print(
                    f"DWA step {step}: goal_dist={goal_dist:.2f}m, "
                    f"track_err={e_track:.2f}m, v={v_cmd:.2f}, w={w_cmd:.2f}"
                )

            rate.sleep()
    finally:
        # Arret final.
        cmd = Twist()
        pub.publish(cmd)

        if tracking_errors and len(cmd_history) > 1:
            mean_err = float(np.mean(tracking_errors))
            max_err = float(np.max(tracking_errors))
            dv = [abs(cmd_history[i][0] - cmd_history[i - 1][0]) for i in range(1, len(cmd_history))]
            dw = [abs(cmd_history[i][1] - cmd_history[i - 1][1]) for i in range(1, len(cmd_history))]
            print("===== Analyse navigation =====")
            print(f"Erreur moyenne de suivi : {mean_err:.3f} m")
            print(f"Erreur max de suivi     : {max_err:.3f} m")
            print(f"Smoothness lineaire (moy |Delta v|) : {float(np.mean(dv)):.3f} m/s")
            print(f"Smoothness angulaire (moy |Delta w|): {float(np.mean(dw)):.3f} rad/s")

        if gazebo_proc is not None:
            os.kill(gazebo_proc.pid, signal.SIGINT)
        if master_proc is not None:
            os.kill(master_proc.pid, signal.SIGINT)
