#!/usr/bin/env python3
import math
import os
import signal
import subprocess
import time

import rosgraph
import rospy
import torch
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from planners import create_collision_env_grid, grid_to_world, world_to_grid

MODEL_PATH = "/home/ubuntu/project/qlearning_model.pt"

current_x, current_y, current_theta = 0.0, -4.0, 0.0
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def odom_callback(msg):
    global current_x, current_y, current_theta
    current_x = msg.pose.pose.position.x
    current_y = msg.pose.pose.position.y
    q = msg.pose.pose.orientation
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    current_theta = math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def ensure_master_running():
    if rosgraph.is_master_online():
        return None

    print("ROS master non detecte, lancement roscore...")
    p = subprocess.Popen(["roscore"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        if rosgraph.is_master_online():
            return p
        time.sleep(0.2)
    return p


def start_gazebo():
    print("Lancement Gazebo...")
    p = subprocess.Popen(["roslaunch", "circuit.launch"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(8)
    return p


def reset_robot_to_start(x=0.0, y=-4.0, z=0.05):
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
        print(f"Reset ignore: {e}")


def choose_valid_action(q_row, state, grid):
    # Trie les actions par valeur Q decroissante et choisit la premiere valide.
    ranked = torch.argsort(q_row, descending=True).tolist()
    r, c = state
    h, w = grid.shape

    for a in ranked:
        dr, dc = ACTIONS[a]
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
            return int(a), (nr, nc)

    # fallback
    return 0, state


def move_to_cell(pub, target_cell, rate_hz=15):
    tx, ty = grid_to_world(target_cell[0], target_cell[1])
    rate = rospy.Rate(rate_hz)

    for _ in range(45):
        dx = tx - current_x
        dy = ty - current_y
        dist = math.hypot(dx, dy)

        if dist < 0.06:
            break

        desired = math.atan2(dy, dx)
        err = wrap_angle(desired - current_theta)

        cmd = Twist()
        cmd.angular.z = max(-1.2, min(1.2, 2.0 * err))
        if abs(err) > 0.35:
            cmd.linear.x = 0.0
        else:
            cmd.linear.x = min(0.12, dist)

        pub.publish(cmd)
        rate.sleep()

    pub.publish(Twist())


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Modele introuvable: {MODEL_PATH}")
        print("Lance d'abord: python3 train_Qlearning.py")
        return

    payload = torch.load(MODEL_PATH, map_location="cpu")
    q_table = payload["q_table"]
    free_cells = [tuple(c) for c in payload["free_cells"]]
    start_state = tuple(payload["start_state"])
    goal_state = tuple(payload["goal_state"])

    state_to_idx = {cell: i for i, cell in enumerate(free_cells)}

    master_proc = ensure_master_running()

    rospy.init_node("test_qlearning", anonymous=True)
    gazebo_proc = start_gazebo()

    rospy.Subscriber("/odom", Odometry, odom_callback)
    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    reset_robot_to_start(0.0, -4.0, 0.05)
    time.sleep(1.0)

    grid = create_collision_env_grid().astype(int)

    steps = 0
    max_steps = 220

    try:
        while not rospy.is_shutdown() and steps < max_steps:
            state = world_to_grid(current_x, current_y)

            if state == goal_state:
                print("Goal atteint par le modele Q-learning.")
                break

            if state not in state_to_idx:
                print("Etat hors grille libre, arret.")
                break

            s_idx = state_to_idx[state]
            action_idx, target_cell = choose_valid_action(q_table[s_idx], state, grid)

            move_to_cell(pub, target_cell, rate_hz=15)

            if steps % 10 == 0:
                print(f"Step {steps}: state={state}, action={action_idx}, target={target_cell}")

            steps += 1

        if steps >= max_steps:
            print("Test termine (max steps atteint).")

    finally:
        pub.publish(Twist())
        if gazebo_proc is not None:
            os.kill(gazebo_proc.pid, signal.SIGINT)
        if master_proc is not None:
            os.kill(master_proc.pid, signal.SIGINT)


if __name__ == "__main__":
    main()
