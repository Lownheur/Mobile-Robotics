#!/usr/bin/env python3
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from planners import create_gazebo_env_grid, world_to_grid

# ---------------------------------
# Hyperparametres Q-learning (simple)
# ---------------------------------
EPISODES = 3200
MAX_STEPS = 320
ALPHA = 0.20
GAMMA = 0.98
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.997

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N_ACTIONS = len(ACTIONS)
MODEL_PATH = "/home/ubuntu/project/qlearning_model.pt"


def make_state_maps(grid):
    free_cells = list(zip(*np.where(grid == 0)))
    state_to_idx = {cell: i for i, cell in enumerate(free_cells)}
    idx_to_state = {i: cell for i, cell in enumerate(free_cells)}
    return free_cells, state_to_idx, idx_to_state


def choose_action(q_table, s_idx, epsilon):
    if random.random() < epsilon:
        return random.randint(0, N_ACTIONS - 1)
    return int(torch.argmax(q_table[s_idx]).item())


def env_step(grid, state, action_idx, goal_state):
    r, c = state
    dr, dc = ACTIONS[action_idx]
    nr, nc = r + dr, c + dc

    h, w = grid.shape
    hit_wall = False

    if nr < 0 or nr >= h or nc < 0 or nc >= w:
        nr, nc = r, c
        hit_wall = True
    elif grid[nr, nc] == 1:
        nr, nc = r, c
        hit_wall = True

    next_state = (nr, nc)

    if next_state == goal_state:
        return next_state, 200.0, True

    if hit_wall:
        return next_state, -10.0, False

    # Reward shaping leger pour accelerer l'apprentissage.
    d_now = abs(goal_state[0] - r) + abs(goal_state[1] - c)
    d_next = abs(goal_state[0] - nr) + abs(goal_state[1] - nc)

    reward = -1.0 + 0.6 * (d_now - d_next)
    return next_state, reward, False


def extract_greedy_path(grid, q_table, state_to_idx, start_state, goal_state):
    s = start_state
    path = [s]
    visited = {s}

    for _ in range(MAX_STEPS):
        if s == goal_state:
            break

        s_idx = state_to_idx[s]
        a = int(torch.argmax(q_table[s_idx]).item())
        s_next, _, _ = env_step(grid, s, a, goal_state)

        if s_next in visited and s_next != goal_state:
            # on coupe si boucle
            break

        path.append(s_next)
        visited.add(s_next)
        s = s_next

    return path


def train_q_learning(grid, fixed_start, goal_state):
    free_cells, state_to_idx, _ = make_state_maps(grid)
    n_states = len(free_cells)

    q_table = torch.zeros((n_states, N_ACTIONS), dtype=torch.float32)

    epsilon = EPS_START
    rewards = []
    success_count = 0

    for ep in range(EPISODES):
        # 70% des episodes depuis le vrai depart, 30% depuis un start aleatoire libre.
        if random.random() < 0.70:
            s = fixed_start
        else:
            s = random.choice(free_cells)

        ep_reward = 0.0

        for _ in range(MAX_STEPS):
            s_idx = state_to_idx[s]
            a = choose_action(q_table, s_idx, epsilon)

            s_next, r, done = env_step(grid, s, a, goal_state)
            s_next_idx = state_to_idx[s_next]

            q_old = q_table[s_idx, a].item()
            max_next = torch.max(q_table[s_next_idx]).item()
            target = r if done else (r + GAMMA * max_next)
            q_table[s_idx, a] = q_table[s_idx, a] + ALPHA * (target - q_old)

            s = s_next
            ep_reward += r

            if done:
                success_count += 1
                break

        rewards.append(ep_reward)
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if (ep + 1) % 200 == 0:
            print(
                f"Episode {ep + 1}/{EPISODES} | "
                f"epsilon={epsilon:.3f} | reward={ep_reward:.1f}"
            )

    success_rate = 100.0 * success_count / EPISODES
    print(f"Training termine. Taux de succes global: {success_rate:.1f}%")

    return q_table, free_cells, state_to_idx, rewards


def save_model(q_table, free_cells, start_state, goal_state, grid_shape):
    payload = {
        "q_table": q_table,
        "free_cells": free_cells,
        "start_state": start_state,
        "goal_state": goal_state,
        "grid_shape": grid_shape,
        "actions": ACTIONS,
    }
    torch.save(payload, MODEL_PATH)
    print(f"Modele enregistre: {MODEL_PATH}")


def plot_results(grid, path, start_state, goal_state, rewards):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].imshow(grid, cmap="gray_r")
    if len(path) > 1:
        pr, pc = zip(*path)
        axs[0].plot(pc, pr, "r-", linewidth=2, label="Q policy path")
    axs[0].plot(start_state[1], start_state[0], "go", markersize=8, label="Start")
    axs[0].plot(goal_state[1], goal_state[0], "bo", markersize=8, label="Goal")
    axs[0].set_title(f"Learned path (points={len(path)})")
    axs[0].axis("off")
    axs[0].legend()

    axs[1].plot(rewards)
    axs[1].set_title("Episode rewards")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Reward")

    plt.tight_layout()
    plt.show()


def main():
    # Entrainement sur map avec inflation legere (marge pres des murs).
    grid = create_gazebo_env_grid().astype(np.uint8)

    start_state = world_to_grid(0.0, -4.0)
    goal_state = world_to_grid(0.0, 4.0)

    # Deblocage local depart/arrivee pour eviter un faux blocage numerique.
    rs, cs = start_state
    rg, cg = goal_state
    grid[max(0, rs - 1):min(grid.shape[0], rs + 2), max(0, cs - 1):min(grid.shape[1], cs + 2)] = 0
    grid[max(0, rg - 1):min(grid.shape[0], rg + 2), max(0, cg - 1):min(grid.shape[1], cg + 2)] = 0

    q_table, free_cells, state_to_idx, rewards = train_q_learning(grid, start_state, goal_state)

    # Sauvegarde modele.
    save_model(q_table, free_cells, start_state, goal_state, grid.shape)

    # Evaluation simple depuis le vrai depart.
    path = extract_greedy_path(grid, q_table, state_to_idx, start_state, goal_state)
    reached = path[-1] == goal_state
    print(f"Path learned points: {len(path)} | Goal reached: {reached}")

    plot_results(grid, path, start_state, goal_state, rewards)


if __name__ == "__main__":
    main()
