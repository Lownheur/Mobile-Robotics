#!/usr/bin/env python3
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from planners import create_gazebo_env_grid, world_to_grid

# ---------------------------------
# Hyperparametres Q-learning
# ---------------------------------
EPISODES = 5000       # Un peu plus d'episodes car l'exploration est partout
MAX_STEPS = 400
ALPHA = 0.20
GAMMA = 0.98
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.998

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N_ACTIONS = len(ACTIONS)
MODEL_PATH = "/home/ubuntu/project/qlearning_2point_modele.pt"


def make_state_maps(grid):
    free_cells = list(zip(*np.where(grid == 0)))
    state_to_idx = {cell: i for i, cell in enumerate(free_cells)}
    return free_cells, state_to_idx


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
        return next_state, 300.0, True

    if hit_wall:
        return next_state, -10.0, False

    # Reward shaping leger
    d_now = abs(goal_state[0] - r) + abs(goal_state[1] - c)
    d_next = abs(goal_state[0] - nr) + abs(goal_state[1] - nc)

    reward = -1.0 + 0.8 * (d_now - d_next)
    return next_state, reward, False


def train_q_learning(grid, goal_state, goal_name):
    free_cells, state_to_idx = make_state_maps(grid)
    n_states = len(free_cells)

    q_table = torch.zeros((n_states, N_ACTIONS), dtype=torch.float32)

    epsilon = EPS_START
    success_count = 0

    print(f"\n--- Entrainement pour le point {goal_name} ---")
    for ep in range(EPISODES):
        # 100% random spawn
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

        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if (ep + 1) % 500 == 0:
            print(f"Episode {ep + 1}/{EPISODES} | epsilon={epsilon:.3f}")

    success_rate = 100.0 * success_count / EPISODES
    print(f"Training {goal_name} termine. Taux de succes : {success_rate:.1f}%")

    return q_table, free_cells, state_to_idx


def save_models(qA, qB, free_cells, pt_a, pt_b, grid_shape):
    payload = {
        "q_table_A": qA,
        "q_table_B": qB,
        "free_cells": free_cells,
        "point_A": pt_a,
        "point_B": pt_b,
        "grid_shape": grid_shape,
        "actions": ACTIONS,
    }
    torch.save(payload, MODEL_PATH)
    print(f"\nModele global enregistre: {MODEL_PATH}")


def main():
    grid = create_gazebo_env_grid().astype(np.uint8)

    point_A = world_to_grid(0.0, -4.0)
    point_B = world_to_grid(0.0, 4.0)

    # Deblocage local depart/arrivee
    for p in [point_A, point_B]:
        r, c = p
        grid[max(0, r - 1):min(grid.shape[0], r + 2), max(0, c - 1):min(grid.shape[1], c + 2)] = 0

    # L'entrainement : Q-table distinctes pour chaque objectif
    q_table_A, free_cells, _ = train_q_learning(grid, point_A, "A (0.0, -4.0)")
    q_table_B, _, _ = train_q_learning(grid, point_B, "B (0.0, 4.0)")

    save_models(q_table_A, q_table_B, free_cells, point_A, point_B, grid.shape)

if __name__ == "__main__":
    main()
