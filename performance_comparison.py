#!/usr/bin/env python3
import random
import statistics
import time

import numpy as np
import torch

from planners import create_collision_env_grid, run_astar, world_to_grid

MODEL_PATH = "/home/ubuntu/project/qlearning_model.pt"
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_STEPS_RL = 260
N_EPISODES = 60
SEED = 42


def shortest_path_len(grid, start, goal):
    path, _, _ = run_astar(grid, start, goal)
    return len(path), path


def classical_navigation_episode(grid, start, goal):
    t0 = time.perf_counter()
    path, _, _ = run_astar(grid, start, goal)

    if not path:
        return {
            "success": False,
            "steps": 0,
            "collisions": 0,
            "compute_time": time.perf_counter() - t0,
            "efficiency": 0.0,
        }

    # Controller idealise: suit exactement le chemin global.
    steps = len(path) - 1
    collisions = 0
    for r, c in path:
        if grid[r, c] == 1:
            collisions += 1

    shortest_len = len(path) - 1
    efficiency = 1.0 if shortest_len > 0 else 0.0

    return {
        "success": path[-1] == goal,
        "steps": steps,
        "collisions": collisions,
        "compute_time": time.perf_counter() - t0,
        "efficiency": efficiency,
    }


def rl_navigation_episode(grid, start, goal, q_table, state_to_idx):
    t0 = time.perf_counter()

    s = start
    collisions = 0
    visited = {s}
    steps = 0

    # Pour l'efficacite: compare avec le plus court chemin de reference.
    shortest_len, _ = shortest_path_len(grid, start, goal)

    for _ in range(MAX_STEPS_RL):
        if s == goal:
            break

        if s not in state_to_idx:
            break

        s_idx = state_to_idx[s]
        a = int(torch.argmax(q_table[s_idx]).item())

        r, c = s
        dr, dc = ACTIONS[a]
        nr, nc = r + dr, c + dc

        h, w = grid.shape
        hit = False
        if nr < 0 or nr >= h or nc < 0 or nc >= w:
            hit = True
            nr, nc = r, c
        elif grid[nr, nc] == 1:
            hit = True
            nr, nc = r, c

        if hit:
            collisions += 1

        s = (nr, nc)
        steps += 1

        # Coupe les boucles evidentes en echec.
        if s in visited and s != goal and steps > 20:
            break
        visited.add(s)

    success = s == goal

    if success and shortest_len > 1 and steps > 0:
        efficiency = (shortest_len - 1) / float(steps)
    else:
        efficiency = 0.0

    return {
        "success": success,
        "steps": steps,
        "collisions": collisions,
        "compute_time": time.perf_counter() - t0,
        "efficiency": efficiency,
    }


def summarize(name, rows):
    success_rate = 100.0 * sum(1 for r in rows if r["success"]) / len(rows)
    avg_steps = statistics.mean(r["steps"] for r in rows)
    avg_coll = statistics.mean(r["collisions"] for r in rows)
    collision_free = 100.0 * sum(1 for r in rows if r["collisions"] == 0) / len(rows)
    avg_eff = statistics.mean(r["efficiency"] for r in rows)
    avg_time_ms = 1000.0 * statistics.mean(r["compute_time"] for r in rows)

    return {
        "name": name,
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "avg_collisions": avg_coll,
        "collision_free_rate": collision_free,
        "avg_efficiency": avg_eff,
        "avg_compute_ms": avg_time_ms,
    }


def print_table(metrics):
    print("\n=== Part 6 - Performance Comparison ===")
    print("Legend: efficiency ~ shortest_path/actual_path (1.0 = optimal)")
    print(
        "{:<16} {:>10} {:>12} {:>14} {:>14} {:>12} {:>14}".format(
            "Method",
            "Success%",
            "Avg Steps",
            "Avg Coll",
            "Coll-Free%",
            "Efficiency",
            "Compute (ms)",
        )
    )
    for m in metrics:
        print(
            "{:<16} {:>10.1f} {:>12.1f} {:>14.2f} {:>14.1f} {:>12.3f} {:>14.3f}".format(
                m["name"],
                m["success_rate"],
                m["avg_steps"],
                m["avg_collisions"],
                m["collision_free_rate"],
                m["avg_efficiency"],
                m["avg_compute_ms"],
            )
        )


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    payload = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    q_table = payload["q_table"]
    free_cells_model = [tuple(c) for c in payload["free_cells"]]
    goal = tuple(payload["goal_state"])

    state_to_idx = {cell: i for i, cell in enumerate(free_cells_model)}

    grid = create_collision_env_grid().astype(int)

    # Echantillonne des starts atteignables pour une comparaison juste.
    free_cells = list(zip(*np.where(grid == 0)))
    random.shuffle(free_cells)

    starts = []
    for s in free_cells:
        if s == goal:
            continue
        plen, _ = shortest_path_len(grid, s, goal)
        if plen > 0:
            starts.append(s)
        if len(starts) >= N_EPISODES:
            break

    if not starts:
        print("Aucun start atteignable trouve.")
        return

    classical_rows = []
    rl_rows = []

    for s in starts:
        classical_rows.append(classical_navigation_episode(grid, s, goal))
        rl_rows.append(rl_navigation_episode(grid, s, goal, q_table, state_to_idx))

    metrics = [
        summarize("Classical", classical_rows),
        summarize("RL (Q-Learning)", rl_rows),
    ]

    print_table(metrics)


if __name__ == "__main__":
    main()
