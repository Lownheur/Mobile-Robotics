#!/usr/bin/env python3
import torch
import time
from planners import create_collision_env_grid

MODEL_PATH = "/home/ubuntu/project/qlearning_model.pt"
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

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

def main():
    payload = torch.load(MODEL_PATH, map_location="cpu")
    q_table = payload["q_table"]
    free_cells = [tuple(c) for c in payload["free_cells"]]
    start_state = tuple(payload["start_state"])
    goal_state = tuple(payload["goal_state"])
    
    state_to_idx = {cell: i for i, cell in enumerate(free_cells)}
    grid = create_collision_env_grid().astype(int)

    num_episodes = 60
    success_count = 0
    max_steps = 220
    
    print(f"Lancement de {num_episodes} tests logiques (sans Gazebo/ROS)...")
    start_time = time.time()

    for episode in range(num_episodes):
        state = start_state
        steps = 0
        success = False

        while steps < max_steps:
            if state == goal_state:
                success = True
                break
            
            if state not in state_to_idx:
                break

            s_idx = state_to_idx[state]
            action_idx, target_cell = choose_valid_action(q_table[s_idx], state, grid)
            state = target_cell
            steps += 1
        
        if success:
            success_count += 1
    
    elapsed = time.time() - start_time

    print("\n====================================")
    print(f"Resultats Q-learning (Rapide / Hors ligne)")
    print(f"Succes : {success_count}/{num_episodes}")
    print(f"Taux de reussite : {(success_count/num_episodes)*100:.2f}%")
    print(f"Temps d'execution : {elapsed:.4f} secondes")
    print("====================================\n")
    print("Note importante : Sans Simulateur Physique (Gazebo), le déplacement n'a plus l'incertitude")
    print("du monde réel. L'algorithme Q-Learning fait toujours le choix optimal de la Q-table, ce qui le rend")
    print("parfaitement déterministe. Ainsi, s'il trouve la solution, il le fera 100% du temps.")

if __name__ == "__main__":
    main()
