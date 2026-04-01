#!/usr/bin/env python3
import torch
import time
import random
from planners import create_collision_env_grid

MODEL_PATH = "/home/ubuntu/project/qlearning_2point_modele.pt"
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def choose_valid_action(q_row, state, grid):
    ranked = torch.argsort(q_row, descending=True).tolist()
    r, c = state
    h, w = grid.shape

    for a in ranked:
        dr, dc = ACTIONS[a]
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
            return int(a), (nr, nc)
    return 0, state

def is_reachable(grid, start, goal):
    # Verifie par un simple BFS s'il existe physiquement un chemin
    queue = [start]
    visited = {start}
    h, w = grid.shape
    while queue:
        r, c = queue.pop(0)
        if (r, c) == goal:
            return True
        for dr, dc in ACTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return False

def main():
    try:
        payload = torch.load(MODEL_PATH, map_location="cpu")
    except FileNotFoundError:
        print(f"Fichier modèle introuvable. Lancez d'abord: python3 train_Qlearning_2point.py")
        return
        
    q_table_A = payload["q_table_A"]
    q_table_B = payload["q_table_B"]
    free_cells = [tuple(c) for c in payload["free_cells"]]
    point_A = tuple(payload["point_A"])
    point_B = tuple(payload["point_B"])
    
    state_to_idx = {cell: i for i, cell in enumerate(free_cells)}
    grid = create_collision_env_grid().astype(int)

    # Demander à l'utilisateur avant le test
    choix = input("Vers quel point voulez-vous aller ? Point A (départ classique) ou Point B (arrivée classique) ? (A/B) : ").strip().upper()
    
    if choix == 'A':
        q_table = q_table_A
        goal_state = point_A
    elif choix == 'B':
        q_table = q_table_B
        goal_state = point_B
    else:
        print("Choix invalide. Arrêt du test.")
        return

    num_episodes = 60
    success_count = 0
    max_steps = 400
    
    print(f"\n--- Lancement de {num_episodes} tests ALEATOIRES RAPIDES vers le point {choix} ---")
    start_time = time.time()

    for episode in range(num_episodes):
        # Départ aléatoire garanti possible
        while True:
            state = random.choice(free_cells)
            if is_reachable(grid, state, goal_state):
                break
        
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
    print(f"Resultats Q-learning vers Point {choix} (Hors Ligne)")
    print(f"Depart : 100% Aleatoire a chaque essai")
    print(f"Succes : {success_count}/{num_episodes}")
    print(f"Taux de reussite : {(success_count/num_episodes)*100:.2f}%")
    print(f"Temps d'execution : {elapsed:.4f} secondes")
    print("====================================\n")

if __name__ == "__main__":
    main()
