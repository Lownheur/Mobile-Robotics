import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import argparse
from planners import create_gazebo_env_grid, world_to_grid, run_dijkstra, run_greedy, run_astar

def animate_path_planning(save_only=False):
    grid = create_gazebo_env_grid()
    
    start_world = (0.0, -4.0)
    goal_world = (0.0, 4.0)
    
    start = world_to_grid(*start_world)
    goal = world_to_grid(*goal_world)
    
    # Force start and goal to be clear
    r_s, c_s = start
    r_g, c_g = goal
    grid[max(0, r_s-1):min(grid.shape[0], r_s+2), max(0, c_s-1):min(grid.shape[1], c_s+2)] = 0
    grid[max(0, r_g-1):min(grid.shape[0], r_g+2), max(0, c_g-1):min(grid.shape[1], c_g+2)] = 0

    print("Running Dijkstra...")
    path_d, exp_d, time_d = run_dijkstra(grid, start, goal)
    print("Running Greedy...")
    path_g, exp_g, time_g = run_greedy(grid, start, goal)
    print("Running A*...")
    path_a, exp_a, time_a = run_astar(grid, start, goal)

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    fig.canvas.manager.set_window_title("Path Planning Algorithms (Circuit)")

    titles = [
        f"Dijkstra\nPath points: {len(path_d)} | Explored: {len(exp_d)} | Time: {time_d:.4f}s",
        f"Greedy\nPath points: {len(path_g)} | Explored: {len(exp_g)} | Time: {time_g:.4f}s",
        f"A*\nPath points: {len(path_a)} | Explored: {len(exp_a)} | Time: {time_a:.4f}s"
    ]
    
    for i in range(3):
        axs[i].imshow(grid, cmap='Greys')
        axs[i].plot(start[1], start[0], 'go', markersize=8, label="Départ")
        axs[i].plot(goal[1], goal[0], 'ro', markersize=8, label="Arrivée")
        axs[i].set_title(titles[i])
        axs[i].axis('off')
        
    line_d, = axs[0].plot([], [], 'b.', markersize=2)
    line_g, = axs[1].plot([], [], 'b.', markersize=2)
    line_a, = axs[2].plot([], [], 'b.', markersize=2)
    
    path_line_d, = axs[0].plot([], [], 'r-', linewidth=2)
    path_line_g, = axs[1].plot([], [], 'r-', linewidth=2)
    path_line_a, = axs[2].plot([], [], 'r-', linewidth=2)

    max_frames = max(len(exp_d), len(exp_g), len(exp_a))
    # Slightly faster animation than previous version.
    steps_per_frame = 6

    def init():
        return line_d, line_g, line_a, path_line_d, path_line_g, path_line_a

    def update(frame):
        idx = frame * steps_per_frame
        
        # Dijkstra
        idx_dt = min(idx, len(exp_d) - 1)
        if idx_dt >= 0:
            pts = list(zip(*exp_d[:idx_dt+1]))
            if pts: line_d.set_data(pts[1], pts[0])
            if idx_dt == len(exp_d) - 1 and path_d:
                ppts = list(zip(*path_d))
                path_line_d.set_data(ppts[1], ppts[0])
                
        # Greedy
        idx_gt = min(idx, len(exp_g) - 1)
        if idx_gt >= 0:
            pts = list(zip(*exp_g[:idx_gt+1]))
            if pts: line_g.set_data(pts[1], pts[0])
            if idx_gt == len(exp_g) - 1 and path_g:
                ppts = list(zip(*path_g))
                path_line_g.set_data(ppts[1], ppts[0])

        # A*
        idx_at = min(idx, len(exp_a) - 1)
        if idx_at >= 0:
            pts = list(zip(*exp_a[:idx_at+1]))
            if pts: line_a.set_data(pts[1], pts[0])
            if idx_at == len(exp_a) - 1 and path_a:
                ppts = list(zip(*path_a))
                path_line_a.set_data(ppts[1], ppts[0])
                
        return line_d, line_g, line_a, path_line_d, path_line_g, path_line_a

    # Ensure last explored node is always reached (important for long Dijkstra runs).
    num_frames = ((max_frames + steps_per_frame - 1) // steps_per_frame) + 1
    ani = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=100, repeat=False)

    project_dir = Path(__file__).resolve().parent
    gif_path = project_dir / "path_planning_animation.gif"
    mp4_path = project_dir / "path_planning_animation.mp4"

    print(f"Saving GIF to: {gif_path}")
    gif_writer = animation.PillowWriter(fps=8)
    ani.save(str(gif_path), writer=gif_writer)
    print("GIF saved.")

    # Optional MP4 export if ffmpeg is available.
    try:
        ffmpeg_writer = animation.FFMpegWriter(fps=8, bitrate=1800)
        ani.save(str(mp4_path), writer=ffmpeg_writer)
        print(f"MP4 saved to: {mp4_path}")
    except Exception:
        print("FFmpeg not available, MP4 export skipped.")
    
    plt.tight_layout()
    if save_only:
        plt.close(fig)
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-only", action="store_true", help="Save animation files without opening window")
    args = parser.parse_args()
    animate_path_planning(save_only=args.save_only)

    