import xml.etree.ElementTree as ET
import numpy as np

def update_map():
    # Parse circuit.world
    tree = ET.parse('/home/ubuntu/project/circuit.world')
    root = tree.getroot()
    
    # We want a 8x12 meters grid to give 1m padding around the 6x10 maze.
    # Bottom Left: X = -4.0, Y = -6.0
    # Top Right: X = 4.0, Y = 6.0
    # So origin (bottom left) is at X=-4.0, Y=-6.0
    
    resolution = 0.05
    width_m = 8.0
    height_m = 12.0
    width = int(width_m / resolution)   # 8 / 0.05 = 160
    height = int(height_m / resolution) # 12 / 0.05 = 240
    
    orig_x = -4.0
    orig_y = 6.0  # Top edge (since image rows go down)
    
    grid = np.ones((height, width), dtype=np.uint8) * 255
    
    def rect_to_pixels(x, y, dx, dy):
        # convert gazebo coordinate to map coordinate
        min_x = int((x - dx/2.0 - orig_x) / resolution)
        max_x = int((x + dx/2.0 - orig_x) / resolution)
        # Y goes backwards because rows go down!
        min_y = int((orig_y - (y + dy/2.0)) / resolution)
        max_y = int((orig_y - (y - dy/2.0)) / resolution)
        
        # clamp
        min_x, max_x = max(0, min_x), min(width, max_x)
        min_y, max_y = max(0, min_y), min(height, max_y)
        
        grid[min_y:max_y, min_x:max_x] = 0

    for model in root.findall('.//model'):
        name = model.get('name', 'none')
        if name == 'ground_plane': continue
        
        pose = model.find('pose')
        if pose is None: continue
        p = pose.text.split()
        cx, cy = float(p[0]), float(p[1])
        
        geom = model.find('.//collision/geometry/box/size')
        if geom is not None:
            s = geom.text.split()
            dx, dy = float(s[0]), float(s[1])
            rect_to_pixels(cx, cy, dx, dy)
            
    header = f"P5\n{width} {height}\n255\n".encode()
    with open('/home/ubuntu/project/circuit_map.pgm', 'wb') as f:
        f.write(header)
        f.write(grid.tobytes())

update_map()
print("Map circuit_map.pgm générée (8x12m) correspondante avec le circuit.")
