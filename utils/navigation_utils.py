import math 

def euclidean_distance(pos1, pos2):
    return math.sqrt((pos1['x'] - pos2['x']) ** 2 + (pos1['y'] - pos2['y']) ** 2 + (pos1['z'] - pos2['z']) ** 2)

def generate_neighbors(pos, step_size):
    x, y, z = pos['x'], pos['y'], pos['z']
    neighbors = []
    for angle in range(0, 360, 45):
        dx = step_size * math.cos(math.radians(angle))
        dz = step_size * math.sin(math.radians(angle))
        neighbors.append({'x': x + dx, 'y': y, 'z': z + dz})
    return neighbors

def is_valid_transition(pos1, pos2, gridSize=0.25):
    threshold = math.sqrt(gridSize**2 + gridSize**2)
    if (euclidean_distance(pos1, pos2) < threshold):
        return True
    return False