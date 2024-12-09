# import pdb
from helpers import normalize, blur

def initialize_beliefs(grid):
    height = len(grid)
    width = len(grid[0])
    area = height * width
    belief_per_cell = 1.0 / area
    beliefs = []
    for i in range(height):
        row = []
        for j in range(width):
            row.append(belief_per_cell)
        beliefs.append(row)
    return beliefs

# The sense function is in principle the same as for the 1D world, only exception is we iterate through a 2d list.
def sense(color, grid, beliefs, p_hit, p_miss):    
    new_beliefs = []

    for y, row in enumerate(beliefs):
        new_row = []
        for x, belief in enumerate(row):
            hit = (color == grid[y][x])
            if hit:
                new_belief = p_hit * belief
            else:
                new_belief = p_miss * belief
            #new_belief = belief * ((1 - p_miss) + hit * p_hit)
        
            new_row.append(new_belief)
    
        new_beliefs.append(new_row)
    
    new_beliefs = normalize(new_beliefs)
            
    return new_beliefs

def move(dy, dx, beliefs, blurring):
    height = len(beliefs)
    width = len(beliefs[0])
    new_G = [[0.0 for i in range(width)] for j in range(height)]
    for i, row in enumerate(beliefs):
        for j, cell in enumerate(row):
            new_i = (i + dy) % height  # corrected width to height for correct dimension wrapping
            new_j = (j + dx) % width   # corrected height to width for correct dimension wrapping
            # pdb.set_trace()
            new_G[int(new_i)][int(new_j)] = cell
    return blur(new_G, blurring)
