import numpy as np
from itertools import permutations

def cost_function(color, position, robot_pos):
        Vx = 1 # velocity of x in box/sec
        Vy = 5 # velocity of y in box/sec

        if color == "red":
            offset = -1
        elif color == "blue":
            offset = 1
        else:
            offset = 0

        cost = np.abs(position[0] - robot_pos[0]) * Vx + np.abs(position[1] + offset - robot_pos[1]) * Vy
        return cost, [position[0], position[1] + offset]

print(cost_function("green", [2,2], [2,2]))
def BoxPath(robot_init ,BoxColor):
    # Init min Cost
    min_cost = float('inf')

    # create box and position of box in index
    Box = [[BoxColor[i][j], [i, j]] for i in range(3) for j in range(3)]

    # create permutation of possible way to pick box
    color_permutations = permutations(Box,3)

    for perm in color_permutations:
        # use only without duplicate color
        if all(perm[i][0] != perm[j][0] for i in range(2) for j in range(i+1, 3)):

            # Init path cost
            RobotPath = [robot_init]
            Cost = []

            # loop for 3 box picking
            for i in range(3):
                cost, robot_pos = cost_function(perm[i][0], perm[i][1], RobotPath[i])
                RobotPath.append(robot_pos)
                Cost.append(cost)
                
   
            # Goal state check
            if sum(Cost) < min_cost:
                min_cost = sum(Cost)
                CostA = Cost
                min_path = RobotPath[1:]
                color = perm[0][0],perm[1][0],perm[2][0]

    return min_path, color, min_cost, CostA

boxc = [['green', 'blue', 'green'], ['red', 'red', 'blue'], ['red', 'blue', 'green']]

path, color, cost, Sort = BoxPath([2,1], boxc)
print(path, color, cost, Sort)