import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
'''
Line segment collision checker inspired by: https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
'''

# Helper Functions
def ccw(A,B,C):  # Returns True if the points are counter clockwise
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def intersect(A,B,C,D):  # Checks if two line segments (A, B) and (C, D) intersect
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def is_odd(num):
    return bool(num%2)

@lru_cache(maxsize=128)
def rot_mat(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]])

@lru_cache(maxsize=128)
def rect_state_to_vertices(state):
    center, edge_length, angle = state
    vertices = np.array([[-edge_length[0]/2, -edge_length[1]/2],
                         [-edge_length[0]/2, edge_length[1]/2],
                         [edge_length[0]/2, edge_length[1]/2],
                         [edge_length[0]/2, -edge_length[1]/2],])
    vertices = (rot_mat(angle)@vertices.T).T + center

    return vertices

@lru_cache(maxsize=128)
def ordered_angles_from_polygon(polygon):
    polygon = np.array(polygon)
    angles = np.zeros(len(polygon)-1)
    lengths = np.zeros(len(polygon)-1)
    for i in range(len(polygon)-1):
        vector = polygon[i+1] - polygon[i]
        lengths[i] = np.linalg.norm(vector)
        angle = np.arctan2(vector[1], vector[0])  # Find angles of each edges
        if angle<0:  # is to help remove duplicates as all angles should be within 0, pi
            angle += np.pi
        angles[i] = angle
    return [angle for _, angle in sorted(zip(lengths, angles))]

@lru_cache(maxsize=128)
def ordered_angle_list_from_polygon(polygon):
    angles = ordered_angles_from_polygon(polygon)
    roundedAngles = np.round(angles, decimals=4)
    angleSet = set(roundedAngles)
    return [angle for angle in roundedAngles if angle in angleSet]

@lru_cache(maxsize=256)
def rotate_polygon(angle, polygon):
    polygon = np.array(polygon)
    proj = rot_mat(angle)
    return (proj@polygon.T).T

# Main Functions for Rectangle Collision Checking
def col_checker(polygon, point, radius=0):
    '''
    collision checker for convex polygon
    '''
    polygon = np.array(polygon)
    point = np.array(point)
    polygon = np.vstack((polygon, polygon[0, :]))

    angles = ordered_angle_list_from_polygon(tuple(tuple(vertex) for vertex in polygon))

    # Run through the angles
    for i, angle in enumerate(angles):
        proj = rot_mat(-angle)
        rotated_polygon = rotate_polygon(-angle, tuple(tuple(vertex) for vertex in polygon))
        rotated_point = proj@point.T

        condition = (rotated_point[0]<(np.max(rotated_polygon[:, 0])+radius)) and (rotated_point[0]>(np.min(rotated_polygon[:, 0])-radius))
        if not condition:
            return False
    
    return True

def rectangle_col_checker(state, point, radius=0):
    '''
    collision checker for a rectangle defined by the state tuple
    '''
    vertices = rect_state_to_vertices(state)

    return col_checker(tuple(tuple(vertex) for vertex in vertices), tuple(point), radius)

# Main Functions for Boundary Collision Checking
def arbitrary_col_checker(vertices, point):
    '''
    collision checker for arbitrary polygon (even concave), no radius support
    '''
    ray = np.array([0, 1000.01])
    vertices = np.vstack((vertices, vertices[0, :]))

    hits = 0
    for i in range(len(vertices)-1):
        pointXCoord = point[0]
        # if point's x coordinate is in between the x coordinates of the two vertices, check for ray intersection
        # (this only works because the ray is vertical)
        if pointXCoord < max(vertices[i,0], vertices[i+1,0]) and pointXCoord > min(vertices[i,0], vertices[i+1,0]):
            hits += intersect(vertices[i], vertices[i+1], point, point+ray)  # Adds one if intersected
    
    return is_odd(hits)

def boundary_edge_col_checker(vertices, point, radius=0):
    '''
    checks to see if point is near the specified boundary
    approximates each edge as a rectangle
    '''
    vertices = np.vstack((vertices, vertices[0, :]))
    eps = 0.01

    for i in range(len(vertices)-1):
        vector = vertices[i+1] - vertices[i]
        center = tuple(0.5*vector + vertices[i])
        edge_length = (np.linalg.norm(vector), eps)
        angle = np.arctan2(vector[1], vector[0])
        state = (center, edge_length, angle)

        if rectangle_col_checker(state, point, radius):
            return True

    return False

def is_inside_boundary(vertices, point, radius=0.25):
    if arbitrary_col_checker(vertices, point):
        return not boundary_edge_col_checker(vertices, point, radius) 
    return False

def main():
    # This code was used to test if the functionality of the collision checkers
    polygon = np.array([[0, 0],
                        [0, 1],
                        [1, 2],
                        [1, 0]])
    rect = (np.array((0, 0)), np.array((1, 2)), 0)
    # polygon = rect_state_to_vertices(rect)
    point = np.array((0.11, 0.11))
    res = is_inside_boundary(polygon, point, radius=0.1)
    print(res)

    plt.plot(polygon[:, 0], polygon[:, 1])
    plt.plot(point[0], point[1], 'o')
    plt.title(f'result is: {res}')
    plt.show()

if __name__ == '__main__':
    main()