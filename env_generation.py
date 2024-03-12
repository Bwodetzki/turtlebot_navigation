'''
generate_polygon(), random_angle_steps(), clip(), and visualize() taken from
https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon
'''

import math, random
from typing import List, Tuple
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt


# Functions
def generate_polygon(center: Tuple[float, float], avg_radius: float,
                     irregularity: float, spikiness: float,
                     num_vertices: int) -> List[Tuple[float, float]]:
    """
    Start with the center of the polygon at center, then creates the
    polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.

    Args:
        center (Tuple[float, float]):
            a pair representing the center of the circumference used
            to generate the polygon.
        avg_radius (float):
            the average radius (distance of each generated vertex to
            the center of the circumference) used to generate points
            with a normal distribution.
        irregularity (float):
            variance of the spacing of the angles between consecutive
            vertices.
        spikiness (float):
            variance of the distance of each vertex to the center of
            the circumference.
        num_vertices (int):
            the number of vertices of the polygon.
    Returns:
        List[Tuple[float, float]]: list of vertices, in CCW order.
    """
    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity)

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        radius = clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
        point = np.array((center[0] + radius * math.cos(angle),
                 center[1] + radius * math.sin(angle)))
        points.append(point)
        angle += angle_steps[i]

    return np.array(points)

def random_angle_steps(steps: int, irregularity: float) -> List[float]:
    """Generates the division of a circumference in random angles.

    Args:
        steps (int):
            the number of angles to generate.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
    Returns:
        List[float]: the list of the random angles.
    """
    # generate n angle steps
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle

    # normalize the steps so that point 0 and point n+1 are the same
    cumsum /= (2 * math.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles

def clip(value, lower, upper):
    """
    Given an interval, values outside the interval are clipped to the interval
    edges.
    """
    return min(upper, max(value, lower))

def visualize(vertices):
    black = (0, 0, 0)
    white = (255, 255, 255)
    img = Image.new('RGB', (500, 500), white)
    im_px_access = img.load()
    draw = ImageDraw.Draw(img)

    # either use .polygon(), if you want to fill the area with a solid colour
    draw.polygon(vertices, outline=black, fill=white)

    # or .line() if you want to control the line thickness, or use both methods together!
    # draw.line(vertices + [vertices[0]], width=2, fill=black)

    img.show()

    # now you can save the image (img), or do whatever else you want with it.

def generate_obstacles(center_bounds=[100, 100], edge_len_bounds=[1, 20], seed=1, n=20):
    bounding_box = np.array([[0, 0], center_bounds])  # [center, edge_lengths]
    np.random.seed(seed)
    centers = (2*np.random.rand(n, 2)-1)*bounding_box[1, :]
    edge_lengths = np.random.randint(edge_len_bounds[0], edge_len_bounds[1], size=(n, 2))
    angles = (2*np.random.rand(n) - 1)*np.pi

    return (centers, edge_lengths, angles)

def plot_obstacle(center, edge_lengths, angle):
    x_len = edge_lengths[0]
    y_len = edge_lengths[1]

    x_cen = center[0]
    y_cen = center[1]
    
    obs = np.array([[-x_len, -y_len],
                    [-x_len, y_len],
                    [x_len, y_len],
                    [x_len, -y_len]])
    
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    obs = (rot_mat@obs.T).T + np.array([x_cen, y_cen])
    
    plt.plot(obs[:, 0], obs[:, 1], 'g')
    plt.plot([obs[0, 0], obs[-1, 0]], [obs[0, 1], obs[-1, 1]], 'g')


def main():
    center = (0,0)
    avg_radius = 100
    irregularity = 1.0
    spikiness = 0.4
    num_vertices = 10
    vertices = generate_polygon(center, avg_radius, irregularity, spikiness, num_vertices)

    # This is our barrier
    plt.plot(vertices[:, 0], vertices[:, 1], 'b')
    plt.plot([vertices[0, 0], vertices[-1, 0]], [vertices[0, 1], vertices[-1, 1]], 'b')

    # These are our obstacles
    centers, edge_lengths, angles = generate_obstacles()
    for center, edge_length, angle in zip(centers, edge_lengths, angles):
        plot_obstacle(center, edge_length, angle)

    
    plt.show()




# Main code
if __name__=='__main__':
    main()