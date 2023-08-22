import numpy as np
from matplotlib import pyplot


def get_ax():
    fig = pyplot.figure()
    return fig.add_subplot()

def plot_points(ax, points=None, connecting_line=False):
    if not points:
        points = [np.array([0.1, 0.3, 0.6]), np.array([0.35, 0.2, 0.45])]  # arbitrary
    if connecting_line:
        # This draws an infinite straight line through two points
        ax.axline(points[0], points[1], color='green', linestyle="dotted")
    for p in points:
        ax.scatter(p[0], p[1])

    #add labels p and q onto the first two points
    pyplot.text(*points[0], "p")
    pyplot.text(*points[1], "q")


def plot_simplex(ax):
    triedges = [{'x': [0, 0], 'y': [0, 1]}, {'x': [0, 1], 'y': [0, 0]}, {'x': [0, 1], 'y': [1, 0]}]
    for edge in triedges:
        ax.plot(edge['x'], edge['y'], color="black")

def plot_shape(ax, vs):
    l = len(vs)
    for i, v in enumerate(vs):
        # plot a line for each edge
        x = [v[0], vs[(i + 1) % l][0]]
        y = [v[1], vs[(i + 1) % l][1]]
        ax.plot(x, y, color="gray")

"""
There must be two points in a list, with 3 dimensions. The first two dimensions will be plotted, the third must sum 
it all up to 1 so that Nielson's code works. 
Brute force style and slow. Don't care!
Default points as None will trigger auto generate.
"""
def paint_distances_from_two_3d_points(ax, points=None):
    from hilbert import nielson_dist as hdist
    if not points:
        points = [np.array([0.1, 0.3, 0.6]), np.array([0.35, 0.2, 0.45])]  # arbitrary
    x, y = 0, 0
    point = np.zeros(3)
    while x < 1:
        x += 0.015
        print(f'\r{x * 100} / 100', end='')
        point[0] = x
        while y + x < 1:
            y += 0.015
            point[1] = y
            point[2] = 1 - (point[0] + point[1])
            try:
                co = 1 / (1 + min(hdist(point, points[0]), hdist(point, points[1])))
                ax.scatter(x, y, color=(0.5, co, co), marker='square')
            except (ValueError, ZeroDivisionError):
                continue
        y = 0
        point[1] = y
    return points

def show(ax, title):
    # title doesnt work
    # ax works but whatever it was from testing i haven't fixed
    pyplot.title = title
    pyplot.show()