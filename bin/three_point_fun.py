import hilbert
from matplotlib import pyplot
import visualization
from numpy.random import dirichlet

def straight_line_through_three_points(p1, p2, q):
    omega = hilbert.Omega()
    pq1 = hilbert.HilbertianHodgePodge(p1, q, omega)
    pq2 = hilbert.HilbertianHodgePodge(p2, q, omega)

    m1 = pq1.get_midpoint()
    m2 = pq2.get_midpoint()
    ax = visualization.get_ax()
    visualization.plot_simplex(ax)
    visualization.plot_points(ax, [p1, p2, q], ['p1', 'p2', 'q'], ['blue', 'blue', 'red'], connecting_line=[m1, m2])
    visualization.show(ax, 'hi')



if __name__ == "__main__":
    while True:
        straight_line_through_three_points(*[x[:2] for x in dirichlet([1, 1, 1], 3)])

