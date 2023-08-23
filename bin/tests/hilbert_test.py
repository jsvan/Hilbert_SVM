import sys
sys.path.append('..')
import hilbert
from unittest import TestCase
import visualization
import numpy as np

"""

Most of these tests aren't "real" tests, they simply make a plot with points for you to visually verify...

Notes on Matplotlib:
I've wrapped matplotlib in the visualization.py file, which is mostly unnecessary. Pyplot holds some state of 
lines on axes, which can be set, altered, and then displayed on screen (with pyplot.show()). Once displayed, the underlying
state is cleared. While the plot is showing on screen, the rest of your code is blocked until it's closed. 
It is always imported as:
    from matplotlib import pyplot
Where you play with pyplot to build underlying state before .show()-ing it. 
    pyplot.plot(x=[list of x coords], y=[list of y coords], options) creates a line plot.
    pyplot.scatter(x=[list of x coords], y=[list of y coords], options) creates individual points
You can call .plot() or .scatter() or whatever many times to overlay independent items onto the plot.
Sometimes you need the actual axes object for some effects which aren't available if you only handle pyplot, so that's why 
I'm playing with 'ax'. 

Notes on import troubles:
Running this might have importing packages trouble. I'm using PyCharm, and I've set the "sources root" to bin/ which 
makes all the imports search in the correct place. If you're getting import errors, you may need to put this at the 
beginning of the file so that python knows to look:
    import sys
    sys.path.append('..')

Notes on defining points:
Notice the points aren't python lists, but numpy arrays (np.array()). They are a mathematics wrapper library on top of numerical
array data, which is needed in my other files (hilbert.py for example)  
"""
class Test_Dividing_Line(TestCase):
    def test_a(self):
        omega = hilbert.Omega()
        p, q = (np.array([0.01, 0.01]), np.array([0.31, 0.21]))
        hodgepodge = hilbert.HilbertianHodgePodge(p, q, omega)
        midpt, vanpt = hodgepodge.get_best_dividing_line()
        ax = visualization.get_ax()
        visualization.plot_simplex(ax)
        visualization.plot_points(ax, [p, q, midpt])
        ax.axline(midpt, vanpt, color='blue')
        visualization.show(ax, "hi")

    def test_dirichlet_infinite(self):
        omega = hilbert.Omega()
        while True:
            p, q = np.random.dirichlet((1,1,1), 2)
            p, q = p[:2], q[:2]
            hodgepodge = hilbert.HilbertianHodgePodge(p, q, omega)
            midpt, vanpt = hodgepodge.get_best_dividing_line()
            ax = visualization.get_ax()
            visualization.plot_simplex(ax)
            visualization.plot_points(ax, [p, q, midpt])
            ax.axline(midpt, vanpt, color='blue')
            visualization.show(ax, "hi")

    def test_dirichlet_infinite_complex_convex(self):
        omega = hilbert.Omega(vertices=[(1.4, 0.0),
                                    (1.1, 0.8),
                                    (0.6, 1.1),
                                    (0.0, 1.2),
                                    (-.5, 0.8),
                                    (-.8, 0.3),
                                    (-.7, -.2)])
        while True:
            p, q = np.random.dirichlet((1,1,1), 2)
            p, q = p[:2], q[:2]
            hodgepodge = hilbert.HilbertianHodgePodge(p, q, omega)
            midpt, vanpt = hodgepodge.get_best_dividing_line()
            ax = visualization.get_ax()
            visualization.plot_shape(ax, omega.vertices)
            visualization.plot_points(ax, [p, q, midpt], connecting_line=[p, q])
            boundaries = hodgepodge.get_boundaries()
            ax.plot([x[0] for x in boundaries[0]], [x[1] for x in boundaries[0]], color="red")
            ax.plot([x[0] for x in boundaries[1]], [x[1] for x in boundaries[1]], color='orange')
            ax.axline(midpt, vanpt, color='blue')
            visualization.show(ax, "hi")



class Test_Midpoint(TestCase):
    def test_a(self):
        omega = hilbert.Omega()
        p, q = (np.array([0.01, 0.01]), np.array([0.31, 0.21]))
        hodgepodge = hilbert.HilbertianHodgePodge(p, q, omega)
        bipt = hodgepodge.get_midpoint()
        ax = visualization.get_ax()
        visualization.plot_simplex(ax)
        visualization.plot_points(ax, [p, q, bipt])
        visualization.show(ax, "hi")

    def test_horizontal(self):
        omega = hilbert.Omega()
        p, q = (np.array([0.11, 0.11]), np.array([0.21, 0.11]))
        hodgepodge = hilbert.HilbertianHodgePodge(p, q, omega)
        bipt = hodgepodge.get_midpoint()
        ax = visualization.get_ax()
        visualization.plot_simplex(ax)
        visualization.plot_points(ax, [p, q, bipt])
        visualization.show(ax, "hi")


    def test_diagonal(self):
        omega = hilbert.Omega()
        p, q = (np.array([0.21, 0.11]), np.array([0.11, 0.21]))
        hodgepodge = hilbert.HilbertianHodgePodge(p, q, omega)
        bipt = hodgepodge.get_midpoint()
        ax = visualization.get_ax()
        visualization.plot_simplex(ax)
        visualization.plot_points(ax, [p, q, bipt])
        visualization.show(ax, "hi")





class Test_Simplex_Boundary_Line_Find(TestCase):
    def test_vertical(self):
        omega = hilbert.Omega()
        points = ([0.11,0.11], [0.11,0.21])
        hodgepodge = hilbert.HilbertianHodgePodge(*points, omega)
        boundaries = hodgepodge.get_boundaries()
        print(boundaries)
        self.assertEqual(boundaries, (([1, 0], [0, 0]),
                                      ([0, 1], [1, 0])))
        ax = visualization.get_ax()
        ax.plot([1, 0], [0, 0])
        ax.plot([0, 1], [1, 0])
        ax.plot(*boundaries[0], color='red')
        ax.plot(*boundaries[1], color='orange')
        visualization.plot_points(ax, points)
        visualization.show(ax, "results")


    def test_diagonal(self):
        omega = hilbert.Omega()
        points = ([0.21, 0.11], [0.11, 0.21])
        hodgepodge = hilbert.HilbertianHodgePodge(*points, omega)
        boundaries = hodgepodge.get_boundaries()
        print(boundaries)
        self.assertEqual(boundaries, (([1, 0], [0, 0]),
                                      ([0, 0], [0, 1])))
        ax = visualization.get_ax()
        ax.plot([1, 0], [0, 0])
        ax.plot([0, 0], [0, 1])
        ax.plot(*boundaries[0], color='red')
        ax.plot(*boundaries[1], color='orange')
        visualization.plot_points(ax, points)
        visualization.show(ax, "results")

    def test_horizontal(self):
        omega = hilbert.Omega()
        points = ([0.11, 0.11], [0.21, 0.11])
        hodgepodge = hilbert.HilbertianHodgePodge(*points, omega)
        boundaries = hodgepodge.get_boundaries()
        print(boundaries)
        self.assertEqual(boundaries, (([0, 0], [0, 1]),
                                      ([0, 1], [1, 0])))
        ax = visualization.get_ax()
        ax.plot([0, 0], [0, 1])
        ax.plot([0, 1], [1, 0])
        ax.plot(*boundaries[0], color='red')
        ax.plot(*boundaries[1], color='orange')
        visualization.plot_points(ax, points)
        visualization.show(ax, "results")


class Test_Arbitrary_Convex_Boundary_Line_Find(TestCase):
    omega = hilbert.Omega(vertices=[(1.4, 0.0),
                                    (1.1, 0.8),
                                    (0.6, 1.1),
                                    (0.0, 1.2),
                                    (-.5, 0.8),
                                    (-.8, 0.3),
                                    (-.7, -.2)])

    def test_vertical(self):
        print('vertical')
        points = ([0.11,0.11], [0.11,0.21])
        hodgepodge = hilbert.HilbertianHodgePodge(*points, self.omega)
        boundaries = hodgepodge.get_boundaries()
        print(boundaries)
        #self.assertEqual(boundaries, (([1, 0], [0, 0]),
        #                              ([0, 1], [1, 0])))
        ax = visualization.get_ax()
        visualization.plot_shape(ax, self.omega.vertices)
        ax.plot([x[0] for x in boundaries[0]], [x[1] for x in boundaries[0]], color="red")
        ax.plot([x[0] for x in boundaries[1]], [x[1] for x in boundaries[1]], color='orange')
        visualization.plot_points(ax, points)
        visualization.show(ax, "results")
        self.assertEqual(1+1,3-1)



    def test_diagonal(self):
        points = ([0.21, 0.11], [0.11, 0.21])
        hodgepodge = hilbert.HilbertianHodgePodge(*points, self.omega)
        boundaries = hodgepodge.get_boundaries()
        print(boundaries)
        ax = visualization.get_ax()
        visualization.plot_shape(ax, self.omega.vertices)
        ax.plot([x[0] for x in boundaries[0]], [x[1] for x in boundaries[0]], color="red")
        ax.plot([x[0] for x in boundaries[1]], [x[1] for x in boundaries[1]], color='orange')
        visualization.plot_points(ax, points)
        visualization.show(ax, "results")
        self.assertEqual(1+1,3-1)

    def test_horizontal(self):
        points = ([0.11, 0.11], [0.21, 0.11])
        hodgepodge = hilbert.HilbertianHodgePodge(*points, self.omega)
        boundaries = hodgepodge.get_boundaries()
        print(boundaries)
        ax = visualization.get_ax()
        visualization.plot_shape(ax, self.omega.vertices)
        ax.plot([x[0] for x in boundaries[0]], [x[1] for x in boundaries[0]], color="red")
        ax.plot([x[0] for x in boundaries[1]], [x[1] for x in boundaries[1]], color='orange')
        visualization.plot_points(ax, points)
        visualization.show(ax, "results")
        self.assertEqual(1 + 1, 3 - 1)


if __name__ == "__main__":
    Test_Dividing_Line().test_dirichlet_infinite_complex_convex()
    # Or whatever code you want
