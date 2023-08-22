from matplotlib import pyplot
import visualization as viz

fig = pyplot.figure()
ax = fig.add_subplot()


viz.plot_simplex(ax)
viz.plot_two_points(ax)

pyplot.show()




