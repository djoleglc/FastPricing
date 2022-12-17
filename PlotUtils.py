from matplotlib import cm
import matplotlib.pyplot as plt


def plot3d(x, y, z, xlab, ylab, zlab, title, view=(25, 30)):
    # we can also visualize the price
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection="3d")
    ax.view_init(view[0], view[1])
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.set_zlabel(zlab)

    ax.set_xlim(x.max() * 1.1, x.min() * 0.90)
    ax.set_ylim(y.max() * 1.1, y.min() * 0.9)
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=1, antialiased=True)
