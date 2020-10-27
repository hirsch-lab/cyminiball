import context
import miniball
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

################################################################################
def generate_data(n=50):
    d = 2           # Number of dimensions
    dt = np.float   # Different data types are supported
    rs = np.random.RandomState(42)
    points = rs.randn(n, d)
    points = points.astype(dt)
    return points

################################################################################
def visualize_data(ax, points, lim=5):
    ax.plot(points[:,0], points[:,1], "kx")
    ax.axis("square")
    ax.grid("on")
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_title("miniball example")

################################################################################
def visualize_circle(ax, info, points):
    radius = info["radius"]
    center = info["center"]
    hp = ax.plot(center[0], center[1], "ob-")
    hl = ax.plot(points[info["support"],0],
                 points[info["support"],1], "ro-")
    hc = plt.Circle(center, radius, fill=False, color="blue")
    ax.add_artist(hc)
    if "ids_max" in info:
        hm = ax.plot(points[info["ids_max"],0],
                     points[info["ids_max"],1], "mo-")
    return hp, hl, hc

################################################################################
def example_basic():
    points = generate_data()
    points[0] = [3, 0]
    points[-1] = [-2, -3]

    C, r2, info = miniball.compute(points, details=True)
    info = miniball.compute_max_chord(points=points, info=info)
    print("Center:   %s" % C)
    print("Radius:   %.3f" % np.sqrt(r2))
    print("Info:")
    pprint(info, indent=4)

    # Visualize.
    fig, ax = plt.subplots()
    visualize_data(ax, points)
    visualize_circle(ax, info, points)

################################################################################
def example_animated():
    # Data.
    points = generate_data(100)
    points[0] = [3, 0]
    points[-1] = [-2, -3]
    xrange = np.linspace(-4, 4, 120)

    # Set up animation.
    fig, ax = plt.subplots()
    visualize_data(ax, points, lim=7)
    _, _, info = miniball.compute(points, details=True)
    center, line, circle = visualize_circle(ax, info, points)
    circle = circle         # Circle artist
    center = center[0]      # Line2D artist
    line = line[0]          # Line2D artist
    point = ax.plot(0,0, 'gx-')[0] # line2D artist
    ax.legend((center, line, point),
              ("bounding circle", "support", "moving point"))

    def init():
        return circle, center, line, point

    def update(x):
        # Update x coordinate of last point.
        points[-1,0] = x
        # Re-compute miniball.
        C, r2, info = miniball.compute(points, details=True)
        # Update artists.
        circle.center = C
        circle.radius = np.sqrt(r2)
        center.set_data(C)
        line.set_data(points[info["support"],:].T)
        point.set_data([[xrange[0], points[-1,0]],
                        [points[-1,1], points[-1,1]]])
        return circle, center, line, point

    from matplotlib.animation import FuncAnimation
    ani = FuncAnimation(fig, update, frames=xrange, interval=30,
                        init_func=init, blit=True)

################################################################################
if __name__ == "__main__":
    example_basic()
    example_animated()
    plt.show()
