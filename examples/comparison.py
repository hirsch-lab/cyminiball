"""
Compare different implementations of miniball:

1. cyminiball:   https://pypi.org/project/cyminiball/
2. miniballcpp:  https://pypi.org/project/MiniballCpp/
3. miniball:     https://pypi.org/project/miniball/

Since miniball and miniballcpp have the same package name (miniball), they
cannot be used side by side. The former is very slow, hence it is excluded here.
"""

import timeit
import numpy as np
import miniball     # miniballcpp or miniball
import cyminiball
import matplotlib.pyplot as plt


################################################################################
def create_data(n, d, dt):
    points = np.random.randn(n, d)
    points = points.astype(dt)
    return points


################################################################################
def run_timer(statement, context):
    timer = timeit.Timer(statement, globals=context)
    n_reps, delta = timer.autorange()
    return delta/n_reps, n_reps


################################################################################
def measure_performance(d=2, dt=np.float64, n_steps=10):
    a_min = 3
    a_max = 7
    ns = np.logspace(a_min, a_max, n_steps).astype(int)
    t1 = np.zeros(len(ns))
    t2 = np.zeros(len(ns))
    statement1 = "cyminiball.compute(points)"
    statement2 = "miniball.Miniball(points)"
    #statement2 = "miniball.get_bounding_ball(points)"

    print("Running...")

    for i,n in enumerate(ns):
        points = create_data(n=n, d=d, dt=dt)
        context = dict(miniball=miniball, cyminiball=cyminiball, points=points)
        delta1, n_reps1 = run_timer(statement=statement1, context=context)
        t1[i] = delta1
        delta2, n_reps2 = run_timer(statement=statement2, context=context)
        t2[i] = delta2
        print(f"%2d/%2d (n: %{a_max+1}d, d: %d, n_reps: %d, %d)"
              % (i+1, len(ns), n, d, n_reps1, n_reps2))

    ratio = t2/t1
    print("Done!")
    print()
    print("ratio:  %.2f ± %.2f" % (ratio.mean(), ratio.std()))
    print("ratios: %s" % ", ".join(map(str, ratio.round(1))))
    return ns, t1, t2


################################################################################
def plot_results(ns, t1, t2, d, dt):
    fig, ax = plt.subplots()
    ax.plot(ns, t1*1000, "-o", label="cyminiball")
    ax.plot(ns, t2*1000, "-x", label="miniball")
    ax.set_xlabel("Number of points")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Point cloud: d=%d, type=%s" % (d, dt.__name__))
    ax.legend()
    ax.grid(True)
    ax.set_yscale("log")
    ax.set_xscale("log")
    plt.show()


################################################################################
if __name__ == "__main__":
    d = 2
    dt = np.float64
    n_steps = 10
    ns, t1, t2 = measure_performance(d=d, n_steps=n_steps)
    plot_results(ns=ns, t1=t1, t2=t2, d=d, dt=dt)
