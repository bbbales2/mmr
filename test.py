#%%

import numpy
import scipy
import bisect
import matplotlib.pyplot as plt
import scipy.stats

N = 200

pool = numpy.ones(N) * 3000
tpool = numpy.array([1500] * (N / 2) + [4500] * (N / 2))

means = []
stds = []

for r in range(1000000):
    p = numpy.random.choice(range(N))

    probs = scipy.stats.cauchy.pdf(pool, pool[p], 50)

    g = set()
    while len(g) < 10:
        sums = numpy.cumsum(probs)

        po = bisect.bisect_left(sums, numpy.random.rand() * sums[-1])

        if po not in g:
            g.add(po)

            probs[po] = 0

    g = numpy.array(list(g))
    numpy.random.shuffle(g)

    team1 = sum(tpool[g[:len(g) / 2]])
    team2 = sum(tpool[g[len(g) / 2:]])

    rand = numpy.random.rand() * (team1 + team2)

    if rand < team1:
        for po in g[:len(g) / 2]:
            pool[po] += 25

        for po in g[len(g) / 2:]:
            pool[po] -= 25
    else:
        for po in g[:len(g) / 2]:
            pool[po] -= 25

        for po in g[len(g) / 2:]:
            pool[po] += 25

    #print g
    #print pool
    if r % 10000 == 0:
        plt.hist(pool)
        plt.title(r)
        plt.show()

        means.append(numpy.mean(pool))
        stds.append(numpy.std(pool))

        plt.plot(stds)
        plt.show()