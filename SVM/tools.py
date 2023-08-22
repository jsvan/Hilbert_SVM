from operator import truediv, floordiv
import numpy as np

COUNTER_CW = 1
CLOCKWISE = -1


def orient(p, q, r):
    """
    a   b   c
    d   e   f
    g   h   i

    1   px  py
    1   qx  qy
    1   rx  ry

    det = aei + bfg + cdh - ceg - bdi - afh
    det = ei + bf + ch - ce - bi - fh
    det = e*(i-c) + b(f-i) + h(c-f)
    det = q[x] * (r[y] - p[y]) \
        + p[x] * (q[y] - r[y]) \
        + r[x] * (p[y] - q[y])

    :return: 1 == Counter Clockwise, 0 == Straight, -1 == Clockwise
    """
    x, y = 0, 1
    return np.sign(q[x] * (r[y] - p[y])
                 + p[x] * (q[y] - r[y])
                 + r[x] * (p[y] - q[y]))


class BinarySearcher:
    """
    [mini, maxi)
    inclusive, exclusive
    """
    def __init__(self, mini, maxi, discrete=False):
        if mini > maxi:
            mini, maxi = maxi, mini
        self.mini = mini
        self.maxi = maxi
        self.discrete = discrete
        self.divisor = floordiv if self.discrete else truediv
        self.mid = self.calc_mid()

    def calc_mid(self):
        return self.divisor((self.maxi - self.mini), 2) + self.mini

    def next(self):
        return self.mid

    """
    higher is a boolean
    """
    def feedback(self, higher):
        if (self.discrete and self.mini == self.maxi) or \
                (not self.discrete and np.isclose(self.mini, self.maxi)):
            raise StopIteration

        if higher:
            self.mini = self.mid
        else:
            self.maxi = self.mid
        self.mid = self.calc_mid()