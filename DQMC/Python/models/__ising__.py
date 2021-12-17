from src.common.__commonFuns__ import kPSep, math


class IsingDisorder:
    N = 1
    M = 1

    def __init__(self, L, J, J0, g, g0, h, w, _BC):
        self.L = L
        self.J = J
        self.J0 = J0
        self.g = g
        self.g0 = g0
        self.h = h
        self.w = w
        self.BC = _BC
        self.directory = "results" + kPSep
        self.N = math.pow(2, L)

    def getInfo(self):
        return f'L={self.L},J0={self.J0:.2f},g={self.g:.2f},g0={self.g0:.2f},h={self.h:.2f},w={self.w:.2f}'
