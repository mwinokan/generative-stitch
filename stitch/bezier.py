import bezier
import numpy as np
import matplotlib.pyplot as plt


class Bezier:

    def __init__(self, coords, curve):
        self.nodes = np.asfortranarray(coords).T
        self.curve = curve

    @classmethod
    def from_nodes(cls, coords):
        self = cls.__new__(cls)
        nodes = np.asfortranarray(coords).T
        curve = bezier.Curve.from_nodes(nodes)
        self.__init__(coords=coords, curve=curve)
        return self

    def plot(
        self, num_pts=256, nodes: bool = True, lines: bool = True, ax=None, **kwargs
    ):

        if not ax:
            fig, ax = plt.subplots(figsize=(4, 4), layout="constrained")

        if nodes:
            ax.scatter(self.nodes[0], self.nodes[1])

        if lines:
            ax.plot(self.nodes[0], self.nodes[1])

        return self.curve.plot(num_pts=num_pts, ax=ax, **kwargs)
