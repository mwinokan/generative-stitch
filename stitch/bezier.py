import bezier
import numpy as np
import matplotlib.pyplot as plt
from python_tsp.distances import euclidean_distance_matrix
from python_tsp.heuristics import solve_tsp_local_search


class Bezier:

    def __init__(self, coords, curve):
        self.coords = np.asfortranarray(coords)
        self.nodes = np.asfortranarray(coords).T
        self.curve = curve

    @classmethod
    def from_coords(cls, coords):
        self = cls.__new__(cls)
        nodes = np.asfortranarray(coords).T
        curve = bezier.Curve.from_nodes(nodes)
        self.__init__(coords=coords, curve=curve)
        return self

    @classmethod
    def random(cls, n: int = 5, optimise: bool = True):
        x = np.random.random_sample(n)
        y = np.random.random_sample(n)
        coords = np.asfortranarray([x, y]).T
        if optimise:
            coords = optimise_coords(coords)
        return cls.from_coords(coords)

    def plot(
        self,
        num_pts=256,
        nodes: bool = True,
        lines: bool = True,
        ax=None,
        **kwargs,
    ):

        if not ax:
            fig, ax = plt.subplots(
                figsize=(4, 4),
                layout="constrained",
            )

        if nodes:
            ax.scatter(*self.nodes)

        if lines:
            ax.plot(*self.nodes)

        return self.curve.plot(
            num_pts=num_pts,
            ax=ax,
            **kwargs,
        )


def optimise_coords(coords):
    coords = np.array(coords)
    distance_matrix = euclidean_distance_matrix(coords)
    permutation, distance = solve_tsp_local_search(distance_matrix)
    return coords[permutation]
