import mrich
import bezier
import numpy as np


class Bezier:

    def __init__(self, coords, curve):
        self.coords = np.asfortranarray(coords)
        self.nodes = np.asfortranarray(coords).T
        self.curve = curve

    @classmethod
    def from_coords(cls, coords):
        if isinstance(coords, str):
            from pandas import read_csv

            df = read_csv(coords)
            coords = df[["x", "y"]].values

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
        lines: bool | None = None,
        fill: bool | None = None,
        ax=None,
        color=None,
        **kwargs,
    ):
        import matplotlib.pyplot as plt

        if not ax:
            fig, ax = plt.subplots(
                figsize=(4, 4),
                layout="constrained",
            )

        color = color or (0, 0, 1)

        if lines is None and fill is None:
            if len(self) > 4:
                lines = True
                fill = False
            else:
                lines = False
                fill = True

        if fill:
            ax.fill(*self.nodes, facecolor=(0, 0, 1, 0.3))

        if nodes:
            ax.scatter(*self.nodes, c="black")

        if lines:
            ax.plot(*self.nodes, c="black")

        return self.curve.plot(
            num_pts=num_pts,
            ax=ax,
            color=color,
            **kwargs,
        )

    def write_coords(self, path):
        from pandas import DataFrame

        assert path.endswith(".csv")

        df = DataFrame(dict(x=self.nodes[0], y=self.nodes[1]))

        mrich.writing(path)
        df.to_csv(path, index=False)

    def __len__(self):
        return len(self.nodes[0])


def optimise_coords(coords):
    from python_tsp.distances import euclidean_distance_matrix
    from python_tsp.heuristics import solve_tsp_local_search

    coords = np.array(coords)
    distance_matrix = euclidean_distance_matrix(coords)
    permutation, distance = solve_tsp_local_search(distance_matrix)
    return coords[permutation]
