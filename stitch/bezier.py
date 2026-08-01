import mrich
import bezier
import numpy as np
import json


class Bezier:

    def __init__(self, coords, curve, radius=1):
        self.coords = np.asfortranarray(coords)
        self.nodes = np.asfortranarray(coords).T
        self.curve = curve
        self.radius = radius

    ### FACTORIES

    @classmethod
    def from_coords(cls, coords, radius=1):
        if isinstance(coords, str):
            return cls.from_json(coords)

        self = cls.__new__(cls)
        nodes = np.asfortranarray(coords).T
        curve = bezier.Curve.from_nodes(nodes)
        self.__init__(coords=coords, curve=curve, radius=radius)
        return self

    @classmethod
    def random(cls, n: int = 5, optimise: bool = True):
        x = np.random.random_sample(n)
        y = np.random.random_sample(n)
        coords = np.asfortranarray([x, y]).T
        if optimise:
            coords = optimise_coords(coords)
        return cls.from_coords(coords)

    @classmethod
    def from_json(cls, path):

        assert path.endswith(".json")

        with open(path, "rt") as f:
            data = json.load(f)

        coords = data["coords"]
        radius = data["radius"]
        return cls.from_coords(coords, radius=radius)

    ### METHODS

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

    def to_json(self, path):

        assert path.endswith(".json")

        data = dict(
            coords=[list(c) for c in self.coords],
            radius=self.radius,
        )

        with open(path, "wt") as f:
            mrich.writing(path)
            json.dump(data, f, indent=2)

    def to_svg(
        self,
        path,
        num_pts=256,
        stroke="black",
        scale: int = 100,
        padding=1,
        max_error=0.001,
    ):
        """Write the curve as an SVG with fitted cubic Beziers.

        Parameters
        ----------
        num_pts : int
            Number of sample points used for curve fitting.
        scale : int
            Maps the [0,1) input space to [0,scale) in the SVG.
        max_error : float
            Maximum squared error for Bezier fitting (in scaled coords).
        """
        from .fitcurves import beziers_to_svg_path, fit_curve

        assert path.endswith(".svg")

        # Sample the high-degree curve and scale to output space
        s_vals = np.linspace(0.0, 1.0, num_pts)
        points = self.curve.evaluate_multi(s_vals)
        xy = points.T * scale  # shape (num_pts, 2)

        # Fit cubic Beziers to the sampled points
        beziers = fit_curve(xy, max_error)
        d = beziers_to_svg_path(beziers)

        # Compute viewBox from sampled points
        xs, ys = xy[:, 0], xy[:, 1]
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        width = max_x - min_x
        height = max_y - min_y

        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="{min_x - padding} {min_y - padding} '
            f'{width + 2 * padding} {height + 2 * padding}">\n'
            f'  <path d="{d}" fill="none" stroke="{stroke}" stroke-width="{self.radius}" />\n'
            f"</svg>\n"
        )

        mrich.writing(path)
        with open(path, "w") as f:
            f.write(svg)

    ### DUNDERS

    def __len__(self):
        return len(self.nodes[0])


def optimise_coords(coords):
    from python_tsp.distances import euclidean_distance_matrix
    from python_tsp.heuristics import solve_tsp_local_search

    coords = np.array(coords)
    distance_matrix = euclidean_distance_matrix(coords)
    permutation, distance = solve_tsp_local_search(distance_matrix)
    return coords[permutation]
