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
        rails: bool = True,
        max_error=0.001,
    ):
        """Write the curve as an SVG with fitted cubic Beziers.

        Parameters
        ----------
        num_pts : int
            Number of sample points used for curve fitting.
        scale : int
            Maps the [0,1) input space to [0,scale) in the SVG.
        rails : bool
            If True, include offset curves (locus) at ±radius from the center.
        max_error : float
            Maximum squared error for Bezier fitting (in scaled coords).
        """
        from .fitcurves import beziers_to_svg_path, fit_curve

        assert path.endswith(".svg")

        # Sample the high-degree curve and scale to output space
        s_vals = np.linspace(0.0, 1.0, num_pts)
        points = self.curve.evaluate_multi(s_vals)
        xy = points.T * scale  # shape (num_pts, 2)

        # Fit cubic Beziers to the center curve
        beziers = fit_curve(xy, max_error)
        d_center = beziers_to_svg_path(beziers)

        paths = f'  <path d="{d_center}" fill="none" stroke="{stroke}" stroke-width="{self.radius}" />\n'

        if rails:
            # Compute unit normals at each sample point
            tangents = np.gradient(xy, axis=0)
            lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
            lengths = np.where(lengths == 0, 1, lengths)
            tangents = tangents / lengths
            normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

            r = self.radius

            # Offset curves
            rail_left = xy + normals * r
            rail_right = xy - normals * r

            d_left = beziers_to_svg_path(fit_curve(rail_left, max_error))
            d_right = beziers_to_svg_path(fit_curve(rail_right, max_error))

            paths += f'  <path d="{d_left}" fill="none" stroke="{stroke}" stroke-width="0.5" />\n'
            paths += f'  <path d="{d_right}" fill="none" stroke="{stroke}" stroke-width="0.5" />\n'

        # Compute viewBox from all rendered points
        if rails:
            all_pts = np.vstack([xy, rail_left, rail_right])
        else:
            all_pts = xy
        min_x, min_y = all_pts.min(axis=0)
        max_x, max_y = all_pts.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y

        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="{min_x - padding} {min_y - padding} '
            f'{width + 2 * padding} {height + 2 * padding}">\n'
            f"{paths}"
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
