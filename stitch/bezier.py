import mrich
import bezier
import numpy as np
import json


class Bezier:

    def __init__(self, coords, curve, radius=1, radius_curve=None):
        self.coords = np.asfortranarray(coords)
        self.nodes = np.asfortranarray(coords).T
        self.curve = curve
        self.radius = radius
        self.radius_curve = radius_curve

    ### FACTORIES

    @classmethod
    def from_coords(cls, coords, radius=1, radius_curve=None):
        if isinstance(coords, str):
            return cls.from_json(coords)

        self = cls.__new__(cls)
        nodes = np.asfortranarray(coords).T
        curve = bezier.Curve.from_nodes(nodes)
        self.__init__(
            coords=coords, curve=curve, radius=radius, radius_curve=radius_curve
        )
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
    def random_profile(cls, n: int = 6, aspect: float = 5, taper: bool = True):
        x = np.linspace(0, aspect, n)
        y = np.random.random_sample(n)
        if taper:
            y[0] = 0.0
            y[-1] = 0.0
        coords = np.asfortranarray([x, y]).T
        return cls.from_coords(coords)

    @classmethod
    def from_json(cls, path):

        assert path.endswith(".json")

        with open(path, "rt") as f:
            data = json.load(f)

        coords = data["coords"]
        radius = data["radius"]
        radius_curve = None
        if "radius_curve" in data:
            radius_curve = cls.from_coords(data["radius_curve"])
        return cls.from_coords(coords, radius=radius, radius_curve=radius_curve)

    ### METHODS

    def close(self):
        """Close the curve into a smooth loop with C1 continuity at the join."""
        coords = np.array(self.coords)
        depart = 2 * coords[-1] - coords[-2]
        arrive = 2 * coords[0] - coords[1]
        self.coords = np.asfortranarray(list(coords) + [depart, arrive, coords[0]])
        self.nodes = self.coords.T
        self.curve = bezier.Curve.from_nodes(self.nodes)

    def plot(
        self,
        num_pts=256,
        nodes: bool = True,
        lines: bool | None = None,
        fill: bool | None = None,
        rails: bool = False,
        ax=None,
        color=None,
        scale: int = 100,
        **kwargs,
    ):
        import matplotlib.pyplot as plt

        if not ax:
            fig, ax = plt.subplots(
                figsize=(4, 4),
                layout="constrained",
            )

        ax.set_aspect("equal")

        color = color or (0, 0, 1)

        if lines is None and fill is None:
            if len(self) > 4:
                lines = True
                fill = False
            else:
                lines = False
                fill = True

        if fill:
            ax.fill(*(self.nodes * scale), facecolor=(0, 0, 1, 0.3))

        if nodes:
            ax.scatter(*(self.nodes * scale), c="black")

        if lines:
            ax.plot(*(self.nodes * scale), c="black")

        sampled = self.sample(num_pts=num_pts, rails=rails, scale=scale)
        xy = sampled["center"]
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=color,
            **kwargs,
        )

        if rails:
            rail_color = (*color, 0.5) if len(color) == 3 else color
            ax.plot(
                sampled["rail_left"][:, 0],
                sampled["rail_left"][:, 1],
                color=rail_color,
                linewidth=0.5,
            )
            ax.plot(
                sampled["rail_right"][:, 0],
                sampled["rail_right"][:, 1],
                color=rail_color,
                linewidth=0.5,
            )

        return ax

    def to_json(self, path):

        assert path.endswith(".json")

        data = dict(
            coords=[list(c) for c in self.coords],
            radius=self.radius,
        )

        if self.radius_curve is not None:
            data["radius_curve"] = [list(c) for c in self.radius_curve.coords]

        with open(path, "wt") as f:
            mrich.writing(path)
            json.dump(data, f, indent=2)

    def sample(self, num_pts=256, scale=1, rails=False):
        """Sample the curve and optionally compute offset rail curves.

        Parameters
        ----------
        num_pts : int
            Number of sample points along the curve.
        scale : float
            Maps the [0,1) input space to [0,scale).
        rails : bool
            If True, also compute offset curves at ±radius.

        Returns
        -------
        dict with keys:
            'center' : ndarray, shape (num_pts, 2)
            'rail_left' : ndarray or None
            'rail_right' : ndarray or None
        """
        s_vals = np.linspace(0.0, 1.0, num_pts)
        points = self.curve.evaluate_multi(s_vals)
        xy = points.T * scale

        result = dict(center=xy, rail_left=None, rail_right=None)

        if rails:
            tangents = np.gradient(xy, axis=0)
            lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
            lengths = np.where(lengths == 0, 1, lengths)
            tangents = tangents / lengths
            normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

            if self.radius_curve is not None:
                # Compute cumulative arc length, normalized to [0, 1]
                seg_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
                arc = np.concatenate([[0], np.cumsum(seg_lengths)])
                arc /= arc[-1]

                # Evaluate radius_curve at arc-length positions
                rc_points = self.radius_curve.curve.evaluate_multi(arc)
                r_values = rc_points[1]  # y-axis = radius shape
                # Scale so y-max maps to self.radius
                y_max = r_values.max()
                if y_max > 0:
                    r_values = r_values * (self.radius / y_max)
                r = r_values[:, np.newaxis]
            else:
                r = self.radius

            result["rail_left"] = xy + normals * r
            result["rail_right"] = xy - normals * r

        return result

    def to_svg(
        self,
        path,
        num_pts=256,
        stroke="black",
        scale: int = 100,
        padding=1,
        rails: bool = False,
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

        sampled = self.sample(num_pts=num_pts, scale=scale, rails=rails)
        xy = sampled["center"]

        # Fit cubic Beziers to the center curve
        beziers = fit_curve(xy, max_error)
        d_center = beziers_to_svg_path(beziers)

        paths = f'  <path d="{d_center}" fill="none" stroke="{stroke}" stroke-width="{self.radius}" />\n'

        if rails:
            d_left = beziers_to_svg_path(fit_curve(sampled["rail_left"], max_error))
            d_right = beziers_to_svg_path(fit_curve(sampled["rail_right"], max_error))

            paths += f'  <path d="{d_left}" fill="none" stroke="{stroke}" stroke-width="0.5" />\n'
            paths += f'  <path d="{d_right}" fill="none" stroke="{stroke}" stroke-width="0.5" />\n'

        # Compute viewBox from all rendered points
        if rails:
            all_pts = np.vstack([xy, sampled["rail_left"], sampled["rail_right"]])
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
