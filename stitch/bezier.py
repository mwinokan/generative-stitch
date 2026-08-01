import mrich
import bezier
import numpy as np
import json


class CompoundBezier:

    def __init__(self, curves=None, radius=1, radius_curve=None, connect: bool = True):
        self.curves = curves or []
        self.radius = radius
        self.radius_curve = radius_curve
        self.connect = connect

    ### FACTORIES

    @classmethod
    def from_json(cls, path):

        assert path.endswith(".json")

        with open(path, "rt") as f:
            data = json.load(f)

        curves = []
        for curve in data["curves"]:
            curves.append(
                Bezier.from_coords(
                    curve["coords"],
                    radius=curve["radius"],
                    radius_curve=curve["radius_curve"],
                )
            )

        radius = data["radius"]
        connect = data["connect"]

        radius_curve = None
        if "radius_curve" in data:
            radius_curve = Bezier.from_coords(data["radius_curve"])

        self = cls.__new__(cls)
        self.__init__(curves, radius=radius, radius_curve=radius_curve, connect=connect)

        return self

    ### METHODS

    def append(self, curve):
        self.curves.append(curve)

    def to_json(self, path):

        assert path.endswith(".json")

        data = dict(
            curves=[curve.to_dict() for curve in self.curves],
            radius=self.radius,
            connect=self.connect,
        )

        if self.radius_curve is not None:
            data["radius_curve"] = [list(c) for c in self.radius_curve.coords]

        with open(path, "wt") as f:
            mrich.writing(path)
            json.dump(data, f, indent=2)

    ### METHODS

    def sample(self, num_pts=256, scale=1, rails=False):
        """Sample the compound curve with straight-line connectors between sub-curves.

        Points are distributed proportional to arc length across curves and connectors.
        """
        if not self.curves:
            empty = np.empty((0, 2))
            return dict(center=empty, rail_left=None, rail_right=None)

        # Estimate arc lengths for each segment (curves + connectors)
        ESTIMATE_PTS = 32
        arc_lengths = []
        curve_endpoints = []  # (start_xy, end_xy) per curve, scaled

        for curve in self.curves:
            s = np.linspace(0.0, 1.0, ESTIMATE_PTS)
            pts = curve.curve.evaluate_multi(s).T * scale
            arc = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
            arc_lengths.append(("curve", arc))
            curve_endpoints.append((pts[0], pts[-1]))

        # Connector lengths
        connector_lengths = []
        for i in range(len(self.curves) - 1):
            end_i = curve_endpoints[i][1]
            start_next = curve_endpoints[i + 1][0]
            connector_lengths.append(np.linalg.norm(start_next - end_i))

        # Build ordered segment list: curve, connector, curve, connector, ...
        segments = []  # (type, length, index)
        for i, curve in enumerate(self.curves):
            segments.append(("curve", arc_lengths[i][1], i))
            if self.connect and i < len(self.curves) - 1:
                segments.append(("connector", connector_lengths[i], i))

        total_length = sum(s[1] for s in segments)
        if total_length == 0:
            total_length = 1.0

        # Allocate points proportionally (min 2 per connector)
        n_connectors = sum(1 for s in segments if s[0] == "connector")
        reserved = n_connectors * 2
        distributable = max(num_pts - reserved, len(self.curves))

        point_counts = []
        for seg_type, length, _ in segments:
            if seg_type == "connector":
                n = max(2, int(round(distributable * length / total_length)))
                point_counts.append(n)
            else:
                n = max(2, int(round(distributable * length / total_length)))
                point_counts.append(n)

        # Build center polyline
        parts = []
        segment_boundaries = []  # (start_idx, end_idx, type, curve_idx) into final xy
        offset = 0
        for idx, (seg_type, _, seg_idx) in enumerate(segments):
            n = point_counts[idx]
            if seg_type == "curve":
                s_vals = np.linspace(0.0, 1.0, n)
                pts = self.curves[seg_idx].curve.evaluate_multi(s_vals).T * scale
            else:
                # Straight line from end of curve[seg_idx] to start of curve[seg_idx+1]
                end_pt = curve_endpoints[seg_idx][1]
                start_pt = curve_endpoints[seg_idx + 1][0]
                pts = np.linspace(end_pt, start_pt, n)

            # Avoid duplicating shared endpoints
            if parts:
                pts = pts[1:]
            seg_len = len(pts)
            segment_boundaries.append((offset, offset + seg_len, seg_type, seg_idx))
            offset += seg_len
            parts.append(pts)

        xy = np.vstack(parts)
        result = dict(center=xy, rail_left=None, rail_right=None, segments=segment_boundaries)

        if rails:
            tangents = np.gradient(xy, axis=0)
            lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
            lengths = np.where(lengths == 0, 1, lengths)
            tangents = tangents / lengths
            normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

            if self.radius_curve is not None:
                # Global radius_curve over the entire compound
                seg_lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
                arc = np.concatenate([[0], np.cumsum(seg_lengths)])
                arc /= arc[-1] if arc[-1] > 0 else 1.0
                rc_points = self.radius_curve.curve.evaluate_multi(arc)
                r_values = rc_points[1]
                y_max = r_values.max()
                if y_max > 0:
                    r_values = r_values * (self.radius / y_max)
                r = r_values[:, np.newaxis]
            else:
                # Per-segment radius: use child's radius_curve or radius, fallback to parent's radius
                r = np.full(len(xy), float(self.radius))
                for start, end, seg_type, curve_idx in segment_boundaries:
                    if seg_type == "connector":
                        continue
                    curve = self.curves[curve_idx]
                    if curve.radius_curve is not None:
                        seg_xy = xy[start:end]
                        seg_lengths = np.linalg.norm(np.diff(seg_xy, axis=0), axis=1)
                        arc = np.concatenate([[0], np.cumsum(seg_lengths)])
                        arc /= arc[-1] if arc[-1] > 0 else 1.0
                        rc_points = curve.radius_curve.curve.evaluate_multi(arc)
                        r_values = rc_points[1]
                        y_max = r_values.max()
                        if y_max > 0:
                            r_values = r_values * (curve.radius / y_max)
                        r[start:end] = r_values
                    else:
                        r[start:end] = curve.radius
                r = r[:, np.newaxis]

            result["rail_left"] = xy + normals * r
            result["rail_right"] = xy - normals * r

        return result

    def plot(self, num_pts=256, nodes=False, rails=False, ax=None, color=None, scale=100, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4), layout="constrained")

        ax.set_aspect("equal")
        color = color or (0, 0, 1)

        if nodes:
            for curve in self.curves:
                ax.scatter(*(curve.nodes * scale), c="black", zorder=3)

        sampled = self.sample(num_pts=num_pts, scale=scale, rails=rails)
        segments = sampled["segments"]

        for start, end, seg_type, _ in segments:
            seg = sampled["center"][start:end]
            style = dict(color=color, **kwargs)
            if seg_type == "connector":
                style.setdefault("linestyle", "--")
                style.setdefault("linewidth", 0.5)
            ax.plot(seg[:, 0], seg[:, 1], **style)

        if rails:
            rail_color = (*color, 0.5) if len(color) == 3 else color
            for start, end, seg_type, _ in segments:
                style = dict(color=rail_color, linewidth=0.5)
                if seg_type == "connector":
                    style["linestyle"] = "--"
                ax.plot(
                    sampled["rail_left"][start:end, 0],
                    sampled["rail_left"][start:end, 1],
                    **style,
                )
                ax.plot(
                    sampled["rail_right"][start:end, 0],
                    sampled["rail_right"][start:end, 1],
                    **style,
                )

        return ax

    def to_svg(
        self,
        path,
        num_pts=256,
        stroke="black",
        scale=100,
        padding=1,
        rails=False,
        max_error=0.001,
    ):
        from .fitcurves import beziers_to_svg_path, fit_curve

        assert path.endswith(".svg")

        sampled = self.sample(num_pts=num_pts, scale=scale, rails=rails)
        xy = sampled["center"]

        beziers = fit_curve(xy, max_error)
        d_center = beziers_to_svg_path(beziers)

        paths = f'  <path d="{d_center}" fill="none" stroke="{stroke}" stroke-width="{self.radius}" />\n'

        if rails:
            d_left = beziers_to_svg_path(fit_curve(sampled["rail_left"], max_error))
            d_right = beziers_to_svg_path(fit_curve(sampled["rail_right"], max_error))
            paths += f'  <path d="{d_left}" fill="none" stroke="{stroke}" stroke-width="0.5" />\n'
            paths += f'  <path d="{d_right}" fill="none" stroke="{stroke}" stroke-width="0.5" />\n'

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
        return sum(len(c) for c in self.curves)


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
        if r := self.radius_curve:
            r.coords[1][-1] = r.coords[1][0]
            self.radius_curve = Bezier.from_coords(r.coords)

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

    def to_dict(self):
        data = dict(
            coords=[list(c) for c in self.coords],
            radius=self.radius,
        )

        if self.radius_curve is not None:
            data["radius_curve"] = [list(c) for c in self.radius_curve.coords]
        else:
            data["radius_curve"] = None

        return data

    def to_json(self, path):
        assert path.endswith(".json")
        with open(path, "wt") as f:
            mrich.writing(path)
            json.dump(self.to_dict(), f, indent=2)

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
