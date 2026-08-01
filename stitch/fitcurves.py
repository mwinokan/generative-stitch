"""Attempt to fit cubic Bezier curves to a polyline.

Python implementation of Philip J. Schneider's
"Algorithm for Automatically Fitting Digitized Curves"
from "Graphics Gems", Academic Press, 1990.

Inlined from https://github.com/volkerp/fitCurves (MIT license).
"""

import numpy as np
from numpy.linalg import norm


# --- Cubic Bezier evaluation ---


def _q(ctrl, t):
    """Evaluate cubic bezier at t."""
    return (
        (1.0 - t) ** 3 * ctrl[0]
        + 3 * (1.0 - t) ** 2 * t * ctrl[1]
        + 3 * (1.0 - t) * t**2 * ctrl[2]
        + t**3 * ctrl[3]
    )


def _qprime(ctrl, t):
    """First derivative at t."""
    return (
        3 * (1.0 - t) ** 2 * (ctrl[1] - ctrl[0])
        + 6 * (1.0 - t) * t * (ctrl[2] - ctrl[1])
        + 3 * t**2 * (ctrl[3] - ctrl[2])
    )


def _qprimeprime(ctrl, t):
    """Second derivative at t."""
    return 6 * (1.0 - t) * (ctrl[2] - 2 * ctrl[1] + ctrl[0]) + 6 * t * (
        ctrl[3] - 2 * ctrl[2] + ctrl[1]
    )


# --- Main API ---


def fit_curve(points, max_error):
    """Fit one or more cubic Bezier curves to a set of 2D points.

    Parameters
    ----------
    points : array-like, shape (N, 2)
        Ordered 2D points to fit.
    max_error : float
        Maximum allowed squared distance between points and fitted curve.

    Returns
    -------
    list of list
        Each element is a cubic Bezier [P0, P1, P2, P3] (4 control points).
    """
    points = np.array(points, dtype=float)
    left_tangent = _normalize(points[1] - points[0])
    right_tangent = _normalize(points[-2] - points[-1])
    return _fit_cubic(points, left_tangent, right_tangent, max_error)


# --- Internal fitting ---


def _fit_cubic(points, left_tangent, right_tangent, error):
    if len(points) == 2:
        dist = norm(points[0] - points[1]) / 3.0
        bez = [
            points[0],
            points[0] + left_tangent * dist,
            points[1] + right_tangent * dist,
            points[1],
        ]
        return [bez]

    u = _chord_length_parameterize(points)
    bez = _generate_bezier(points, u, left_tangent, right_tangent)
    max_err, split_point = _compute_max_error(points, bez, u)

    if max_err < error:
        return [bez]

    if max_err < error**2:
        for _ in range(20):
            u_prime = _reparameterize(bez, points, u)
            bez = _generate_bezier(points, u_prime, left_tangent, right_tangent)
            max_err, split_point = _compute_max_error(points, bez, u_prime)
            if max_err < error:
                return [bez]
            u = u_prime

    # Split at max error point and fit recursively
    center_tangent = _normalize(points[split_point - 1] - points[split_point + 1])
    beziers = _fit_cubic(points[: split_point + 1], left_tangent, center_tangent, error)
    beziers += _fit_cubic(points[split_point:], -center_tangent, right_tangent, error)
    return beziers


def _generate_bezier(points, parameters, left_tangent, right_tangent):
    bez = [points[0], None, None, points[-1]]

    A = np.zeros((len(parameters), 2, 2))
    for i, u in enumerate(parameters):
        A[i][0] = left_tangent * 3 * (1 - u) ** 2 * u
        A[i][1] = right_tangent * 3 * (1 - u) * u**2

    C = np.zeros((2, 2))
    X = np.zeros(2)

    for i, (point, u) in enumerate(zip(points, parameters)):
        C[0][0] += np.dot(A[i][0], A[i][0])
        C[0][1] += np.dot(A[i][0], A[i][1])
        C[1][0] += np.dot(A[i][0], A[i][1])
        C[1][1] += np.dot(A[i][1], A[i][1])

        tmp = point - _q([points[0], points[0], points[-1], points[-1]], u)
        X[0] += np.dot(A[i][0], tmp)
        X[1] += np.dot(A[i][1], tmp)

    det_C0_C1 = C[0][0] * C[1][1] - C[1][0] * C[0][1]
    det_C0_X = C[0][0] * X[1] - C[1][0] * X[0]
    det_X_C1 = X[0] * C[1][1] - X[1] * C[0][1]

    alpha_l = 0.0 if det_C0_C1 == 0 else det_X_C1 / det_C0_C1
    alpha_r = 0.0 if det_C0_C1 == 0 else det_C0_X / det_C0_C1

    seg_length = norm(points[0] - points[-1])
    epsilon = 1.0e-6 * seg_length

    if alpha_l < epsilon or alpha_r < epsilon:
        bez[1] = bez[0] + left_tangent * (seg_length / 3.0)
        bez[2] = bez[3] + right_tangent * (seg_length / 3.0)
    else:
        bez[1] = bez[0] + left_tangent * alpha_l
        bez[2] = bez[3] + right_tangent * alpha_r

    return bez


def _reparameterize(bez, points, parameters):
    return [_newton_raphson(bez, point, u) for point, u in zip(points, parameters)]


def _newton_raphson(bez, point, u):
    d = _q(bez, u) - point
    numerator = (d * _qprime(bez, u)).sum()
    denominator = (_qprime(bez, u) ** 2 + d * _qprimeprime(bez, u)).sum()
    if denominator == 0.0:
        return u
    return u - numerator / denominator


def _chord_length_parameterize(points):
    u = [0.0]
    for i in range(1, len(points)):
        u.append(u[i - 1] + norm(points[i] - points[i - 1]))
    u = [v / u[-1] for v in u]
    return u


def _compute_max_error(points, bez, parameters):
    max_dist = 0.0
    split_point = len(points) // 2
    for i, (point, u) in enumerate(zip(points, parameters)):
        dist = norm(_q(bez, u) - point) ** 2
        if dist > max_dist:
            max_dist = dist
            split_point = i
    return max_dist, split_point


def _normalize(v):
    n = norm(v)
    if n == 0:
        return v
    return v / n


# --- SVG helpers ---


def beziers_to_svg_path(beziers):
    """Convert a list of fitted cubic Beziers to an SVG path `d` string.

    Parameters
    ----------
    beziers : list of list
        Output of fit_curve: each element is [P0, P1, P2, P3].

    Returns
    -------
    str
        SVG path data using M and C commands.
    """
    if not beziers:
        return ""

    parts = [f"M {beziers[0][0][0]},{beziers[0][0][1]}"]
    for bez in beziers:
        parts.append(
            f"C {bez[1][0]},{bez[1][1]} {bez[2][0]},{bez[2][1]} {bez[3][0]},{bez[3][1]}"
        )
    return " ".join(parts)
