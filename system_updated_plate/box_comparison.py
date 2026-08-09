"""Tracked-coordinate implementation for the v4 box comparison animation."""

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.sparse.linalg._isolve.utils import make_system

from .cg_projection_enclosure import select_independent_directions


def scipy_cg_with_directions(A, b, x0, maxiter):
    """Run SciPy's unpreconditioned CG recurrence and retain p and A p."""
    A, M, x, b, postprocess = make_system(A, None, x0, b)
    dotprod = np.vdot if np.iscomplexobj(x) else np.dot
    matvec = A.matvec
    psolve = M.matvec
    residual = b - matvec(x) if x.any() else b.copy()

    directions = np.empty((len(b), maxiter), dtype=x.dtype)
    applied_directions = np.empty_like(directions)
    residual_norms = np.empty(maxiter, dtype=float)
    rho_previous = None
    direction = None

    for iteration in range(maxiter):
        preconditioned = psolve(residual)
        rho_current = dotprod(residual, preconditioned)
        if iteration:
            direction *= rho_current / rho_previous
            direction += preconditioned
        else:
            direction = preconditioned.copy()

        directions[:, iteration] = direction
        applied = matvec(direction)
        applied_directions[:, iteration] = applied
        alpha = rho_current / dotprod(direction, applied)
        x += alpha * direction
        residual -= alpha * applied
        rho_previous = rho_current
        residual_norms[iteration] = np.linalg.norm(residual)

    return {
        "x_cg": postprocess(x),
        "residual_norms": residual_norms,
        "directions": directions,
        "applied_directions": applied_directions,
    }


def select_nested_independent_basis(
    directions,
    applied_directions,
    qr_tolerance=1.0e-10,
    max_gram_condition=1.0e12,
):
    """Select the stable v4 direction set and restore chronological order."""
    selection = select_independent_directions(
        directions=directions,
        applied_directions=applied_directions,
        qr_tolerance=qr_tolerance,
        max_gram_condition=max_gram_condition,
    )
    order = np.argsort(selection["selected"])
    selected = selection["selected"][order]
    original_basis = selection["basis"][:, order]
    original_applied_basis = selection["applied_basis"][:, order]
    gram = original_basis.T @ original_applied_basis
    gram = 0.5 * (gram + gram.T)
    factor = cholesky(gram, lower=True, check_finite=False)
    basis = solve_triangular(
        factor, original_basis.T, lower=True, check_finite=False
    ).T
    applied_basis = solve_triangular(
        factor, original_applied_basis.T, lower=True, check_finite=False
    ).T
    return {
        "selected": selected,
        "basis": basis,
        "applied_basis": applied_basis,
        "qr_rank": selection["qr_rank"],
        "retained_rank": selection["retained_rank"],
        "gram_condition": selection["gram_condition"],
    }


def run_tracked_box_group(
    A,
    b,
    boxes,
    iterations,
    snapshot_period,
    tracked_indices,
    qr_tolerance=1.0e-10,
    max_gram_condition=1.0e12,
    progress_label=None,
):
    """Run v4's every-j intersection for boxes having one shared center.

    Only the requested coordinate rows are evaluated because the comparison
    notebook visualizes those rows and does not report full-system bounds.
    All distinct intermediate projected hulls are still intersected; storage
    alone is decimated to ``snapshot_period``.
    """
    names = list(boxes)
    lower = np.column_stack([boxes[name][:, 0] for name in names])
    upper = np.column_stack([boxes[name][:, 1] for name in names])
    centers = 0.5 * (lower + upper)
    # Decimal serialization of very wide boxes can perturb recovered centers
    # by a few ulps through cancellation.
    if not np.allclose(centers, centers[:, [0]], rtol=0.0, atol=1.0e-10):
        raise ValueError("Boxes in one group must have a common center.")
    center = centers[:, 0]
    radii = 0.5 * (upper - lower)
    tracked_indices = np.asarray(tracked_indices, dtype=int)

    cg_run = scipy_cg_with_directions(A, b, center, iterations)
    selection = select_nested_independent_basis(
        cg_run["directions"],
        cg_run["applied_directions"],
        qr_tolerance=qr_tolerance,
        max_gram_condition=max_gram_condition,
    )
    # Release the two large recurrence arrays before building histories.
    del cg_run["directions"], cg_run["applied_directions"]

    basis = selection["basis"]
    applied_basis = selection["applied_basis"]
    selected = selection["selected"]
    coefficients = basis.T @ (b - A @ center)

    snapshot_iterations = np.arange(
        snapshot_period, iterations + 1, snapshot_period, dtype=int
    )
    if snapshot_iterations.size == 0 or snapshot_iterations[-1] != iterations:
        snapshot_iterations = np.append(snapshot_iterations, iterations)
    frame_iterations = np.r_[0, snapshot_iterations]
    snapshot_lookup = {int(value): i for i, value in enumerate(frame_iterations)}

    row_count = len(tracked_indices)
    projector = np.zeros((row_count, len(b)), dtype=float)
    projector[np.arange(row_count), tracked_indices] = 1.0
    projected_center = center[tracked_indices].copy()
    running_lower = lower[tracked_indices].copy()
    running_upper = upper[tracked_indices].copy()
    raw_lower = running_lower.copy()
    raw_upper = running_upper.copy()

    shape = (len(frame_iterations), row_count, len(names))
    lower_history = np.empty(shape)
    upper_history = np.empty(shape)
    raw_lower_history = np.empty(shape)
    raw_upper_history = np.empty(shape)
    lower_history[0] = running_lower
    upper_history[0] = running_upper
    raw_lower_history[0] = running_lower
    raw_upper_history[0] = running_upper

    basis_column = 0
    for iteration in range(1, iterations + 1):
        changed = False
        while basis_column < len(selected) and selected[basis_column] < iteration:
            row_factor = basis[tracked_indices, basis_column]
            projector -= row_factor[:, None] * applied_basis[:, basis_column][None, :]
            projected_center += row_factor * coefficients[basis_column]
            basis_column += 1
            changed = True

        if changed:
            projected_radius = np.abs(projector) @ radii
            raw_lower = projected_center[:, None] - projected_radius
            raw_upper = projected_center[:, None] + projected_radius
            running_lower = np.maximum(running_lower, raw_lower)
            running_upper = np.minimum(running_upper, raw_upper)
            if np.any(running_lower > running_upper + 1.0e-12):
                raise RuntimeError(
                    f"Running tracked-coordinate intersection became empty at "
                    f"iteration {iteration}."
                )

        if iteration in snapshot_lookup:
            frame = snapshot_lookup[iteration]
            if not changed:
                projected_radius = np.abs(projector) @ radii
                raw_lower = projected_center[:, None] - projected_radius
                raw_upper = projected_center[:, None] + projected_radius
            lower_history[frame] = running_lower
            upper_history[frame] = running_upper
            raw_lower_history[frame] = raw_lower
            raw_upper_history[frame] = raw_upper
            if progress_label:
                tightened = np.count_nonzero(
                    running_upper - running_lower
                    < upper[tracked_indices] - lower[tracked_indices] - 1.0e-14
                )
                print(
                    f"{progress_label}: iteration {iteration}/{iterations}; "
                    f"tracked box-coordinate pairs tightened {tightened}/"
                    f"{row_count * len(names)}"
                )

    results = {}
    for box_column, name in enumerate(names):
        initial_width = upper[tracked_indices, box_column] - lower[tracked_indices, box_column]
        width_history = (
            upper_history[:, :, box_column] - lower_history[:, :, box_column]
        )
        shrinkage = np.zeros_like(width_history)
        positive_width = initial_width > 0.0
        shrinkage[:, positive_width] = np.clip(
            100.0
            * (
                1.0
                - width_history[:, positive_width]
                / initial_width[None, positive_width]
            ),
            0.0,
            100.0,
        )
        results[name] = {
            "iterations": frame_iterations,
            "ell_history": lower_history[:, :, box_column],
            "u_history": upper_history[:, :, box_column],
            "raw_ell_history": raw_lower_history[:, :, box_column],
            "raw_u_history": raw_upper_history[:, :, box_column],
            "shrinkage_history": shrinkage,
            "zero_width": ~positive_width,
            "residual_norms": cg_run["residual_norms"],
            "selected": selected,
            "qr_rank": selection["qr_rank"],
            "retained_rank": selection["retained_rank"],
            "gram_condition": selection["gram_condition"],
        }
    return results
