"""Exact no-optimization CG-projection enclosure implementation.

The dense projector is evaluated in row blocks so the full enclosure can be
computed without materializing an n-by-n matrix.
"""

from time import perf_counter

import numpy as np
from scipy.linalg import qr, solve


def fixed_iteration_cg_with_directions(A, b, x0, iterations):
    """Run exactly ``iterations`` CG steps and record p_k and A p_k."""
    b = np.asarray(b, dtype=float)
    x = np.array(x0, dtype=float, copy=True)
    n = b.size
    if not 1 <= iterations <= n:
        raise ValueError(f"iterations must satisfy 1 <= m <= n={n}.")

    directions = np.empty((n, iterations), dtype=float)
    applied_directions = np.empty((n, iterations), dtype=float)
    residual_norms = np.empty(iterations, dtype=float)

    residual = b - A @ x
    direction = residual.copy()

    for k in range(iterations):
        applied_direction = A @ direction
        residual_norm_sq = float(residual @ residual)
        denominator = float(direction @ applied_direction)
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise RuntimeError(
                f"CG denominator is not positive and finite at iteration {k}."
            )

        directions[:, k] = direction
        applied_directions[:, k] = applied_direction

        alpha = residual_norm_sq / denominator
        x = x + alpha * direction
        next_residual = residual - alpha * applied_direction
        residual_norms[k] = np.linalg.norm(next_residual)

        if k < iterations - 1:
            beta = float(next_residual @ next_residual) / residual_norm_sq
            direction = next_residual + beta * direction
        residual = next_residual

    return {
        "x_cg": x,
        "residual": residual,
        "residual_norms": residual_norms,
        "directions": directions,
        "applied_directions": applied_directions,
    }


def _gram_diagnostics(gram):
    identity_error = gram - np.eye(gram.shape[0])
    off_diagonal = identity_error - np.diag(np.diag(identity_error))
    return {
        "gram_condition": float(np.linalg.cond(gram)),
        "gram_identity_fro": float(np.linalg.norm(identity_error, ord="fro")),
        "gram_max_offdiag": float(
            np.max(np.abs(off_diagonal), initial=0.0)
        ),
        "gram_symmetry_error": float(np.linalg.norm(gram - gram.T, ord=np.inf)),
        "gram_diag_max_error": float(
            np.max(np.abs(np.diag(gram) - 1.0), initial=0.0)
        ),
    }


def select_independent_directions(
    directions,
    applied_directions,
    qr_tolerance=1.0e-10,
    max_gram_condition=1.0e12,
):
    """Select a stable, linearly independent subset of recorded directions.

    Scaling a selected direction does not change
    P (P.T A P)^-1 P.T. A-normalization is therefore used as an equivalent
    basis change to avoid arbitrary CG direction magnitudes in the rank and
    Gram tests.
    """
    directions = np.asarray(directions, dtype=float)
    applied_directions = np.asarray(applied_directions, dtype=float)
    if directions.shape != applied_directions.shape:
        raise ValueError("directions and applied_directions must have equal shapes.")

    a_norm_sq = np.sum(directions * applied_directions, axis=0)
    valid = np.isfinite(a_norm_sq) & (a_norm_sq > 0.0)
    if not np.any(valid):
        raise RuntimeError("No recorded direction has a positive finite A-norm.")

    valid_indices = np.flatnonzero(valid)
    a_norm = np.sqrt(a_norm_sq[valid])
    normalized_directions = directions[:, valid] / a_norm[None, :]
    normalized_applied = applied_directions[:, valid] / a_norm[None, :]

    _, triangular, pivots = qr(
        normalized_directions,
        pivoting=True,
        mode="economic",
        check_finite=False,
    )
    qr_diagonal = np.abs(np.diag(triangular))
    if qr_diagonal.size == 0 or qr_diagonal[0] == 0.0:
        raise RuntimeError("Could not determine a nonzero direction rank.")

    qr_rank = int(
        np.sum(qr_diagonal > qr_tolerance * qr_diagonal[0])
    )
    qr_rank = max(qr_rank, 1)
    candidate_pivots = pivots[:qr_rank]
    candidate_directions = normalized_directions[:, candidate_pivots]
    candidate_applied = normalized_applied[:, candidate_pivots]
    candidate_gram = candidate_directions.T @ candidate_applied
    candidate_gram = 0.5 * (candidate_gram + candidate_gram.T)

    # Prefixes in pivot order are nested. For an SPD Gram matrix their
    # condition numbers are nondecreasing, so binary search finds the largest
    # numerically stable independent prefix.
    lower = 1
    upper = qr_rank
    while lower < upper:
        trial = (lower + upper + 1) // 2
        condition = np.linalg.cond(candidate_gram[:trial, :trial])
        if np.isfinite(condition) and condition <= max_gram_condition:
            lower = trial
        else:
            upper = trial - 1

    retained_rank = lower
    selected_pivots = candidate_pivots[:retained_rank]
    selected = valid_indices[selected_pivots]
    basis = normalized_directions[:, selected_pivots]
    applied_basis = normalized_applied[:, selected_pivots]
    gram = candidate_gram[:retained_rank, :retained_rank]

    diagnostics = _gram_diagnostics(gram)
    return {
        "selected": selected,
        "basis": basis,
        "applied_basis": applied_basis,
        "gram": gram,
        "available_direction_count": int(directions.shape[1]),
        "valid_direction_count": int(np.count_nonzero(valid)),
        "qr_rank": qr_rank,
        "retained_rank": retained_rank,
        **diagnostics,
    }


def project_enclosure_rows(
    A,
    b,
    lower,
    upper,
    selection,
    coordinate_indices=None,
    row_block_size=128,
    progress_label=None,
):
    """Evaluate the exact Gram-corrected enclosure on requested projector rows."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    center = 0.5 * (lower + upper)
    initial_radius = 0.5 * (upper - lower)
    n = b.size

    if coordinate_indices is None:
        coordinate_indices = np.arange(n, dtype=int)
    else:
        coordinate_indices = np.asarray(coordinate_indices, dtype=int)

    basis = selection["basis"]
    applied_basis = selection["applied_basis"]
    gram = selection["gram"]

    rhs = basis.T @ (b - A @ center)
    coefficients = solve(
        gram,
        rhs,
        assume_a="sym",
        check_finite=False,
    )
    projected_center = center + basis @ coefficients

    selected_count = coordinate_indices.size
    projected_radius = np.empty(selected_count, dtype=float)
    contracted_lower = np.empty(selected_count, dtype=float)
    contracted_upper = np.empty(selected_count, dtype=float)

    started = perf_counter()
    report_stride = max(row_block_size, 2048)
    for start in range(0, selected_count, row_block_size):
        stop = min(selected_count, start + row_block_size)
        rows = coordinate_indices[start:stop]

        # Pi[rows, :] = E[rows, :] - P[rows, :] G^-1 P.T A.
        left_factor = solve(
            gram,
            basis[rows, :].T,
            assume_a="sym",
            check_finite=False,
        ).T
        projector_rows = -(left_factor @ applied_basis.T)
        projector_rows[np.arange(stop - start), rows] += 1.0

        radius_block = np.abs(projector_rows) @ initial_radius
        projected_radius[start:stop] = radius_block
        contracted_lower[start:stop] = np.maximum(
            lower[rows],
            projected_center[rows] - radius_block,
        )
        contracted_upper[start:stop] = np.minimum(
            upper[rows],
            projected_center[rows] + radius_block,
        )

        if progress_label and (
            stop == selected_count or stop % report_stride < row_block_size
        ):
            print(
                f"{progress_label}: projector rows {stop}/{selected_count} "
                f"({perf_counter() - started:.1f} s)"
            )

    if np.any(contracted_lower > contracted_upper):
        violation = contracted_lower - contracted_upper
        bad = np.flatnonzero(violation > 0.0)
        raise RuntimeError(
            "Projection enclosure produced inverted intervals; "
            f"maximum violation {np.max(violation):.6e}, "
            f"first local coordinates {bad[:10]}."
        )

    constraint_residual = basis.T @ (A @ projected_center - b)
    return {
        "ell": contracted_lower,
        "u": contracted_upper,
        "x_hat": projected_center[coordinate_indices],
        "radius": projected_radius,
        "coordinate_indices": coordinate_indices,
        "max_constraint_residual": float(
            np.max(np.abs(constraint_residual), initial=0.0)
        ),
    }


def run_exact_cg_projection_enclosure(
    A,
    b,
    lower,
    upper,
    iterations,
    tracked_indices,
    diagnostic_period=50,
    qr_tolerance=1.0e-10,
    max_gram_condition=1.0e12,
    row_block_size=128,
    progress_label=None,
):
    """Run the supplied algorithm and retain exact sampled-row diagnostics."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    center = 0.5 * (lower + upper)
    tracked_indices = np.asarray(tracked_indices, dtype=int)

    cg_run = fixed_iteration_cg_with_directions(
        A=A,
        b=b,
        x0=center,
        iterations=iterations,
    )
    directions = cg_run["directions"]
    applied_directions = cg_run["applied_directions"]

    checkpoint_iterations = np.arange(
        diagnostic_period,
        iterations + 1,
        diagnostic_period,
        dtype=int,
    )
    if checkpoint_iterations.size == 0 or checkpoint_iterations[-1] != iterations:
        checkpoint_iterations = np.append(checkpoint_iterations, iterations)

    tracked_lower_history = [lower[tracked_indices].copy()]
    tracked_upper_history = [upper[tracked_indices].copy()]
    tracked_center_history = [center[tracked_indices].copy()]
    tracked_radius_history = [
        0.5 * (upper[tracked_indices] - lower[tracked_indices])
    ]
    diagnostics = []
    final_selection = None

    for checkpoint in checkpoint_iterations:
        selection = select_independent_directions(
            directions=directions[:, :checkpoint],
            applied_directions=applied_directions[:, :checkpoint],
            qr_tolerance=qr_tolerance,
            max_gram_condition=max_gram_condition,
        )
        tracked = project_enclosure_rows(
            A=A,
            b=b,
            lower=lower,
            upper=upper,
            selection=selection,
            coordinate_indices=tracked_indices,
            row_block_size=row_block_size,
        )
        tracked_lower_history.append(tracked["ell"])
        tracked_upper_history.append(tracked["u"])
        tracked_center_history.append(tracked["x_hat"])
        tracked_radius_history.append(tracked["radius"])
        diagnostics.append({
            "iteration": int(checkpoint),
            "available_direction_count": selection["available_direction_count"],
            "valid_direction_count": selection["valid_direction_count"],
            "qr_rank": selection["qr_rank"],
            "retained_rank": selection["retained_rank"],
            "gram_condition": selection["gram_condition"],
            "gram_identity_fro": selection["gram_identity_fro"],
            "gram_max_offdiag": selection["gram_max_offdiag"],
            "gram_symmetry_error": selection["gram_symmetry_error"],
            "gram_diag_max_error": selection["gram_diag_max_error"],
            "max_constraint_residual": tracked["max_constraint_residual"],
        })
        if checkpoint == iterations:
            final_selection = selection

    full_enclosure = project_enclosure_rows(
        A=A,
        b=b,
        lower=lower,
        upper=upper,
        selection=final_selection,
        coordinate_indices=None,
        row_block_size=row_block_size,
        progress_label=progress_label,
    )

    # Do not retain the large direction matrices after the enclosure is built.
    return {
        "ell": full_enclosure["ell"],
        "u": full_enclosure["u"],
        "x_hat": full_enclosure["x_hat"],
        "radius": full_enclosure["radius"],
        "iterations": np.r_[0, checkpoint_iterations],
        "ell_history": np.asarray(tracked_lower_history),
        "u_history": np.asarray(tracked_upper_history),
        "projected_center_history": np.asarray(tracked_center_history),
        "projected_radius_history": np.asarray(tracked_radius_history),
        "diagnostics": diagnostics,
        "residual_norms": cg_run["residual_norms"],
        "x_cg": cg_run["x_cg"],
        "selected": final_selection["selected"],
        "qr_rank": final_selection["qr_rank"],
        "retained_rank": final_selection["retained_rank"],
        "gram_condition": final_selection["gram_condition"],
        "gram_identity_fro": final_selection["gram_identity_fro"],
        "gram_max_offdiag": final_selection["gram_max_offdiag"],
        "gram_symmetry_error": final_selection["gram_symmetry_error"],
        "gram_diag_max_error": final_selection["gram_diag_max_error"],
        "max_constraint_residual": full_enclosure["max_constraint_residual"],
    }
