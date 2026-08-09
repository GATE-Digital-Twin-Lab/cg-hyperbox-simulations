"""Build the symmetrized plate system and verified enclosure data."""

from pathlib import Path
import pickle

import numpy as np
from scipy.sparse import diags, save_npz
from scipy.sparse.linalg import spsolve


SOURCE = Path(__file__).resolve().parents[1] / "system_plate" / "plate_hole_data.pkl"
OUTPUT_DIR = Path(__file__).resolve().parent


def make_offset_box(
    solution: np.ndarray,
    radius_multiplier: float,
    phase: float,
) -> np.ndarray:
    """Return a nonuniform box containing, but not centered on, ``solution``."""
    indices = np.arange(solution.size, dtype=float)
    solution_scale = max(float(np.max(np.abs(solution))), 1.0)
    local_scale = np.maximum(np.abs(solution), 1.0e-4 * solution_scale)

    # The radius and center offset both vary coordinate by coordinate.
    modulation = 0.75 + 0.50 * (0.5 + 0.5 * np.sin(indices * 0.017 + phase))
    radius = radius_multiplier * local_scale * modulation
    offset = 0.45 * radius * np.sin(indices * 0.031 + phase + 0.4)
    center = solution + offset

    box = np.column_stack((center - radius, center + radius))
    if not np.all((box[:, 0] <= solution) & (solution <= box[:, 1])):
        raise RuntimeError("Constructed box does not contain the solution.")
    if np.any(np.isclose(center, solution, rtol=0.0, atol=1.0e-15)):
        raise RuntimeError("Constructed box is centered on the solution in a coordinate.")
    if np.ptp(box[:, 1] - box[:, 0]) == 0.0:
        raise RuntimeError("Constructed box diameters do not vary.")
    return box


def make_comparison_boxes(solution: np.ndarray) -> dict[str, np.ndarray]:
    """Return five strongly anisotropic boxes for the v4 comparison.

    The original 0.8x and 1.6x boxes have broadly similar coordinate-radius
    profiles, and their projected interval hulls never tighten the supplied
    bounds.  Pilot runs showed that even generic 1000:1 anisotropy was not
    enough: off-diagonal interval-hull contributions still dominated.  These
    profiles therefore make a sparse subset of the 80 animation coordinates
    wide and make coupled coordinates nearly point-valued.

    All five boxes use the same small displacement on five driver coordinates,
    so differences among them are caused by widths rather than different CG
    trajectories. ``box_7`` makes every non-driver, non-selected coordinate
    exactly zero width to exercise a degenerate interval input.
    """
    animation_indices = np.unique(
        np.linspace(0, solution.size - 1, min(solution.size, 80))
        .round()
        .astype(int)
    )
    solution_scale = max(float(np.max(np.abs(solution))), 1.0)
    local_scale = np.maximum(np.abs(solution), 1.0e-4 * solution_scale)
    driver_indices = np.array([1, 4321, 8765, 13579, 20001], dtype=int)
    driver_signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    common_offset = np.zeros_like(solution)
    common_offset[driver_indices] = (
        0.25 * local_scale[driver_indices] * driver_signs
    )
    center = solution + common_offset

    boxes = {}
    for box_number in range(3, 8):
        name = f"box_{box_number}"
        radius = 1.0e-10 * local_scale
        radius[driver_indices] = local_scale[driver_indices]

        if box_number == 3:
            selected = animation_indices[np.arange(len(animation_indices)) % 4 == 0]
            radius[selected] = 1000.0 * local_scale[selected]
        elif box_number == 4:
            selected = animation_indices[np.arange(len(animation_indices)) % 4 == 1]
            radius[selected] = 500.0 * local_scale[selected]
        elif box_number == 5:
            selected = animation_indices[
                np.isin(np.arange(len(animation_indices)) % 8, [2, 3])
            ]
            radius[selected] = (
                np.geomspace(100.0, 2000.0, len(selected)) * local_scale[selected]
            )
        elif box_number == 6:
            selected = animation_indices[
                np.argsort(np.abs(solution[animation_indices]))[-15:]
            ]
            radius[selected] = (
                200.0
                + 800.0
                * np.abs(solution[selected])
                / np.max(np.abs(solution[selected]))
            ) * local_scale[selected]
        else:
            radius[:] = 0.0
            radius[driver_indices] = local_scale[driver_indices]
            selected = animation_indices[np.arange(len(animation_indices)) % 6 == 0]
            radius[selected] = 800.0 * local_scale[selected]

        box = np.column_stack((center - radius, center + radius))
        if not np.all((box[:, 0] <= solution) & (solution <= box[:, 1])):
            raise RuntimeError(f"Constructed {name} does not contain the solution.")
        boxes[name] = box
    return boxes


def make_uniform_width_boxes(solution: np.ndarray) -> dict[str, np.ndarray]:
    """Return boxes 10--19 as five equal-width center/edge pairs.

    For each prescribed full width, the even-numbered box places the solution
    5% of a radius away from the box center. The following odd-numbered box
    places it 95% of a radius away, so only 5% of a radius remains between the
    solution and its nearest edge. Coordinate-dependent signs alternate which
    edge is near without changing any coordinate width.
    """
    widths = (0.5, 2.0, 10.0, 100.0, 500.0)
    indices = np.arange(solution.size, dtype=float)
    signs = np.where(np.sin(0.019 * indices + 0.3) >= 0.0, 1.0, -1.0)
    boxes = {}

    for pair_index, width in enumerate(widths):
        radius = 0.5 * width
        for pair_offset, displacement_fraction in enumerate((0.05, 0.95)):
            box_number = 10 + 2 * pair_index + pair_offset
            center = solution + displacement_fraction * radius * signs
            box = np.column_stack((center - radius, center + radius))
            actual_widths = box[:, 1] - box[:, 0]
            if not np.all((box[:, 0] <= solution) & (solution <= box[:, 1])):
                raise RuntimeError(
                    f"Constructed box_{box_number} does not contain the solution."
                )
            if not np.allclose(actual_widths, width, rtol=0.0, atol=1.0e-12):
                raise RuntimeError(
                    f"Constructed box_{box_number} does not have width {width}."
                )
            boxes[f"box_{box_number}"] = box
    return boxes


def main() -> None:
    with SOURCE.open("rb") as file:
        stored_matrix, stored_rhs, *_ = pickle.load(file)

    matrix = (
        stored_matrix + stored_matrix.T - diags(stored_matrix.diagonal())
    ).tocsr()
    rhs = np.asarray(stored_rhs, dtype=np.float64).reshape(-1)

    solution = np.asarray(spsolve(matrix.tocsc(), rhs), dtype=np.float64)
    residual = matrix @ solution - rhs
    absolute_residual = float(np.linalg.norm(residual))
    relative_residual = absolute_residual / max(float(np.linalg.norm(rhs)), 1.0)
    componentwise_residual = float(np.max(np.abs(residual)))

    if not np.all(np.isfinite(solution)):
        raise RuntimeError("The computed solution contains non-finite values.")
    if relative_residual > 1.0e-11:
        raise RuntimeError(
            f"Solution verification failed: relative residual {relative_residual:.6e}."
        )

    box_1 = make_offset_box(solution, radius_multiplier=0.80, phase=0.20)
    box_2 = make_offset_box(solution, radius_multiplier=1.60, phase=1.10)
    comparison_boxes = make_comparison_boxes(solution)
    uniform_width_boxes = make_uniform_width_boxes(solution)

    save_npz(OUTPUT_DIR / "A.npz", matrix, compressed=True)
    np.savetxt(OUTPUT_DIR / "b.dat", rhs, fmt="%.17e")
    np.savetxt(OUTPUT_DIR / "soln_x.dat", solution, fmt="%.17e")
    np.savetxt(OUTPUT_DIR / "box_1.dat", box_1, fmt="%.17e")
    np.savetxt(OUTPUT_DIR / "box_2.dat", box_2, fmt="%.17e")
    for name, box in comparison_boxes.items():
        np.savetxt(OUTPUT_DIR / f"{name}.dat", box, fmt="%.17e")
    for name, box in uniform_width_boxes.items():
        np.savetxt(OUTPUT_DIR / f"{name}.dat", box, fmt="%.17e")

    print(f"matrix shape: {matrix.shape}")
    print(f"matrix nonzeros: {matrix.nnz}")
    print(f"symmetry defect: {np.max(np.abs((matrix - matrix.T).data), initial=0.0):.6e}")
    print(f"absolute residual norm: {absolute_residual:.6e}")
    print(f"relative residual norm: {relative_residual:.6e}")
    print(f"maximum componentwise residual: {componentwise_residual:.6e}")
    for name, box in (
        ("box_1", box_1),
        ("box_2", box_2),
        *comparison_boxes.items(),
        *uniform_width_boxes.items(),
    ):
        widths = box[:, 1] - box[:, 0]
        centers = np.mean(box, axis=1)
        print(
            f"{name}: width range [{widths.min():.6e}, {widths.max():.6e}], "
            f"zero widths {np.count_nonzero(widths == 0.0)}, "
            f"max center displacement {np.max(np.abs(centers - solution)):.6e}"
        )


if __name__ == "__main__":
    main()
