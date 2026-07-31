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

    save_npz(OUTPUT_DIR / "A.npz", matrix, compressed=True)
    np.savetxt(OUTPUT_DIR / "b.dat", rhs, fmt="%.17e")
    np.savetxt(OUTPUT_DIR / "soln_x.dat", solution, fmt="%.17e")
    np.savetxt(OUTPUT_DIR / "box_1.dat", box_1, fmt="%.17e")
    np.savetxt(OUTPUT_DIR / "box_2.dat", box_2, fmt="%.17e")

    print(f"matrix shape: {matrix.shape}")
    print(f"matrix nonzeros: {matrix.nnz}")
    print(f"symmetry defect: {np.max(np.abs((matrix - matrix.T).data), initial=0.0):.6e}")
    print(f"absolute residual norm: {absolute_residual:.6e}")
    print(f"relative residual norm: {relative_residual:.6e}")
    print(f"maximum componentwise residual: {componentwise_residual:.6e}")
    for name, box in (("box_1", box_1), ("box_2", box_2)):
        widths = box[:, 1] - box[:, 0]
        centers = np.mean(box, axis=1)
        print(
            f"{name}: width range [{widths.min():.6e}, {widths.max():.6e}], "
            f"max center displacement {np.max(np.abs(centers - solution)):.6e}"
        )


if __name__ == "__main__":
    main()
