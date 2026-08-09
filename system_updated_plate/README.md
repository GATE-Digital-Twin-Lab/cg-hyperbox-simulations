# Updated plate system

This directory contains the system derived from
`system_plate/plate_hole_data.pkl`.

- `A.npz`: sparse CSR matrix computed as `K + K.T - diag(K)`.
- `b.dat`: unchanged right-hand side from the original data.
- `soln_x.dat`: direct sparse solution of `A x = b`.
- `box_1.dat` through `box_7.dat`: lower and upper bounds, one coordinate per
  row. The first two are the original mildly nonuniform boxes. Boxes 3--7 are
  strongly anisotropic comparison cases (offset sparse supports, graded sparse
  widths, solution-magnitude weighting, and an exact zero-width case).
- `box_10.dat` through `box_19.dat`: five equal-width pairs with full widths
  0.5, 2, 10, 100, and 500. In each pair, the even-numbered box has the exact
  solution slightly off center and the odd-numbered box has it near an edge.
- `build_updated_plate_data.py`: reproducible construction and validation.
- `cg_projection_enclosure.py`: fixed-iteration, Gram-corrected implementation
  of the no-optimization CG-projection enclosure. It evaluates
  `Pi = I - P (P.T A P)^(-1) P.T A` in row blocks and computes the exact
  interval-hull radius `abs(Pi) @ d0` without storing the dense full projector.

The build script rejects a solution whose relative residual exceeds `1e-11`
and validates the containment, varying widths, and displaced centers of both
boxes before writing the files.
