# Updated plate system

This directory contains the system derived from
`system_plate/plate_hole_data.pkl`.

- `A.npz`: sparse CSR matrix computed as `K + K.T - diag(K)`.
- `b.dat`: unchanged right-hand side from the original data.
- `soln_x.dat`: direct sparse solution of `A x = b`.
- `box_1.dat`, `box_2.dat`: lower and upper bounds, one coordinate per row.
  Both boxes contain `soln_x`, have coordinate-dependent diameters, and have
  centers displaced from the solution in every coordinate.
- `build_updated_plate_data.py`: reproducible construction and validation.
- `cg_projection_enclosure.py`: fixed-iteration, Gram-corrected implementation
  of the no-optimization CG-projection enclosure. It evaluates
  `Pi = I - P (P.T A P)^(-1) P.T A` in row blocks and computes the exact
  interval-hull radius `abs(Pi) @ d0` without storing the dense full projector.

The build script rejects a solution whose relative residual exceeds `1e-11`
and validates the containment, varying widths, and displaced centers of both
boxes before writing the files.
