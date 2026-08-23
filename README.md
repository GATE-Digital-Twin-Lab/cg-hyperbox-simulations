# Interval Conjugate Gradient

This repository includes several versions of the Interval Conjugate Gradient (ICG) alogirthm curently under development. 

- `CG-simulation-initial` presents a simple 3D system that aims to illustrate the main point behind using the constraints induced by the search directions
- `v1` is an implementation of the algorithm using linear optimization to find the intersection of the constraint planes with the hyperbox planes, which, however, often fails due to numerical instabilities and high computational cost
- `v2` is a relaxation of the constrains from `v1` to allow for slabs of width $\tau$
- `v3` implements a completely new approach to the hyperbox shrinking, utilizing error projection operators and zonotope representation of the intervals 
- `v4` implements a slight modification to `v3` where the zonotope hulls are calculated at every $m$ and intersected at the end. Given the fact that each next one is not necessarily smaller than the previous ones, this algorithm guarantees monotonicity. 
- `v5` is a test trial of a full-rank $\Pi$ that combines the ordinary projection with a Richardson inverse approximation. Alas, unsuccesful attempt. 
- `v6` is an unsuccesful attempt to implement the intersect first, then take the hull algorithm. Way too costly though 
- `v7` is an implementation of the partial Cholesky version of the algorithm



