# Interval Conjugate Gradient

This repository includes several versions of the Interval Conjugate Gradient (ICG) alogirthm curently under development. 

- `CG-simulation-initial` presents a simple 3D system that aims to illustrate the main point behind using the constraints induced by the search directions
- `v1` is an implementation of the algorithm using linear optimization to find the intersection of the constraint planes with the hyperbox planes, which, however, often fails due to numerical instabilities and high computational cost
- `v2` is a relaxation of the constrains from `v1` to allow for slabs of width $\tau$
- `v3` implements a completely new approach to the hyperbox shrinking, utilizing error projection operators and zonotope representation of the intervals 



