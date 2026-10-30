# Face-centered curl update

`update_curl.cpp` provides two `MeshData<Real>` task functions:

- `EmfCurl`: overwrite the selected face-centered RHS with `-curl(E)`.
- `UpdateWithEmfCurl`: apply `u0 = gam0*u0 + gam1*u1 + beta_dt*(-curl(E))`.

Both use `EmfCurlHelper<face>` in `update.hpp`. The stencil sums oriented
edge contributions weighted by coordinate-provided edge lengths, then divides
by the corresponding face area. Bounds come from each face's topology.
Callers select fields by name; the implementation does not require `Bface`.

## Assumptions

- Selected variables are face-centered and have associated edge fluxes storing
  electric-field-like values per unit length, not already integrated circulations.
- Input, output, and register packs refer to corresponding blocks and variables
  in the same order, with compatible spatial layouts and allocation patterns.
  Explicit checks enforce matching block and component counts only.
- The caller supplies valid edge data, including boundary exchange and any
  required refinement corrections, before scheduling the update.
- The dimensional convention is F1 only in 1D, F1/F2 in 2D, and all faces in 3D.
  The 1D F1 curl is zero; the caller handles transverse cell-centered components.
- Face areas are nonzero. RHS evaluation does not include a timestep factor.

## Testing so far

- AthenaPK GPU fast-magnetosonic linear-wave refinement runs in 1D, 2D, and 3D
  approach second-order convergence. Finest refinement orders for the aggregate
  error are approximately 2.02, 1.97, and 1.95, respectively.
- The 3D component L1 errors match the original AthenaPK curl implementation at
  saved output precision; this is not a claim of bitwise equality.
- Recorded maximum face-divergence diagnostics remain zero in 1D and at roughly
  roundoff scale in the multidimensional runs.
- An additional AthenaPK run with eight total meshblocks passed.

These results exercise the AthenaPK integration of the prototype, not a
standalone Parthenon regression suite.

## Current limitations

- Only `MeshData<Real>` specializations are implemented; no `MeshBlockData`
  specialization or explicit fine-cell-level interface is provided.
- The standalone RHS path has not yet received dedicated analytic validation.
  Parthenon-native tests of curl signs, face bounds, register coefficients, and
  failure cases remain to be added.
- Pack count checks do not verify block identity, variable ordering, metadata,
  spatial extents, or matching sparse allocation. Unallocated entries are skipped.
- Curved/nonuniform geometry, sparse-field cases, MPI execution, and AMR
  interfaces have not been established by the tests summarized above.
- This is an edge-to-face negative-curl operation, not the full generalized
  Stokes machinery in the fine-advection example. Its sign and reduced-dimensional
  conventions must be accounted for when integrating it into that example.
