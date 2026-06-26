# Non-Linear Cardiac Mechanics

Parallel finite-element solver for nonlinear cardiac mechanics, implemented with deal.II, MPI, and Trilinos.

The code solves a nonlinear equilibrium problem for a 3D left-ventricular geometry under internal pressure loading, using a Newton-Raphson method with backtracking line search.

A detailed description of the mathematical formulation, finite-element implementation, constitutive models, and numerical methodology, and validation studies is available in `docs/Report.pdf`.

## Material models

Three hyperelastic material models are available, sharing the same Newton solver (`LVBase`) and differing only in the strain energy function and in how fiber directions are computed:

- **Holzapfel-Ogden** — anisotropic model with fiber, sheet and normal directions. Fiber directions are obtained by solving a Laplace problem for the transmural coordinate and applying Gram-Schmidt orthogonalization.
- **Guccione** — transversely isotropic model. Fiber directions are computed analytically from prolate spheroidal coordinates.
- **Neo-Hooke** — isotropic model, mainly used as a simpler reference case.

In addition, `cardiac_beam` runs the Guccione-type energy (same form, own set of parameters in `constants::beam`) on a simple 3D beam geometry instead of the ventricle, and is used for validation and convergence checks.

All material and solver constants (material parameters, pressure range, Newton/GMRES settings, boundary IDs, fiber angles) are centralized in `src/constants.hpp`, one namespace per model.


## Boundary IDs

| ID | Condition |
|----|-----------|
| 2  | Dirichlet — zero displacement (apex ring) |
| 3  | Neumann — follower pressure (endocardium) |
| 4  | Robin — epicardial spring |

## Dependencies

- C++17 compiler
- MPI
- deal.II >= 9.3.1, built with Trilinos support
- Trilinos (Sacado, used for automatic differentiation of the strain energy)
- Boost >= 1.72 (filesystem, iostreams, serialization)
- gmsh

## Mesh generation

The ventricular and beam geometries are stored as `.geo` files under `geometry/` and need to be converted to deal.II-compatible `.msh` files before running. This conversion is triggered automatically by CMake: running `cmake ..` calls `scripts/generate_meshes.sh`, which uses gmsh to generate the meshes into `mesh/ventricular_meshes/` and `mesh/beam/`.

If this step fails, for example because gmsh is not installed or not in `PATH` at configure time, CMake only prints a warning and continues. In that case install gmsh and run the script manually before running:

```bash
bash scripts/generate_meshes.sh
```

The script skips meshes that are already up to date with respect to their `.geo` file, so it can be re-run safely at any time.

## Build

```bash
mkdir -p build && cd build
cmake ..
make
```

This produces four executables, one per material model: `cardiac_holzapfel`, `cardiac_guccione`, `cardiac_neohooke`, `cardiac_beam`.

## Run

The number of MPI ranks can be adjusted based on the available hardware resources.

```bash
cd build
mpirun -n 4 ./cardiac_holzapfel
mpirun -n 4 ./cardiac_guccione
mpirun -n 4 ./cardiac_neohooke
mpirun -n 4 ./cardiac_beam
```

Each executable also accepts a `--convergence` flag, which runs a self-convergence study on a sequence of meshes of decreasing size instead of the normal simulation, and writes the results to a CSV file:

```bash
mpirun -n 4 ./cardiac_guccione --convergence
```

The mesh paths used in both modes are set directly in the corresponding `src/main_*.cpp` file.

## Output

Parallel VTU output is written to the working directory at each pressure step:

- `output-<mesh_stem>_<step>.pvtu`, to be opened in ParaView
