# Non-Linear Cardiac Mechanics

Parallel finite-element solver for nonlinear cardiac mechanics, implemented with deal.II, MPI, and Trilinos.

The code solves a nonlinear equilibrium problem for a 3D left-ventricular geometry under internal pressure loading, using a Newton–Raphson method with backtracking line search. Two hyperelastic material models are provided.

## Material models

### Holzapfel-Ogden (2009)
Anisotropic model with fiber (`f0`), sheet (`s0`), and normal (`n0`) directions. Fiber directions are computed by solving a Laplace problem (transmural coordinate λ) and applying Gram-Schmidt orthogonalization. The strain energy is:

$$W = \frac{a}{2b} \left(e^{b(I_1-3)}-1\right) + \sum_{i \in \{f,s\}} \frac{a_i}{2b_i}\left(e^{b_i(I_{4i}-1)^2}-1\right) + \frac{a_{fs}}{2b_{fs}}\left(e^{b_{fs} I_{8fs}^2}-1\right) + \frac{\kappa}{2}(J-1)^2$$

### Guccione (1991)
Transversely isotropic model. Fiber directions are computed analytically from prolate spheroidal coordinates. The strain energy is:

$$W = \frac{C}{2}\left(e^Q - 1\right) + \frac{\kappa}{2}(J-1)^2, \quad Q = b_{ff}E_{ff}^2 + b_{ss}E_{ss}^2 + \cdots$$

## Code structure

```
src/
├── constants.hpp               # all model and solver constants
├── cardiac_solver.hpp          # ISolver interface
├── material_model.hpp          # IMaterialModel interface
├── lv_base.hpp / lv_base.cpp   # shared Newton solver, assembly, output
├── common.hpp
├── system_assembler.*
├── holzapfel/
│   ├── left_ventricle.hpp/cpp  # Holzapfel LV: Poisson solve + compute_rhs
│   ├── tensor_utils.hpp/cpp    # Holzapfel-Ogden W(F) via AD
│   ├── Poisson.hpp/cpp         # transmural coordinate λ
│   └── main.cpp
└── guccione/
    ├── left_ventricle.hpp/cpp  # Guccione LV: compute_rhs + convergence study
    ├── tensor_utils.hpp/cpp    # Guccione W(F) via AD + prolate coords
    └── main.cpp
```

`LVBase` provides `setup()`, `assemble_system()`, `solve_linear_system()`, `line_search()`, `solve_newton()`, `solve_loop()`, and `output()`. Each model only implements `compute_rhs()`.

All numerical and physical constants (material parameters, pressure ranges, solver settings, boundary IDs, fiber angles, geometry) are centralized in `src/constants.hpp` under `cardiac::constants::holzapfel` and `cardiac::constants::guccione`.

## Boundary IDs

| ID | Condition |
|----|-----------|
| 2  | Dirichlet — zero displacement (apex ring) |
| 3  | Neumann — follower pressure (endocardium) |
| 4  | Robin — epicardial spring |

## Build requirements

- C++17 compiler
- MPI
- deal.II ≥ 9.3.1 (with Trilinos enabled)
- Boost ≥ 1.72 (filesystem, iostreams, serialization)
- Sacado (from Trilinos)

## Build

```bash
mkdir -p build && cd build
cmake ..
make
```

This produces three executables: `cardiac_holzapfel`, `cardiac_guccione`, and `cardiac` (demo showing both models via the common `ISolver` interface).

## Run
The number of MPI ranks can be adjusted based on the available hardware resources.
```bash
cd build
mpirun -n 4 ./cardiac_holzapfel
mpirun -n 4 ./cardiac_guccione
```

The mesh path is set in the respective `main.cpp`.

## Output

Parallel VTU output written to the working directory at each pressure step:

- `output-<mesh_stem>_<step>.pvtu` — open in ParaView
- Scalar field `solution_mag` = $|u|$ included alongside the displacement vector

## Convergence study

```cpp
guccione::LV::run_convergence_study({"mesh_coarse.msh", "mesh_fine.msh"}, r, "convergence.csv");
```

Runs on all meshes, gathers solutions to rank 0, and computes self-convergence rates in L2 and H1.
