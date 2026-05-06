#include "holzapfel/left_ventricle.hpp"
#include "guccione/left_ventricle.hpp"
#include "cardiac_solver.hpp"
#include <iostream>
#include <memory>
#include <string>

// Demonstrates that both material models share the same ISolver interface
// and can be driven by identical code.
//
// Usage: ./cardiac_demo [holzapfel|guccione] [mesh_path]
//   default model : holzapfel
//   default mesh  : ../ventricular_meshes/msh/ventricle_0_99.msh

static void run_solver(cardiac::ISolver &solver) {
  solver.setup();
  solver.solve();
}

int main(int argc, char *argv[]) {
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const std::string model_name = (argc > 1) ? std::string(argv[1]) : "holzapfel";
  const std::string mesh       = (argc > 2) ? std::string(argv[2])
                                             : "../ventricular_meshes/msh/ventricle_0_99.msh";
  const unsigned int degree = 2;

  if (model_name == "guccione") {
    guccione::LV solver(mesh, degree);
    run_solver(solver);
  } else {
    holzapfel::LV solver(mesh, degree);
    run_solver(solver);
  }

  return 0;
}
