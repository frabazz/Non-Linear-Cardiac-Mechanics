#include "left_ventricle.hpp"
#include <iostream>
#include <string>
#include <vector>

using guccione::LV;


int main(int argc, char* argv[]){
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  //initialize MPI environment

// This program solves a *nonlinear* problem (cardiac-like geometry) in parallel.
// Pipeline:
//   1) setup():    read mesh, build FE space/DoFs, sparsity patter, vectors, matrices...
//   2) solve_newton(): repeatedly assemble residual+Jacobian and solve for updates
//   3) output():   write solution to VTU/PVTU for visualization
// MPI_InitFinalize keeps MPI alive for the entirity of main()


  // Self-convergence study on ventricular meshes:
  //   ./executable --convergence
  if (argc > 1 && std::string(argv[1]) == "--convergence")
    {
      const std::vector<std::string> ventricle_meshes = {
        "../ventricular_meshes/msh/ventricle_0_99.msh", // h = 0.99 (coarsest)
        "../ventricular_meshes/msh/ventricle_0_7.msh",  // h = 0.7
        "../ventricular_meshes/msh/ventricle_0_5.msh",  // h = 0.5
        "../ventricular_meshes/msh/ventricle_0_3.msh",  // h = 0.3 (finest, reference)
      };
      LV::run_convergence_study(ventricle_meshes, 2, "convergence_ventricle.csv");
      return 0;
    }


  std::string ventricular_mesh_path = "../ventricular_meshes/msh/ventricle_0_99.msh";
 // ventricular meshes created with geo file in mesh/ventricular_meshes/geo
  
  std::string cubic_mesh_path = "../mesh/cube_0_2.msh";

  


  LV model = LV(ventricular_mesh_path, 2); //2 is the polynomial degree
  


  model.setup();


  model.solve(); //solve the nonlinear problem by (pseudo)Newton+line search


  
  
  return 0;
}
