#include "cardiac/lv.hpp"
#include "cardiac/strain_energy.hpp"
#include "cardiac/strain_energy.hpp"
#include "cardiac/fiber_field.hpp"
#include "cardiac/Poisson.hpp"
#include "constants.hpp"

#include <deal.II/dofs/dof_tools.h>
#include <iostream>
#include <memory>
#include <string>

// Usage: ./cardiac [holzapfel|guccione] [mesh_path]
//   default model : holzapfel
//   default mesh  : ../ventricular_meshes/msh/ventricle_0_99.msh

int main(int argc, char *argv[]) {
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const std::string model_name = (argc > 1) ? std::string(argv[1]) : "holzapfel";
  const std::string mesh       = (argc > 2) ? std::string(argv[2])
                                             : "../ventricular_meshes/msh/ventricle_0_3.msh";
  const unsigned int degree = 2;

  if (model_name == "guccione") {
    namespace ccg = cardiac::constants::guccione;
    LVBase::SolverParams params;
    params.p_start                    = ccg::P_START;
    params.p_end                      = ccg::P_END;
    params.n_steps                    = ccg::N_STEPS;
    params.gmres_max_iter             = ccg::GMRES_MAX_ITER;
    params.gmres_tol_factor           = ccg::GMRES_TOL_FACTOR;
    params.gmres_restart              = ccg::GMRES_RESTART;
    params.amg_elliptic               = ccg::AMG_ELLIPTIC;
    params.amg_aggregation_threshold  = ccg::AMG_AGG_THRESHOLD;
    params.amg_extract_constant_modes = ccg::AMG_CONST_MODES;
    params.alpha_robin                = ccg::ALPHA_ROBIN;

    cardiac::LV solver(mesh, degree,
                        std::make_unique<cardiac::GuccioneEnergy>(),
                        std::make_unique<cardiac::AnalyticalProlateFibers>(),
                        params);
    solver.setup();
    solver.solve();

  } else {
    LVBase::SolverParams params;
    params.alpha_robin = cardiac::constants::holzapfel::ALPHA_ROBIN;

    cardiac::LV solver(mesh, degree,
                        std::make_unique<cardiac::HolzapfelOgdenEnergy>(),
                        nullptr,
                        params);
    solver.setup();

    DoFHandler<3> dof_handler_poisson;
    TrilinosWrappers::MPI::Vector lambda;

    Poisson poisson(solver.get_mesh(), lambda,
                    solver.get_fe(), solver.get_quadrature(), solver.get_quadrature_face(),
                    solver.get_pcout());
    poisson.setup();
    poisson.assemble();
    poisson.solve();
    poisson.output();

    dof_handler_poisson.reinit(solver.get_mesh());
    dof_handler_poisson.distribute_dofs(*solver.get_fe());
    IndexSet locally_relevant;
    DoFTools::extract_locally_relevant_dofs(dof_handler_poisson, locally_relevant);

    TrilinosWrappers::MPI::Vector lambda_ghost(
        lambda.locally_owned_elements(), locally_relevant, MPI_COMM_WORLD);
    lambda_ghost = lambda;

    solver.set_fibers(std::make_unique<cardiac::LaplaceFibers>(
        dof_handler_poisson, lambda_ghost, *solver.get_fe(), *solver.get_quadrature()));

    solver.solve();
  }

  return 0;
}
