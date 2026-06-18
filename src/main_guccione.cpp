#include "cardiac/lv.hpp"
#include "cardiac/strain_energy.hpp"
#include "cardiac/fiber_field.hpp"
#include "constants.hpp"
#include "cardiac/Poisson.hpp"
#include <iostream>
#include <string>
#include <vector>

static LVBase::SolverParams guccione_params()
{
  namespace ccg = cardiac::constants::guccione;
  LVBase::SolverParams p;
  p.p_start                    = ccg::P_START;
  p.p_end                      = ccg::P_END;
  p.n_steps                    = ccg::N_STEPS;
  p.gmres_max_iter             = ccg::GMRES_MAX_ITER;
  p.gmres_tol_factor           = ccg::GMRES_TOL_FACTOR;
  p.gmres_restart              = ccg::GMRES_RESTART;
  p.amg_elliptic               = ccg::AMG_ELLIPTIC;
  p.amg_aggregation_threshold  = ccg::AMG_AGG_THRESHOLD;
  p.amg_extract_constant_modes = ccg::AMG_CONST_MODES;
  p.alpha_robin                = ccg::ALPHA_ROBIN;
  return p;
}

std::unique_ptr<cardiac::LaplaceFibers> compute_laplace_fibers(cardiac::LV& model){
    DoFHandler<3> dof_handler_poisson;
  TrilinosWrappers::MPI::Vector lambda;

  Poisson poisson(model.get_mesh(), lambda,
                  model.get_fe(), model.get_quadrature(), model.get_quadrature_face(),
                  model.get_pcout());
  poisson.setup();
  poisson.assemble();
  poisson.solve();
  poisson.output();

  dof_handler_poisson.reinit(model.get_mesh());
  dof_handler_poisson.distribute_dofs(*model.get_fe());
  IndexSet locally_relevant;
  DoFTools::extract_locally_relevant_dofs(dof_handler_poisson, locally_relevant);

  TrilinosWrappers::MPI::Vector lambda_ghost(
      lambda.locally_owned_elements(), locally_relevant, MPI_COMM_WORLD);
  lambda_ghost = lambda;

  return std::make_unique<cardiac::LaplaceFibers>(
      dof_handler_poisson, lambda_ghost, *model.get_fe(), *model.get_quadrature());

}

int main(int argc, char *argv[]) {
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  if (argc > 1 && std::string(argv[1]) == "--convergence") {
    const std::vector<std::string> ventricle_meshes = {
      "../mesh/ventricular_meshes/ventricle_0_99.msh",
      "../mesh/ventricular_meshes/ventricle_0_7.msh",
      "../mesh/ventricular_meshes/ventricle_0_5.msh",
      "../mesh/ventricular_meshes/ventricle_0_3.msh",
    };
    cardiac::LV::run_convergence_study(
        ventricle_meshes, 2, 
        [](cardiac::LV&) {     
          return std::make_pair(
              std::make_unique<cardiac::GuccioneEnergy>(),
              std::make_unique<cardiac::AnalyticalProlateFibers>());
        },
        guccione_params(),
        cardiac::constants::guccione::CONVERGENCE_STUDY_STEPS,
        "convergence_ventricle.csv");
    return 0;
  }

  const std::string mesh = "../mesh/ventricular_meshes/ventricle_0_3.msh";

  cardiac::LV model(mesh, 2,
                    std::make_unique<cardiac::GuccioneEnergy>(),
                    std::make_unique<cardiac::AnalyticalProlateFibers>(),
                    guccione_params());
  model.setup();

  model.output_fibers("fibers_ventricle_prolate_0_7");
  
  model.solve();
  return 0;
}
