#include "cardiac/lv.hpp"
#include "cardiac/strain_energy_guccione.hpp"
#include "cardiac/fibers_analytical.hpp"
#include "constants.hpp"
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

int main(int argc, char *argv[]) {
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  if (argc > 1 && std::string(argv[1]) == "--convergence") {
    const std::vector<std::string> ventricle_meshes = {
      "../ventricular_meshes/msh/ventricle_0_99.msh",
      "../ventricular_meshes/msh/ventricle_0_7.msh",
      "../ventricular_meshes/msh/ventricle_0_5.msh",
      "../ventricular_meshes/msh/ventricle_0_3.msh",
    };
    cardiac::LV::run_convergence_study(
        ventricle_meshes, 2,
        []() {
          return std::make_pair(
              std::make_unique<cardiac::GuccioneEnergy>(),
              std::make_unique<cardiac::AnalyticalProlateFibers>());
        },
        guccione_params(),
        cardiac::constants::guccione::CONVERGENCE_STUDY_STEPS,
        "convergence_ventricle.csv");
    return 0;
  }

  const std::string mesh = "../ventricular_meshes/msh/ventricle_0_99.msh";

  cardiac::LV model(mesh, 2,
                    std::make_unique<cardiac::GuccioneEnergy>(),
                    std::make_unique<cardiac::AnalyticalProlateFibers>(),
                    guccione_params());
  model.setup();
  model.solve();
  return 0;
}
