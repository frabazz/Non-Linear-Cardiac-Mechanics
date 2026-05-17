#include "cardiac/lv.hpp"
#include "cardiac/strain_energy_holzapfel.hpp"
#include "cardiac/fibers_laplace.hpp"
#include "holzapfel/Poisson.hpp"
#include "constants.hpp"

#include <deal.II/dofs/dof_tools.h>

int main(int argc, char *argv[]) {
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const std::string mesh = "../ventricular_meshes/msh/ventricle_0_7.msh";

  LVBase::SolverParams params;
  params.alpha_robin = cardiac::constants::holzapfel::ALPHA_ROBIN;

  cardiac::LV model(mesh, 2,
                    std::make_unique<cardiac::HolzapfelOgdenEnergy>(),
                    nullptr,
                    params);
  model.setup();

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

  model.set_fibers(std::make_unique<cardiac::LaplaceFibers>(
      dof_handler_poisson, lambda_ghost, *model.get_fe(), *model.get_quadrature()));

  model.solve();
  return 0;
}
