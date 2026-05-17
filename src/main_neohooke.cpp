#include "cardiac/lv.hpp"
#include "cardiac/strain_energy.hpp"
#include "cardiac/fiber_field.hpp"

int main(int argc, char *argv[]) {
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const std::string mesh = "../ventricular_meshes/msh/ventricle_0_99.msh";

  LVBase::SolverParams params;
  params.alpha_robin = 0.0;

  cardiac::LV model(mesh, 2,
                    std::make_unique<cardiac::NeoHookeEnergy>(),
                    std::make_unique<cardiac::TrivialFibers>(),
                    params);
  model.setup();
  model.solve();
  return 0;
}
