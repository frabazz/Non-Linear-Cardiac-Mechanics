#pragma once

#include "../lv_base.hpp"

namespace holzapfel {

class LV : public LVBase {
public:
  LV(const std::string &mesh_file_name_, const unsigned int &r_)
      : LVBase(mesh_file_name_, r_, SolverParams{}) {}

  void init_poisson();
  void solve() override;

protected:
  DoFHandler<dim>               dof_handler_poisson;
  TrilinosWrappers::MPI::Vector lambda_ghost;
  TrilinosWrappers::MPI::Vector lambda;

  void compute_rhs() override;
};

} // namespace holzapfel
