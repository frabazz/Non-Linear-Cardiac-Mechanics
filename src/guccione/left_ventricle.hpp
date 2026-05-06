#pragma once

#include "../lv_base.hpp"

#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/base/convergence_table.h>

namespace guccione {

class LV : public LVBase {
public:
  static SolverParams default_params() {
    namespace ccg = cardiac::constants::guccione;
    SolverParams p;
    p.p_start                  = ccg::P_START;
    p.p_end                    = ccg::P_END;
    p.n_steps                  = ccg::N_STEPS;
    p.gmres_max_iter           = ccg::GMRES_MAX_ITER;
    p.gmres_tol_factor         = ccg::GMRES_TOL_FACTOR;
    p.gmres_restart            = ccg::GMRES_RESTART;
    p.amg_elliptic             = ccg::AMG_ELLIPTIC;
    p.amg_aggregation_threshold  = ccg::AMG_AGG_THRESHOLD;
    p.amg_extract_constant_modes = ccg::AMG_CONST_MODES;
    return p;
  }

  LV(const std::string &mesh_file_name_, const unsigned int &r_)
      : LVBase(mesh_file_name_, r_, default_params()) {}

  void solve() override { solve_loop(params.n_steps); }
  void solve(int num_steps) { solve_loop(num_steps); }

  struct SerialReplica {
    Triangulation<dim>                  tria;
    std::unique_ptr<FiniteElement<dim>> fe;
    std::unique_ptr<FESystem<dim>>      fs;
    std::unique_ptr<MappingFE<dim>>     mapping;
    std::unique_ptr<Quadrature<dim>>    quadrature;
    DoFHandler<dim>                     dof_handler;
    Vector<double>                      solution;
  };

  std::unique_ptr<SerialReplica> gather_to_rank0() const;

  static double compute_difference_serial(const SerialReplica  &a,
                                          const SerialReplica  &b,
                                          VectorTools::NormType norm);

  static double h_from_mesh_filename(const std::string &mesh_file);

  double compute_difference(const LV &reference, VectorTools::NormType norm) const;

  static void run_convergence_study(const std::vector<std::string> &mesh_files,
                                    unsigned int r,
                                    const std::string &csv_filename = "convergence.csv");

protected:
  void compute_rhs() override;
};

} // namespace guccione
