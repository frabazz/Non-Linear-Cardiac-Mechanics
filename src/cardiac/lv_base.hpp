#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/differentiation/ad/ad_drivers.h>
#include <deal.II/differentiation/ad/ad_helpers.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>

#include "../common.hpp"
#include "../constants.hpp"

using namespace dealii;

class LVBase{
public:
  static constexpr unsigned int dim = 3;

  struct SolverParams {
    double p_start = cardiac::constants::holzapfel::P_START;
    double p_end   = cardiac::constants::holzapfel::P_END;
    int    n_steps = cardiac::constants::holzapfel::N_STEPS;

    unsigned int gmres_max_iter      = cardiac::constants::holzapfel::GMRES_MAX_ITER;
    double       gmres_tol_factor    = cardiac::constants::holzapfel::GMRES_TOL_FACTOR;
    unsigned int gmres_restart       = cardiac::constants::holzapfel::GMRES_RESTART;

    bool   amg_elliptic               = cardiac::constants::holzapfel::AMG_ELLIPTIC;
    double amg_aggregation_threshold  = cardiac::constants::holzapfel::AMG_AGG_THRESHOLD;
    bool   amg_extract_constant_modes = cardiac::constants::holzapfel::AMG_CONST_MODES;

    double alpha_robin = cardiac::constants::holzapfel::ALPHA_ROBIN;
  };

  class ForcingTerm : public Function<dim> {
  public:
    ForcingTerm() {}
  };

  class FunctionG : public Function<dim> {
  public:
    FunctionG() : Function<dim>(dim) {}
    virtual double value(const Point<dim> &, const unsigned int = 0) const override{
      return 0.0;
    }
  };

  FunctionG function_g;

  LVBase(const std::string &mesh_file_name_,
         const unsigned int &r_,
         const SolverParams &params_);

  void setup();

  parallel::fullydistributed::Triangulation<dim>& get_mesh() { return mesh; }
  const std::unique_ptr<FiniteElement<dim>>&       get_fe()  const { return fe; }
  const std::unique_ptr<Quadrature<dim>>&          get_quadrature() const { return quadrature; }
  const std::unique_ptr<Quadrature<dim - 1>>&      get_quadrature_face() const { return quadrature_face; }
  ConditionalOStream&                              get_pcout() { return pcout; }
  void assemble_system();
  void solve_linear_system();
  void solve_newton();
  void output(unsigned int step) const;

protected:
  const std::string  mesh_file_name;
  const unsigned int r;
  const unsigned int mpi_size;
  const unsigned int mpi_rank;

  SolverParams params;

  ForcingTerm forcing_term;
  double      pressure      = 0.0;
  int         last_iteration = 0;

  parallel::fullydistributed::Triangulation<dim> mesh;

  std::unique_ptr<FiniteElement<dim>>    fe;
  std::unique_ptr<FESystem<dim>>         fs;
  std::unique_ptr<Quadrature<dim>>       quadrature;
  std::unique_ptr<Quadrature<dim - 1>>   quadrature_face;

  DoFHandler<dim> dof_handler;

  TrilinosWrappers::SparseMatrix      system_matrix;
  TrilinosWrappers::SparseMatrix      jacobian_matrix;
  TrilinosWrappers::MPI::Vector       system_rhs;
  TrilinosWrappers::MPI::Vector       solution;
  TrilinosWrappers::MPI::Vector       solution_owned;
  TrilinosWrappers::MPI::Vector       delta_owned;

  ConditionalOStream pcout;
  IndexSet           locally_owned_dofs;
  IndexSet           locally_relevant_dofs;

  virtual void compute_rhs() = 0;

  void solve_loop(int num_steps);

  struct LineSearchResult {
    bool   accepted;
    bool   stagnated;
    double alpha;
    double residual;
  };

  LineSearchResult line_search(const TrilinosWrappers::MPI::Vector &solution_prev,
                                const TrilinosWrappers::MPI::Vector &delta_prev,
                                double                               residual_prev);

  virtual ~LVBase() = default;
};
