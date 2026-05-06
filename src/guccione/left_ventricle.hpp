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
#include "../cardiac_solver.hpp"

using namespace dealii;

namespace guccione {

// Class managing the differential problem.

class LV : public cardiac::ISolver {
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;
  double compute_pressure(const Point<dim> &) const; //todo it will be interesting to compute pressure not as a constant but this should be done as a ultimate change

  // Self-convergence: ||u_this - u_reference|| in the chosen norm.

  double compute_difference(const LV &reference,
                            VectorTools::NormType norm) const;

  // Drives a self-convergence study over a list of mesh files.
  // Each solve runs in full MPI parallel; the post-solve comparison is done
  // serially on rank 0 using gathered SerialReplica objects.
  static void run_convergence_study(const std::vector<std::string> &mesh_files,
                                    unsigned int r,
                                    const std::string &csv_filename =
                                      "convergence.csv");

  // Serial in-memory copy of solver state needed by FEFieldFunction:
  struct SerialReplica {
    Triangulation<dim>                  tria;
    std::unique_ptr<FiniteElement<dim>> fe;
    std::unique_ptr<FESystem<dim>>      fs;
    std::unique_ptr<MappingFE<dim>>     mapping;
    std::unique_ptr<Quadrature<dim>>    quadrature;
    DoFHandler<dim>                     dof_handler;
    Vector<double>                      solution;
  };

  // Gather the parallel solution to rank 0 and rebuild a serial replica
  
  std::unique_ptr<SerialReplica> gather_to_rank0() const;

  static double compute_difference_serial(const SerialReplica  &a,
                                          const SerialReplica  &b,
                                          VectorTools::NormType norm);

  // Extract h from mesh filename 
   static double h_from_mesh_filename(const std::string &mesh_file);

  // Forcing term.
  class ForcingTerm : public Function<dim>{
  public:
    // Constructor.
    ForcingTerm() {}
  };

  // Dirichlet boundary conditions.
  class FunctionG : public Function<dim> {
  public:
    // Constructor.
    FunctionG() : Function<dim>(dim) {}

    // Evaluation of the boundary condition which in our setting is always going to be 0  (todo check if this is an overly semplification, I don't think it is)
    virtual double value(const Point<dim> & /*p*/, 
                         const unsigned int /*component*/ = 0) const override {
      return 0.0;
    }
  };

  FunctionG function_g;

  // Constructor.
  LV(const std::string &mesh_file_name_, const unsigned int &r_)
      : mesh_file_name(mesh_file_name_), r(r_),
        mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
        mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
        mesh(MPI_COMM_WORLD), pcout(std::cout, mpi_rank == 0) {}

  // Initialization.
  void setup() override;

  // System assembly.
  void assemble_system();

  // System solution.
  void solve_linear_system();

  void solve_newton();

  void solve() override { solve(100); }
  void solve(int num_steps);
  // Output.
  void output(unsigned int step) const;

protected:
  // Path to the mesh file.
  const std::string mesh_file_name;

  // Polynomial degree.
  const unsigned int r;

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // TODO forcing term
  ForcingTerm forcing_term;

  double pressure;
  int last_iteration;
  
  // Triangulation. The parallel::fullydistributed::Triangulation class manages
  // a triangulation that is completely distributed (i.e. each process only
  // knows about the elements it owns and its ghost elements).
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Finite System
  std::unique_ptr<FESystem<dim>> fs;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;
  std::unique_ptr<Quadrature<dim - 1>> quadrature_face;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // System matrix.
  TrilinosWrappers::SparseMatrix system_matrix;

  // Jacobian Matrix
  TrilinosWrappers::SparseMatrix jacobian_matrix;

  // System right-hand side.
  TrilinosWrappers::MPI::Vector system_rhs;

  // System solution.
  TrilinosWrappers::MPI::Vector solution;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;

  TrilinosWrappers::MPI::Vector delta_owned;

  // Parallel output stream.
  ConditionalOStream pcout;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs needed to assemble locally owned cells (owned + ghosts).
  IndexSet locally_relevant_dofs;

  void compute_rhs();

  
  
  struct LineSearchResult {
    bool accepted;
    bool stagnated;
    double alpha;
    double residual;
  };

  LineSearchResult line_search(
      const TrilinosWrappers::MPI::Vector &solution_prev,
      const TrilinosWrappers::MPI::Vector &delta_prev,
      const double residual_prev);


};

} // namespace guccione
