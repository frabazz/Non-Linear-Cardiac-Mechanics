#ifndef DIFFUSION_REACTION_HPP
#define DIFFUSION_REACTION_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace dealii;

/**
 * Class managing the differential problem.
 */
class Poisson
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;

  // Constructor.
  Poisson(
    parallel::fullydistributed::Triangulation<dim> &mesh_,
    TrilinosWrappers::MPI::Vector &solution_,
    const std::unique_ptr<FiniteElement<dim>>& fe_,
    const std::unique_ptr<Quadrature<dim>>& quadrature_,
    const std::unique_ptr<Quadrature<dim-1>>& quadrature_boundary_,
    ConditionalOStream& pcout_
    )
    : mesh(mesh_)
    , solution(solution_)
    , fe(fe_)
    , quadrature(quadrature_)
    , quadrature_boundary(quadrature_boundary_)
    , pcout(pcout_)
  {}

  // Initialization.
  void
  setup();

  // System assembly.
  void
  assemble();

  // System solution.
  void
  solve();

  // Output.
  void
  output() const;

  // Compute the error against a given exact solution.
  double
  compute_error(const VectorTools::NormType &norm_type,
                const Function<dim>         &exact_solution) const;

protected:

  // Triangulation. The parallel::fullydistributed::Triangulation class manages
  // a mesh that is completely distributed across all MPI processes (i.e. each
  // process only stores its own locally relevant cells).
  parallel::fullydistributed::Triangulation<dim>& mesh;

    // System solution.
  TrilinosWrappers::MPI::Vector& solution;

  
  // Finite element space.
  const std::unique_ptr<FiniteElement<dim>>& fe;

  // Quadrature formula.
  const std::unique_ptr<Quadrature<dim>>& quadrature;

  // Quadrature formula for boundary integrals.
  const std::unique_ptr<Quadrature<dim - 1>>& quadrature_boundary;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // System matrix.
  TrilinosWrappers::SparseMatrix system_matrix;

  // System right-hand side.
  TrilinosWrappers::MPI::Vector system_rhs;


  // Output stream for process 0.
  ConditionalOStream& pcout;

  // Locally owned DoFs for current process.
  IndexSet locally_owned_dofs;
};

#endif
