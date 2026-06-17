#include "Poisson.hpp"

void
Poisson::setup()
{
  pcout << "-----------------------------------------------" << std::endl;

    {
    pcout << "Initializing the DoF handler" << std::endl;
    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    // We retrieve the set of locally owned DoFs, which will be useful when
    // initializing linear algebra classes.
    locally_owned_dofs = dof_handler.locally_owned_dofs();

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  
  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    // For the sparsity pattern, we use Trilinos' class, which manages some of
    // the inter-process communication.
    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);

    // After initialization, we need to call compress, so that all processes
    // retrieve the information they need from the rows they own (i.e. the rows
    // corresponding to locally owned DoFs).
    sparsity.compress();

    pcout << "  Initializing the system matrix" << std::endl;
    // Since the sparsity pattern is partitioned by row, so will be the matrix.
    system_matrix.reinit(sparsity);

    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  }
}

void
Poisson::assemble()
{
  pcout << "===============================================" << std::endl;

  pcout << "  Assembling the linear system" << std::endl;

  // Number of local DoFs for each element.
  const unsigned int dofs_per_cell = fe->dofs_per_cell;

  // Number of quadrature points for each element.
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  // Local matrix and vector.
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  // Reset the global matrix and vector, just in case.
  system_matrix = 0.0;
  system_rhs    = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      // If current cell is not locally owned, we skip it.
      if (!cell->is_locally_owned())
        continue;

      // On all other cells (which are owned by current process) we perform
      // assembly as usual.

      fe_values.reinit(cell);

      cell_matrix = 0.0;
      cell_rhs    = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_matrix(i, j) += fe_values.shape_grad(i, q) * //
                                       fe_values.shape_grad(j, q) * //
                                       fe_values.JxW(q);
                }
            }
        }
    

    cell->get_dof_indices(dof_indices);
      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
    }

  // Each process might have written to some rows it does not own (for instance,
  // if it owns elements that are adjacent to elements owned by some other
  // process). Therefore, at the end of assembly, processes need to exchange
  // information: the compress method allows to do this.
  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  // Dirichlet boundary conditions.
  {
    std::map<types::global_dof_index, double> boundary_values;
    Functions::ZeroFunction<dim>              zero_function;
    Functions::ConstantFunction<dim>          one_function(1.0);

    std::map<types::boundary_id, const Function<dim> *> boundary_functions;
    boundary_functions[4] = &one_function;
    boundary_functions[3] = &zero_function;

    VectorTools::interpolate_boundary_values(dof_handler,
                                             boundary_functions,
                                             boundary_values);

    MatrixTools::apply_boundary_values(
      boundary_values, system_matrix, solution, system_rhs, false);
  }
}

void
Poisson::solve()
{
  pcout << "===============================================" << std::endl;

  TrilinosWrappers::PreconditionSSOR preconditioner;
  preconditioner.initialize(
    system_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  ReductionControl solver_control(/* maxiter = */ 10000,
                                  /* tolerance = */ 1.0e-16,
                                  /* reduce = */ 1.0e-6);

  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

  pcout << "  Solving the linear system" << std::endl;

  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  pcout << "  " << solver_control.last_step() << " CG iterations" << std::endl;
}

void
Poisson::output() const
{
  pcout << "===============================================" << std::endl;


  

  
    class LambdaGradientPostprocessor : public DataPostprocessorVector<dim>
  {
  public:
    LambdaGradientPostprocessor()
      : DataPostprocessorVector<dim>("lambda_gradient", update_gradients)
    {}

    virtual void
evaluate_scalar_field(
  const DataPostprocessorInputs::Scalar<dim> &inputs,
  std::vector<Vector<double>> &computed_quantities) const override
{
  for (unsigned int p = 0; p < inputs.solution_values.size(); ++p)
    {
      const double lambda = inputs.solution_values[p]; // d (distanza transmurale 0-1)
      
      // e2 transmural = grad lambda
      Tensor<1, dim> e2 = inputs.solution_gradients[p];
      e2 /= e2.norm(); 

      // e1 longitudinal ie z axis
      Tensor<1, dim> k_axis;
      k_axis[2] = 1.0;
    
      Tensor<1, dim> e1 = k_axis - (k_axis * e2) * e2;
      e1 /= e1.norm();

      Tensor<1, dim> e0 = cross_product_3d(e1, e2);
      e0 /= e0.norm();

      const double alpha_endo = 60.0  * M_PI / 180.0; 
      const double alpha_epi  = -60.0 * M_PI / 180.0;
      //const double beta_endo  = -30.0 * M_PI / 180.0; 
      //const double beta_epi   = 30.0  * M_PI / 180.0;

      const double alpha = alpha_endo * (1.0 - lambda) + alpha_epi * lambda;
      //      const double beta  = beta_endo  * (1.0 - lambda) + beta_epi  * lambda;

      Tensor<1, dim> f0    = std::cos(alpha) * e0 + std::sin(alpha) * e1;
      //Tensor<1, dim> g_hat = -std::sin(alpha) * e0 + std::cos(alpha) * e1;

      //      Tensor<1, dim> s0 = std::cos(beta) * g_hat + std::sin(beta) * e2;
      //Tensor<1, dim> n0 = -std::sin(beta) * g_hat + std::cos(beta) * e2;

      for (unsigned int d = 0; d < dim; ++d)
        computed_quantities[p](d) = f0[d]; 
    }
}
    
    

  };
  
  const IndexSet locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler);

  // To correctly export the solution, each process needs to know the solution
  // DoFs it owns, and the ones corresponding to elements adjacent to the ones
  // it owns (the locally relevant DoFs, or ghosts). We create a vector to store
  // them.
  TrilinosWrappers::MPI::Vector solution_ghost(locally_owned_dofs,
                                               locally_relevant_dofs,
                                               MPI_COMM_WORLD);

  // This assignment performs the necessary communication so that the locally
  // relevant DoFs are received from other processes and stored inside
  // solution_ghost.
  solution_ghost = solution;

  // Then we build and fill the DataOut class as usual, using the vector with
  // ghosts.
  DataOut<dim> data_out;

  data_out.add_data_vector(dof_handler, solution_ghost, "solution_poisson");

  // We also add a vector to represent the parallel partitioning of the mesh,
  // for the sake of visualization.
  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  LambdaGradientPostprocessor grad_post;
  data_out.add_data_vector(dof_handler, solution_ghost, grad_post);
  data_out.build_patches();

  const std::string output_file_name = "output_poisson";

  // Finally, we need to write in a format that supports parallel output. This
  // can be achieved in multiple ways (e.g. XDMF/H5). We choose VTU/PVTU files.
  data_out.write_vtu_with_pvtu_record(/* folder = */ "./",
                                      /* basename = */ output_file_name,
                                      /* index = */ 0,
                                      MPI_COMM_WORLD);

  pcout << "Output written to " << output_file_name << std::endl;

  pcout << "===============================================" << std::endl;
}
