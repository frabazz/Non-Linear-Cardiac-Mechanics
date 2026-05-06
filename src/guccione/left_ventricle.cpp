#include "left_ventricle.hpp"
#include "tensor_utils.hpp"

#include <cmath>
#include <Sacado.hpp>
#include <iomanip>
#include <limits>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <deal.II/base/mpi.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/base/convergence_table.h>

namespace guccione {

void LV::setup() {
  last_iteration = 0;
  pcout << "===============================================" << std::endl;

  // Create the mesh.

// read mesh, partition the serial mesh across MPI ranks
// created a distributed triangulation (mesh) used by DoFHandler/assembly
  {
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);

    std::ifstream grid_in_file(mesh_file_name);
    AssertThrow(grid_in_file.good(),
          ExcMessage("Could not open mesh file: '" + mesh_file_name +
                 "'. Current working directory is '" + std::filesystem::current_path().string()));
    grid_in.read_msh(grid_in_file);

    GridTools::partition_triangulation(mpi_size, mesh_serial); 
    const auto construction_data = TriangulationDescription::Utilities::
        create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;

    // Debug: list boundary ids present in the mesh.
    //we had a lot of problems with boundary ids so this is useful to check them (2=dirichlet, 3=neumann,4=robin)
    {
      std::map<types::boundary_id, unsigned long long> boundary_face_counts;
      for (const auto &cell : mesh.active_cell_iterators()) {
        if (!cell->is_locally_owned())
          continue;
        if (!cell->at_boundary())
          continue;

        for (unsigned int f = 0; f < cell->n_faces(); ++f)
          if (cell->face(f)->at_boundary())
            ++boundary_face_counts[cell->face(f)->boundary_id()];
      }

      pcout << "  Boundary ids present (local face counts):";
      if (boundary_face_counts.empty())
        pcout << " boundary ids not found or not at boundary, problem!";
      for (const auto &[bid, cnt] : boundary_face_counts)
        pcout << " id=" << static_cast<unsigned int>(bid) << ":" << cnt;
      pcout << std::endl;
    }
  }
  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space. This is the same as in serial codes.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r); //scalar element of degree r
    fs = std::make_unique<FESystem<dim>>(*fe, dim); //takes the same logic as the previous but vectorial 


    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fs->dofs_per_cell
          << std::endl;

    check((fs->dofs_per_cell == expected_dofs_per_cell),
          "this needs to be fixed some how");

    quadrature = std::make_unique<QGaussSimplex<dim>>(fe->degree + 1);
    quadrature_face = std::make_unique<QGaussSimplex<dim - 1>>(fe->degree + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
    pcout << "  Quadrature points per face = " << quadrature_face->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fs);

    // We retrieve the set of locally owned DoFs, which will be useful when
    // initializing linear algebra classes.
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    pcout << "  Initializing the sparsity pattern" << std::endl;

    // To initialize the sparsity pattern, we use Trilinos' class, that manages
    // some of the inter-process communication.
    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    pcout << "  Initializing the Jacobian matrix" << std::endl;
    jacobian_matrix.reinit(sparsity); //allocate the jacobian matrix, will be used in the newton iteration

    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    delta_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);

// solution is just a ghost vector used to evaluate gradients during assembly(robin term).
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  }
}


//assemble residual and jacobian for the current solution
//used in newton iteration
void LV::assemble_system() {

// Assemble residual R(u) and Jacobian J(u) = dR/du for the current solution vector.
  compute_rhs();


// assemble each rank's contributions to the global system 
jacobian_matrix.compress(VectorOperation::add);
system_rhs.compress(VectorOperation::add);




  std::map<types::global_dof_index, double> boundary_values;
  Functions::ZeroFunction<dim> zero_function(dim);


     VectorTools::interpolate_boundary_values(dof_handler,
                                              2,
                                              zero_function,
                                              boundary_values);

     MatrixTools::apply_boundary_values(
       boundary_values, jacobian_matrix, delta_owned, system_rhs, false); //false needed for newton


}

void LV::compute_rhs() {
  //pcout << "===============================================" << std::endl;

  //pcout << "  Computing Right Hand Side" << std::endl;

  const unsigned int dofs_per_cell = fs->dofs_per_cell;
  const unsigned int n_q = quadrature->size();
  const unsigned int n_q_face = quadrature_face->size();


  //gives access to shape functions, gradients, JxW, at each quadrature point
  FEValues<dim> fe_values(*fs, *quadrature,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const FEValuesExtractors::Vector vec_index(0);

  FEFaceValues<dim> fe_face_values(*fs, *quadrature_face,
                                   update_values | update_normal_vectors |
                                       update_JxW_values | update_gradients | update_quadrature_points);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  //reset global matrix and rhs
  jacobian_matrix = 0.0;
  system_rhs = 0.0;


  std::vector<Tensor<1, dim>> solution_values_face(n_q_face);
    std::vector<Tensor<2, dim>> solution_gradient_loc_face(n_q_face);
    std::vector<Tensor<2, dim>> solution_gradient_loc(n_q);
    //those vectors will hold the gradient of the current solution at quadrature points

    TensorUtils t_utils; //needed to compute P and dP/dF through automatic differentiation


  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned())
      continue;

      //  binds FEValues to this specific cell and computes 
      //  all requested FE data at quadrature points.
    fe_values.reinit(cell);

    cell_rhs = 0.0;
    cell_matrix = 0.0;

    cell->get_dof_indices(dof_indices);

    fe_values[vec_index].get_function_gradients(solution,
                                                solution_gradient_loc);

    Tensor<1,expected_dofs_per_cell,VecADNumberType> local_solution;
    for (unsigned int i = 0; i < expected_dofs_per_cell; ++i)
      local_solution[i] = solution(dof_indices[i]);

    auto local_rhs_assembler =
      [&](Tensor<1,expected_dofs_per_cell,VecADNumberType>& local_u) {
        //volume quadrature loop
          for (unsigned int q = 0; q < n_q; ++q) {

            Tensor<2, dim, VecADNumberType> F;

            F.clear();// sanity check;
            for(unsigned int i = 0;i < dim; ++i)
              F[i][i] = VecADNumberType(1.0);
            

            //todo remove debugging. But it gave me so much headaches that I leave it for now
            for(unsigned int i = 0;i < dofs_per_cell; ++i){
              const Tensor<2, dim> grad_phi_i =
                  fe_values[vec_index].gradient(i, q);
               F += local_u[i] * grad_phi_i;
            }
            
            //debugging det(F)
            Tensor<2, dim> F_d;
            for (unsigned int ii = 0; ii < dim; ++ii)
              for (unsigned int jj = 0; jj < dim; ++jj)
                F_d[ii][jj] = Sacado::ScalarValue<VecADNumberType>::eval(F[ii][jj]);

            const double det_F = determinant(F_d);
            AssertThrow(std::isfinite(det_F) && det_F > 1e-12,
                        ExcMessage("Non-positive or near-singular det(F) in volume quadrature"));

            Tensor<2, dim> P;
            Tensor<4, dim> dP_dF;


            //for now it is implemented only the isotropic model (neo Hooke)
            //todo work on guccione anisotropic model


            // P is the PK stress tensor and is calculed by differentiating a neo-Hookean strain energy W(F)
            // dP_dF is the material elasticity tensor (4th order) needed for the jacobian assembly
            t_utils.compute_tensors(F_d, fe_values.quadrature_point(q), P, dP_dF); //computation through AD

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
              const Tensor<2, dim> grad_phi_i =
                  fe_values[vec_index].gradient(i, q);
              cell_rhs(i) += scalar_product(P, grad_phi_i) * fe_values.JxW(q);
              //internal residuals assembly

              for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                const Tensor<2, dim> grad_phi_j =
                    fe_values[vec_index].gradient(j, q);

                Tensor<2, dim> dP;
                dP.clear();
                for (unsigned int a = 0; a < dim; ++a)
                  for (unsigned int b = 0; b < dim; ++b)
                    for (unsigned int c = 0; c < dim; ++c)
                      for (unsigned int d = 0; d < dim; ++d)
                        dP[a][b] += dP_dF[a][b][c][d] * grad_phi_i[c][d];


                //newton jacobian assembly
                cell_matrix(i, j) +=
                    scalar_product(dP, grad_phi_j) * fe_values.JxW(q);
              }
            }
          }

          //quite unsure of these boundary terms, they were applied in the guccione example
          //105.0; //in mmhg //d'ore not converge with correct pressure 
          double alpha = 0.999918;

           if (cell->at_boundary()) {
             for (unsigned int f = 0; f < cell->n_faces(); ++f) {
               if (cell->face(f)->at_boundary() &&
                   cell->face(f)->boundary_id() == 3) { //neumann BC
                 fe_face_values.reinit(cell, f);
                 fe_face_values[vec_index].get_function_gradients(
                     solution, solution_gradient_loc_face);

                  Tensor<2, dim> eye= Tensor<2, dim>();
                  for (unsigned int d = 0; d < dim; ++d)
                    eye[d][d] = 1.0;


                 for (unsigned int q = 0; q < n_q_face; ++q) {
                   {
                     
                     Tensor<2, dim> Fh;

                     Fh.clear();
                     for (unsigned int d = 0; d < dim; ++d)
                       Fh[d][d] = 1.0;
                     Fh += solution_gradient_loc_face[q];
                        
                     const double det_Fh = determinant(Fh);
                     AssertThrow(std::isfinite(det_Fh) && det_Fh > 0.0,
                           ExcMessage(
                             "Non-positive or NaN det(Fh) on boundary face -> cannot invert Fh"));

                    Tensor<2, dim> aux_F = transpose(invert(Fh));
                     Tensor<2, dim> H = det_Fh * aux_F;

                     for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                       const Tensor<1, dim> term1 =
                           H * fe_face_values.normal_vector(q);
                        const Tensor<1, dim> phi_i = fe_face_values[vec_index].value(i, q);
                    cell_rhs(i) += pressure * scalar_product(term1, phi_i) * fe_face_values.JxW(q);
                   

                      for (unsigned int j=0 ;j < dofs_per_cell; ++j){
                        const Tensor<2,dim> grad_phi_j = fe_face_values[vec_index].gradient(j, q);
                        const double prod = scalar_product (grad_phi_j,aux_F);

                        Tensor<2,dim> term2;
                        
                        for (unsigned int d=0; d<dim; ++d)
                            term2[d][d] = prod;

                        const Tensor<2,dim> term3 = aux_F * transpose(grad_phi_j);

                        const Tensor<2,dim> term = term2 - term3;

                        const Tensor<1, dim> H_N = H * fe_face_values.normal_vector(q);

                        const Tensor<1, dim> term4 = term * H_N;

                          cell_matrix(i,j) += pressure * scalar_product(term4, phi_i) * fe_face_values.JxW(q);


                      }
                    }
                   }
                 }
               }

               // ROBIN TERM
               
               
               if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id()== 4) 
               {
                
               fe_face_values.reinit(cell,f);
       
                 fe_face_values[vec_index].get_function_values(solution, solution_values_face);
                 
                 for (unsigned int q=0; q<n_q_face;++q){
                 const Tensor<1, dim> u_q = solution_values_face[q];
       
                 for (unsigned int i=0;i<dofs_per_cell;++i){
                   const Tensor<1, dim> phi_i = fe_face_values[vec_index].value(i, q);
                   cell_rhs(i) += alpha * scalar_product(u_q, phi_i) * 
                   fe_face_values.JxW(q);
                  // removed a reduntatnt for loop, iterated two times over i
                   for (unsigned int j=0;j<dofs_per_cell;++j){
                     const Tensor<1, dim> phi_i = fe_face_values[vec_index].value(i, q);
                     const Tensor<1, dim> phi_j = fe_face_values[vec_index].value(j, q);
                     cell_matrix(i,j) += alpha * scalar_product(phi_i, phi_j) * 
                                         fe_face_values.JxW(q);
                                        }
                                }
                   }
                  }                
                }


           }
        };

        local_rhs_assembler(local_solution);

    

        cell->get_dof_indices(dof_indices);

    jacobian_matrix.add(dof_indices, cell_matrix);
//    printf("the number of nonzero entries is %d\n", jacobian_matrix.n_nonzero_elements());
    system_rhs.add(dof_indices, cell_rhs);
  }
}
// removed commented out Dirichlet BCs which are not needed here. 
// they are applied in solve_newton instead

void LV::solve_linear_system() {

  //initial guess for the Newton increment.
  delta_owned = 0.0;
  delta_owned.compress(VectorOperation::insert);   //assigning the delta_owned vector to a Trilinos vector

  SolverControl solver_control(15000, 1.5e-4 * system_rhs.l2_norm());

  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);
  TrilinosWrappers::PreconditionAMG preconditioner;
  preconditioner.initialize(
      jacobian_matrix, TrilinosWrappers::PreconditionAMG::AdditionalData(1.0));

  solver.solve(jacobian_matrix, delta_owned, system_rhs, preconditioner);
  last_iteration = solver_control.last_step();
  pcout << "  " << solver_control.last_step() << " GMRES iterations "
      << "(final residual " << std::scientific << std::setprecision(3)
      << solver_control.last_value() << ")" << std::endl;
        
}

LV::LineSearchResult
LV::line_search(const TrilinosWrappers::MPI::Vector &solution_prev,
                const TrilinosWrappers::MPI::Vector &delta_prev,
                const double residual_prev) {
  // Backtracking line search to avoid element inversion (det(F) <= 0) and to ensure the residual decreases.


  double alpha_ls = 1.0;
  const unsigned int max_backtracks = 25;
  const double rel_decrease_eps = 1e-8;
  const double abs_decrease_eps = 1e-14;


  const double alpha_min = 1e-7;

  bool last_trial_assembled = false;
  double last_trial_residual = std::numeric_limits<double>::infinity();

  TrilinosWrappers::MPI::Vector last_good_trial = solution_prev;
  double alpha_last_good = 0.0;

  for (unsigned int linesearch_iter = 0; linesearch_iter < max_backtracks;++linesearch_iter) {
    TrilinosWrappers::MPI::Vector trial = solution_prev;
    trial.add(-alpha_ls, delta_prev);

    solution_owned = trial;
    solution = trial;

    bool assembly_ok = true;
    try {
      assemble_system();
    } catch (const dealii::ExceptionBase &) {
      assembly_ok = false;
    } catch (const std::exception &) {
      assembly_ok = false;
    }

    if (!assembly_ok) {
      solution_owned = solution_prev;
      solution = solution_prev;
      alpha_ls *= 0.5;
      continue;
    }

    last_trial_assembled = true;
    const double residual_trial = system_rhs.l2_norm();
    last_trial_residual = residual_trial;

    last_good_trial = trial;
    alpha_last_good = alpha_ls;

    // Accept if we get a meaningful decrease
    if (std::isfinite(residual_trial) &&
        (residual_trial < residual_prev * (1.0 - rel_decrease_eps) ||
         residual_trial < residual_prev - abs_decrease_eps)) {
      //      pcout << "  (alpha=" << std::scientific << std::setprecision(2) << alpha_ls
      //      << ")" << std::endl;
      return {true, false, alpha_ls, residual_trial};
    }
    alpha_ls *= 0.5;
  }

  // Not accepted
  if (last_trial_assembled && std::isfinite(last_trial_residual)) {
    // If we cannot decrease the residual and the step becomes tiny, we are in an infinite loop and that is pesos
    const double step_norm = alpha_last_good * delta_prev.l2_norm();
    const double sol_norm = solution_prev.l2_norm();
    const double step_tol = 1e-12 * (sol_norm + 1.0);

    if (step_norm <= step_tol || alpha_last_good <= alpha_min) {
      solution_owned = solution_prev;
      solution = solution_prev;
      pcout << "  (line search stagnated: alpha=" << std::scientific
            << std::setprecision(2) << alpha_last_good
            << ", step_norm=" << step_norm << ")" << std::endl;
      return {false, true, alpha_last_good, residual_prev};
    }

    // Otherwise, accept the smallest assembled step to avoid crashing,
    // but report it explicitly.
    solution_owned = last_good_trial;
    solution = last_good_trial;
    pcout << "  (alpha=" << std::scientific << std::setprecision(2)
          << alpha_last_good << ", no residual decrease)" << std::endl;
    return {false, false, alpha_last_good, last_trial_residual};
  }

  AssertThrow(false,
              ExcMessage(
                  "Line search failed: could not find a step length that keeps det(F)>0."));
  return {false, true, 0.0, residual_prev};
}


//solve the non linear system with newton's method + line search
void LV::solve_newton() {
  const unsigned int n_max_iters = 1000;
  const double residual_tolerance = 1e-3; // was 1e-5 but convergence is too slow

  unsigned int n_iter = 0;
  double residual_norm = residual_tolerance + 1;

  // We apply the boundary conditions to the initial guess (which is  stored in
  // solution_owned and solution).
  {
    std::set<types::boundary_id> dirichlet_boundary_id={2};
    ComponentMask dirichlet_mask =
        fs->component_mask(FEValuesExtractors::Vector(0)); //this is not actually necessary since we only have one component but now it never gives nans anymore

    IndexSet dirichlet_dofs =
        DoFTools::extract_boundary_dofs(dof_handler, dirichlet_mask,dirichlet_boundary_id);

    dirichlet_dofs = dirichlet_dofs & dof_handler.locally_owned_dofs();

    std::map<types::global_dof_index, double> boundary_values;
    //the boundary function is zero but this could work with any other function
    VectorTools::interpolate_boundary_values(dof_handler,
                                             2,
                                             function_g,
                                             boundary_values);

                                             

    for (const auto &idx : dirichlet_dofs)
      if (const auto it = boundary_values.find(idx);
          it != boundary_values.end())
      solution_owned[idx] = it->second;


    solution_owned.compress(VectorOperation::insert);
    solution = solution_owned;
  }

  

  bool stagnated = false;
  
  while (n_iter < n_max_iters && residual_norm > residual_tolerance) {
    assemble_system();
    residual_norm = system_rhs.l2_norm();

    //debugging
    AssertThrow(std::isfinite(residual_norm),
            ExcMessage("Newton residual is NaN/Inf"));


    pcout << "  Newton iteration " << n_iter << "/" << n_max_iters
          << " - ||r|| = " << std::scientific << std::setprecision(6)
          << residual_norm << std::flush;

    if (residual_norm > residual_tolerance) {
      solve_linear_system();

      const TrilinosWrappers::MPI::Vector solution_prev = solution_owned;
      const TrilinosWrappers::MPI::Vector delta_prev = delta_owned;
      const double residual_prev = residual_norm;

      const LineSearchResult ls = line_search(solution_prev, delta_prev, residual_prev);
      residual_norm = ls.residual;
      if (ls.stagnated) {
        stagnated = true;
        break;
      }
    } else {
      pcout << " < tolerance" << std::endl;
    }

    ++n_iter;
  }

  if (stagnated)
    pcout << "  Newton stopped due to stagnation." << std::endl;


  const bool converged = (!stagnated && residual_norm <= residual_tolerance);
  if (converged)
    pcout << "  Newton converged in " << n_iter << " iterations, final "
          << "residual ||r|| = " << std::scientific << std::setprecision(6)
          << residual_norm << std::endl;
  else
    pcout << "  Newton stopped in " << n_iter << " iterations, final "
          << "residual ||r|| = " << std::scientific << std::setprecision(6)
          << residual_norm << std::endl;
          
    }


void LV::solve(int num_steps){
  //load stepping
  // stability condition
  double obj_pressure = 15.0;
  double start_pressure = 1.0;
  double base_dp = (obj_pressure - start_pressure) / (double)num_steps;
  double dp = base_dp;

  pressure = start_pressure;

  for(int t = 0;t <= num_steps; ++t){
    pcout << "setting pressure to: " << pressure <<  " dp: " << dp << std::endl;

    solve_newton();
    output(t);
    pressure += dp;
  }
}

void
 LV::output(unsigned int step) const //export solution to file
 {
   pcout << "===============================================" <<
   std::endl;

   IndexSet locally_relevant_dofs;
   DoFTools::extract_locally_relevant_dofs(dof_handler,
   locally_relevant_dofs);


   TrilinosWrappers::MPI::Vector solution_ghost(locally_owned_dofs,
                                                locally_relevant_dofs,
                                                MPI_COMM_WORLD);

//   relevant DoFs are received from other processes and stored inside solution_ghost.
   solution_ghost = solution;

   DataOut<dim> data_out;
   
   std::vector<std::string> solution_names(dim, "solution");
   std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
   data_out.add_data_vector(dof_handler, solution_ghost, solution_names, interpretation);

   class SolutionMagnitudePostprocessor : public DataPostprocessorScalar<dim>
   {
   public:
     SolutionMagnitudePostprocessor()
       : DataPostprocessorScalar<dim>("solution_mag", update_values)
     {}

     void evaluate_vector_field(
       const DataPostprocessorInputs::Vector<dim> &inputs,
       std::vector<Vector<double>> &computed_quantities) const override
     {
       const unsigned int n_points = inputs.solution_values.size();

       for (unsigned int p = 0; p < n_points; ++p)
         {
           double sum_sq = 0.0;
           for (unsigned int d = 0; d < dim; ++d)
             sum_sq += inputs.solution_values[p][d] * inputs.solution_values[p][d];
           computed_quantities[p](0) = std::sqrt(sum_sq);
         }
     }
   };

   SolutionMagnitudePostprocessor magnitude_post;
   data_out.add_data_vector(dof_handler, solution_ghost, magnitude_post);

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
   GridTools::get_subdomain_association(mesh, partition_int);
   const Vector<double> partitioning(partition_int.begin(),
   partition_int.end()); data_out.add_data_vector(partitioning,
   "partitioning");

   data_out.build_patches();

   const std::filesystem::path mesh_path(mesh_file_name);
   const std::string output_file_name = "output-" +
   mesh_path.stem().string();


   data_out.write_vtu_with_pvtu_record("./",
                                       output_file_name,
                                       step,
                                       MPI_COMM_WORLD);

   pcout << "Output written to " << output_file_name << std::endl;

   pcout << "===============================================" <<
   std::endl;
 }


// Self-convergence: ||u_this - u_reference|| in the chosen norm, evaluated on
// this mesh by interpolating the reference solution via FEFieldFunction.
// Single-rank only (FEFieldFunction does not support fully distributed meshes
// with non-matching partitions).
double LV::compute_difference(const LV &reference,
                              VectorTools::NormType norm) const
{
  AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
              ExcMessage("compute_difference requires a single MPI rank "
                         "(FEFieldFunction does not support distributed meshes)."));

  MappingFE<dim> mapping(*fe);

  Functions::FEFieldFunction<dim, TrilinosWrappers::MPI::Vector> u_ref(
    reference.dof_handler, reference.solution, mapping);

  Vector<double> difference_per_cell(mesh.n_active_cells());
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    u_ref,
                                    difference_per_cell,
                                    *quadrature,
                                    norm);

  return VectorTools::compute_global_error(mesh, difference_per_cell, norm);
}

double LV::h_from_mesh_filename(const std::string &mesh_file)
{
  const std::filesystem::path p(mesh_file);
  const std::string stem = p.stem().string();

  std::string digits;
  for (auto it = stem.rbegin(); it != stem.rend(); ++it)
    {
      if (std::isdigit(static_cast<unsigned char>(*it)))
        digits.insert(digits.begin(), *it);
      else if (*it == '_' && !digits.empty())
        continue;
      else
        break;
    }

  AssertThrow(!digits.empty(),
              ExcMessage("Mesh filename has no trailing digits: " + mesh_file));

  std::size_t first_non_zero = digits.find_first_not_of('0');
  if (first_non_zero == std::string::npos)
    digits = "0";
  else
    digits = digits.substr(first_non_zero);

  return std::stod("0." + digits);
}

std::unique_ptr<LV::SerialReplica> LV::gather_to_rank0() const
{
  using CellPayload = std::pair<CellId, std::vector<double>>;
  std::vector<CellPayload> local_data;

  std::vector<types::global_dof_index> dof_indices(fs->dofs_per_cell);
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;
      cell->get_dof_indices(dof_indices);
      std::vector<double> values(fs->dofs_per_cell);
      for (unsigned int i = 0; i < fs->dofs_per_cell; ++i)
        values[i] = solution(dof_indices[i]);
      local_data.emplace_back(cell->id(), std::move(values));
    }

  const auto all_data =
    Utilities::MPI::gather(MPI_COMM_WORLD, local_data, 0);

  if (mpi_rank != 0)
    return nullptr;

  std::map<CellId, std::vector<double>> by_cell;
  for (const auto &per_rank : all_data)
    for (const auto &payload : per_rank)
      by_cell[payload.first] = payload.second;

  auto rep = std::make_unique<SerialReplica>();
  rep->fe         = std::make_unique<FE_SimplexP<dim>>(r);
  rep->fs         = std::make_unique<FESystem<dim>>(*rep->fe, dim);
  rep->mapping    = std::make_unique<MappingFE<dim>>(*rep->fe);
  rep->quadrature = std::make_unique<QGaussSimplex<dim>>(rep->fe->degree + 1);

  GridIn<dim> grid_in;
  grid_in.attach_triangulation(rep->tria);
  std::ifstream grid_in_file(mesh_file_name);
  AssertThrow(grid_in_file.good(),
              ExcMessage("gather_to_rank0: could not open mesh file '" +
                         mesh_file_name + "'"));
  grid_in.read_msh(grid_in_file);

  rep->dof_handler.reinit(rep->tria);
  rep->dof_handler.distribute_dofs(*rep->fs);
  rep->solution.reinit(rep->dof_handler.n_dofs());

  std::vector<types::global_dof_index> serial_dof_indices(rep->fs->dofs_per_cell);
  for (const auto &cell : rep->dof_handler.active_cell_iterators())
    {
      const auto it = by_cell.find(cell->id());
      AssertThrow(it != by_cell.end(),
                  ExcMessage("gather_to_rank0: missing cell " +
                             cell->id().to_string()));
      cell->get_dof_indices(serial_dof_indices);
      for (unsigned int i = 0; i < rep->fs->dofs_per_cell; ++i)
        rep->solution(serial_dof_indices[i]) = it->second[i];
    }

  return rep;
}

double LV::compute_difference_serial(const SerialReplica  &a,
                                     const SerialReplica  &b,
                                     VectorTools::NormType norm)
{
  AssertThrow(norm == VectorTools::L2_norm || norm == VectorTools::H1_norm,
              ExcMessage("compute_difference_serial supports only L2 and H1 norms."));

  Functions::FEFieldFunction<dim, Vector<double>> u_b(b.dof_handler,
                                                      b.solution,
                                                      *b.mapping);

  const FEValuesExtractors::Vector vec_index(0);
  const unsigned int n_q = a.quadrature->size();

  UpdateFlags flags = update_values | update_quadrature_points | update_JxW_values;
  if (norm == VectorTools::H1_norm)
    flags |= update_gradients;

  FEValues<dim> fe_values_a(*a.mapping, *a.fs, *a.quadrature, flags);

  std::vector<Tensor<1, dim>> u_a_values(n_q);
  std::vector<Tensor<2, dim>> u_a_grads(n_q);

  double sum_l2_sq = 0.0;
  double sum_h1_seminorm_sq = 0.0;
  unsigned int total_qp = 0, skipped_qp = 0;

  for (const auto &cell : a.dof_handler.active_cell_iterators())
    {
      fe_values_a.reinit(cell);
      fe_values_a[vec_index].get_function_values(a.solution, u_a_values);
      if (norm == VectorTools::H1_norm)
        fe_values_a[vec_index].get_function_gradients(a.solution, u_a_grads);

      for (unsigned int q = 0; q < n_q; ++q)
        {
          ++total_qp;
          const Point<dim> &p = fe_values_a.quadrature_point(q);

          Vector<double> u_b_val(dim);
          std::vector<Tensor<1, dim>> u_b_grad_vec(dim);
          bool found = true;

          try
            {
              u_b.vector_value(p, u_b_val);
              if (norm == VectorTools::H1_norm)
                u_b.vector_gradient(p, u_b_grad_vec);
            }
          catch (const dealii::ExceptionBase &)
            {
              found = false;
            }

          if (!found)
            {
              ++skipped_qp;
              continue;
            }

          double diff_sq = 0.0;
          for (unsigned int d = 0; d < dim; ++d)
            {
              const double diff_d = u_a_values[q][d] - u_b_val[d];
              diff_sq += diff_d * diff_d;
            }
          sum_l2_sq += diff_sq * fe_values_a.JxW(q);

          if (norm == VectorTools::H1_norm)
            {
              double grad_diff_sq = 0.0;
              for (unsigned int d = 0; d < dim; ++d)
                for (unsigned int e = 0; e < dim; ++e)
                  {
                    const double g_de = u_a_grads[q][d][e] - u_b_grad_vec[d][e];
                    grad_diff_sq += g_de * g_de;
                  }
              sum_h1_seminorm_sq += grad_diff_sq * fe_values_a.JxW(q);
            }
        }
    }

  if (skipped_qp > 0)
    std::cout << "  [compute_difference_serial] skipped "
              << skipped_qp << " of " << total_qp
              << " quadrature points (boundary mismatch on curved geometry)"
              << std::endl;

  if (norm == VectorTools::L2_norm)
    return std::sqrt(sum_l2_sq);
  else
    return std::sqrt(sum_l2_sq + sum_h1_seminorm_sq);
}

void LV::run_convergence_study(const std::vector<std::string> &mesh_files,
                               unsigned int r,
                               const std::string &csv_filename)
{
  AssertThrow(mesh_files.size() >= 2,
              ExcMessage(
                "Need at least 2 meshes to compute self-convergence."));

  // Each solve runs in full MPI parallel; only the post-solve comparison is
  // performed serially on rank 0 (using gathered SerialReplica objects).

  const unsigned int rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  ConditionalOStream pcout(std::cout, rank == 0);

  std::ofstream convergence_file;
  if (rank == 0)
    {
      convergence_file.open(csv_filename);
      AssertThrow(convergence_file.good(),
                  ExcMessage("Could not open '" + csv_filename + "'"));
      convergence_file << "h,diffL2,rateL2,diffH1,rateH1" << std::endl;
    }

  std::vector<std::unique_ptr<SerialReplica>> replicas;
  if (rank == 0)
    replicas.reserve(mesh_files.size());

  for (const auto &mesh_file : mesh_files)
    {
      pcout << "Convergence study: solving on " << mesh_file
            << " \n" << std::endl;
      auto model = std::make_unique<LV>(mesh_file, r);
      model->setup();
      model->solve(25);
      auto rep = model->gather_to_rank0();
      if (rank == 0)
        replicas.push_back(std::move(rep));
    }

  if (rank != 0)
    return;

  std::vector<double> hs, l2s, h1s;
  for (std::size_t i = 0; i + 1 < replicas.size(); ++i)
    {
      hs.push_back(LV::h_from_mesh_filename(mesh_files[i]));
      l2s.push_back(compute_difference_serial(*replicas[i],
                                              *replicas[i + 1],
                                              VectorTools::L2_norm));
      h1s.push_back(compute_difference_serial(*replicas[i],
                                              *replicas[i + 1],
                                              VectorTools::H1_norm));
    }


  auto rate = [](double err_prev, double err_curr,
                 double h_prev,   double h_curr) -> double
  {
    return std::log(err_prev / err_curr) / std::log(h_prev / h_curr);
  };

  std::cout << "\nConve table \n" << std::endl;
  std::cout << std::left << std::setw(8) << "h"
            << std::setw(14) << "L2"
            << std::setw(10) << "rate_L2"
            << std::setw(14) << "H1"
            << std::setw(10) << "rate_H1"
            << std::endl;

  for (std::size_t i = 0; i < hs.size(); ++i)
    {
      double rL2 = std::numeric_limits<double>::quiet_NaN();
      double rH1 = std::numeric_limits<double>::quiet_NaN();
      if (i > 0)
        {
          rL2 = rate(l2s[i - 1], l2s[i], hs[i - 1], hs[i]);
          rH1 = rate(h1s[i - 1], h1s[i], hs[i - 1], hs[i]);
        }

      std::cout << std::left
                << std::fixed << std::setprecision(4) << std::setw(8) << hs[i]
                << std::scientific << std::setprecision(4) << std::setw(14) << l2s[i];
      if (i == 0) std::cout << std::setw(10) << "-";
      else        std::cout << std::fixed << std::setprecision(2) << std::setw(10) << rL2;
      std::cout << std::scientific << std::setprecision(4) << std::setw(14) << h1s[i];
      if (i == 0) std::cout << std::setw(10) << "-";
      else        std::cout << std::fixed << std::setprecision(2) << std::setw(10) << rH1;
      std::cout << std::endl;

      convergence_file << hs[i] << "," << l2s[i] << ","
                       << (i == 0 ? std::string("") : std::to_string(rL2)) << ","
                       << h1s[i] << ","
                       << (i == 0 ? std::string("") : std::to_string(rH1))
                       << std::endl;
    }
}

} // namespace guccione
