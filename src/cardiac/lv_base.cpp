#include "lv_base.hpp"
#include "../constants.hpp"

#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <deal.II/base/mpi.h>
#include <deal.II/numerics/data_out.h>

LVBase::LVBase(const std::string &mesh_file_name_,
               const unsigned int &r_,
               const SolverParams &params_)
    : mesh_file_name(mesh_file_name_)
    , r(r_)
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , params(params_)
    , mesh(MPI_COMM_WORLD)
    , pcout(std::cout, mpi_rank == 0)
{}

void LVBase::setup() {
  last_iteration = 0;
  pcout << "===============================================" << std::endl;

  {
    pcout << "Initializing the mesh" << std::endl;

    Triangulation<dim> mesh_serial;
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(mesh_serial);

    std::ifstream grid_in_file(mesh_file_name);
    AssertThrow(grid_in_file.good(),
                ExcMessage("Could not open mesh file: '" + mesh_file_name +
                           "'. Current working directory is '" +
                           std::filesystem::current_path().string()));
    grid_in.read_msh(grid_in_file);

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::
        create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    pcout << "  Number of elements = " << mesh.n_global_active_cells() << std::endl;

    {
      std::map<types::boundary_id, unsigned long long> boundary_face_counts;
      for (const auto &cell : mesh.active_cell_iterators()) {
        if (!cell->is_locally_owned() || !cell->at_boundary())
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

  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_SimplexP<dim>>(r);
    fs = std::make_unique<FESystem<dim>>(*fe, dim);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fs->dofs_per_cell << std::endl;

    check((fs->dofs_per_cell == expected_dofs_per_cell),
          "this needs to be fixed some how");

    quadrature      = std::make_unique<QGaussSimplex<dim>>(fe->degree + 1);
    quadrature_face = std::make_unique<QGaussSimplex<dim - 1>>(fe->degree + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size() << std::endl;
    pcout << "  Quadrature points per face = " << quadrature_face->size() << std::endl;
  }
  pcout << "-----------------------------------------------" << std::endl;

  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fs);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }
  pcout << "-----------------------------------------------" << std::endl;

  {
    pcout << "Initializing the linear system" << std::endl;
    pcout << "  Initializing the sparsity pattern" << std::endl;

    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    pcout << "  Initializing the Jacobian matrix" << std::endl;
    jacobian_matrix.reinit(sparsity);

    pcout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    pcout << "  Initializing the solution vector" << std::endl;
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    delta_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  }
}

void LVBase::assemble_system() {
  compute_rhs();

  jacobian_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  std::map<types::global_dof_index, double> boundary_values;
  Functions::ZeroFunction<dim> zero_function(dim);
  VectorTools::interpolate_boundary_values(dof_handler, cardiac::constants::BID_DIRICHLET, zero_function, boundary_values);
  MatrixTools::apply_boundary_values(boundary_values, jacobian_matrix, delta_owned, system_rhs, false);
}

void LVBase::solve_linear_system() {
  delta_owned = 0.0;
  delta_owned.compress(VectorOperation::insert);

  SolverControl solver_control(params.gmres_max_iter,
                                params.gmres_tol_factor * system_rhs.l2_norm());

  SolverGMRES<TrilinosWrappers::MPI::Vector>::AdditionalData gmres_data;
  gmres_data.max_n_tmp_vectors = params.gmres_restart;
  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control, gmres_data);

  TrilinosWrappers::PreconditionAMG preconditioner;
  TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
  amg_data.elliptic              = params.amg_elliptic;
  amg_data.aggregation_threshold = params.amg_aggregation_threshold;

  if (params.amg_extract_constant_modes) {
    const FEValuesExtractors::Vector displacements(0);
    DoFTools::extract_constant_modes(dof_handler,
                                      fs->component_mask(displacements),
                                      amg_data.constant_modes);
  }

  preconditioner.initialize(jacobian_matrix, amg_data);
  solver.solve(jacobian_matrix, delta_owned, system_rhs, preconditioner);
  last_iteration = solver_control.last_step();
  pcout << "  " << solver_control.last_step() << " GMRES iterations "
        << "(final residual " << std::scientific << std::setprecision(3)
        << solver_control.last_value() << ")" << std::endl;
}

LVBase::LineSearchResult
LVBase::line_search(const TrilinosWrappers::MPI::Vector &solution_prev,
                    const TrilinosWrappers::MPI::Vector &delta_prev,
                    double                               residual_prev) {
  namespace cc = cardiac::constants;
  double             alpha_ls      = 1.0;
  const unsigned int max_backtracks = cc::LS_MAX_BACKTRACKS;

  bool   last_trial_assembled = false;
  double last_trial_residual  = std::numeric_limits<double>::infinity();

  TrilinosWrappers::MPI::Vector last_good_trial = solution_prev;
  double alpha_last_good = 0.0;

  for (unsigned int ls_iter = 0; ls_iter < max_backtracks; ++ls_iter) {
    TrilinosWrappers::MPI::Vector trial = solution_prev;
    trial.add(-alpha_ls, delta_prev);
    solution_owned = trial;
    solution       = trial;

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
      solution       = solution_prev;
      alpha_ls *= 0.5;
      continue;
    }

    last_trial_assembled = true;
    const double residual_trial = system_rhs.l2_norm();
    last_trial_residual = residual_trial;
    last_good_trial     = trial;
    alpha_last_good     = alpha_ls;

    if (std::isfinite(residual_trial) &&
        (residual_trial < residual_prev * (1.0 - cc::LS_REL_DECREASE_EPS) ||
         residual_trial < residual_prev - cc::LS_ABS_DECREASE_EPS))
      return {true, false, alpha_ls, residual_trial};

    alpha_ls *= 0.5;
  }

  if (last_trial_assembled && std::isfinite(last_trial_residual)) {
    const double step_norm = alpha_last_good * delta_prev.l2_norm();
    const double sol_norm  = solution_prev.l2_norm();
    if (step_norm <= cc::LS_STEP_TOL_FACTOR * (sol_norm + 1.0) || alpha_last_good <= cc::LS_ALPHA_MIN) {
      solution_owned = solution_prev;
      solution       = solution_prev;
      pcout << "  (line search stagnated: alpha=" << std::scientific
            << std::setprecision(2) << alpha_last_good
            << ", step_norm=" << step_norm << ")" << std::endl;
      return {false, true, alpha_last_good, residual_prev};
    }
    solution_owned = last_good_trial;
    solution       = last_good_trial;
    pcout << "  (alpha=" << std::scientific << std::setprecision(2)
          << alpha_last_good << ", no residual decrease)" << std::endl;
    return {false, false, alpha_last_good, last_trial_residual};
  }

  AssertThrow(false, ExcMessage("Line search failed: could not keep det(F)>0."));
  return {false, true, 0.0, residual_prev};
}

void LVBase::solve_newton() {
  namespace cc = cardiac::constants;
  const unsigned int n_max_iters       = cc::NEWTON_MAX_ITER;
  const double       residual_tolerance = cc::NEWTON_TOLERANCE;

  unsigned int n_iter       = 0;
  double       residual_norm = residual_tolerance + 1;

  {
    std::set<types::boundary_id> dirichlet_boundary_id = {cc::BID_DIRICHLET};
    ComponentMask dirichlet_mask = fs->component_mask(FEValuesExtractors::Vector(0));

    IndexSet dirichlet_dofs =
        DoFTools::extract_boundary_dofs(dof_handler, dirichlet_mask, dirichlet_boundary_id);
    dirichlet_dofs = dirichlet_dofs & dof_handler.locally_owned_dofs();

    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler, cc::BID_DIRICHLET, function_g, boundary_values);

    for (const auto &idx : dirichlet_dofs)
      if (const auto it = boundary_values.find(idx); it != boundary_values.end())
        solution_owned[idx] = it->second;

    solution_owned.compress(VectorOperation::insert);
    solution = solution_owned;
  }

  bool stagnated = false;

  while (n_iter < n_max_iters && residual_norm > residual_tolerance) {
    assemble_system();
    residual_norm = system_rhs.l2_norm();

    AssertThrow(std::isfinite(residual_norm), ExcMessage("Newton residual is NaN/Inf"));

    pcout << "  Newton iteration " << n_iter << "/" << n_max_iters
          << " - ||r|| = " << std::scientific << std::setprecision(6)
          << residual_norm << std::flush;

    if (residual_norm > residual_tolerance) {
      solve_linear_system();

      const TrilinosWrappers::MPI::Vector solution_prev  = solution_owned;
      const TrilinosWrappers::MPI::Vector delta_prev     = delta_owned;
      const double                        residual_prev  = residual_norm;

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

void LVBase::solve_loop(int num_steps) {
  const double dp = (params.p_end - params.p_start) / num_steps;
  pressure = params.p_start;
  for (int t = 0; t <= num_steps; ++t) {
    pcout << "setting pressure to: " << pressure << " dp: " << dp << std::endl;
    solve_newton();
    output(t);
    pressure += dp;
  }
}

void LVBase::output(unsigned int step) const {
  pcout << "===============================================" << std::endl;

  IndexSet locally_relevant_dofs_out;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs_out);

  TrilinosWrappers::MPI::Vector solution_ghost(locally_owned_dofs,
                                               locally_relevant_dofs_out,
                                               MPI_COMM_WORLD);
  solution_ghost = solution;

  DataOut<dim> data_out;

  std::vector<std::string> solution_names(dim, "solution");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
  data_out.add_data_vector(dof_handler, solution_ghost, solution_names, interpretation);

  class SolutionMagnitudePostprocessor : public DataPostprocessorScalar<dim> {
  public:
    SolutionMagnitudePostprocessor()
        : DataPostprocessorScalar<dim>("solution_mag", update_values) {}

    void evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim> &inputs,
        std::vector<Vector<double>>                &computed_quantities) const override {
      for (unsigned int p = 0; p < inputs.solution_values.size(); ++p) {
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
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  const std::filesystem::path mesh_path(mesh_file_name);
  const std::string output_file_name = "output-" + mesh_path.stem().string();

  data_out.write_vtu_with_pvtu_record("./", output_file_name, step, MPI_COMM_WORLD);
  pcout << "Output written to " << output_file_name << std::endl;
  pcout << "===============================================" << std::endl;
}
