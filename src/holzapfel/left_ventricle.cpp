#include "left_ventricle.hpp"
#include "../constants.hpp"
#include "tensor_utils.hpp"
#include "Poisson.hpp"

#include <cmath>
#include <Sacado.hpp>
#include <limits>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <deal.II/base/mpi.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/fe_field_function.h>

namespace holzapfel {

void LV::init_poisson() {
  dof_handler_poisson.reinit(mesh);
  dof_handler_poisson.distribute_dofs(*fe);

  Poisson poisson(mesh, lambda, fe, quadrature, quadrature_face, pcout);
  poisson.setup();
  poisson.assemble();
  poisson.solve();
  poisson.output();

  IndexSet locally_owned_poisson = dof_handler_poisson.locally_owned_dofs();
  IndexSet locally_relevant_poisson;
  DoFTools::extract_locally_relevant_dofs(dof_handler_poisson, locally_relevant_poisson);

  lambda_ghost.reinit(locally_owned_poisson, locally_relevant_poisson, MPI_COMM_WORLD);
  lambda_ghost = lambda;
}

void LV::compute_rhs() {
  const unsigned int dofs_per_cell = fs->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();
  const unsigned int n_q_face      = quadrature_face->size();

  FEValues<dim> fe_values(*fs, *quadrature,
                           update_values | update_gradients |
                           update_quadrature_points | update_JxW_values);

  FEValues<dim> fe_values_poisson(*fe, *quadrature,
                                   update_values | update_gradients);

  const FEValuesExtractors::Vector vec_index(0);

  FEFaceValues<dim> fe_face_values(*fs, *quadrature_face,
                                    update_values | update_normal_vectors |
                                    update_JxW_values | update_gradients |
                                    update_quadrature_points);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  jacobian_matrix = 0.0;
  system_rhs      = 0.0;

  std::vector<double>         lambda_values(n_q);
  std::vector<Tensor<1, dim>> lambda_gradients(n_q);
  std::vector<Tensor<1, dim>> solution_values_face(n_q_face);
  std::vector<Tensor<2, dim>> solution_gradient_loc_face(n_q_face);
  std::vector<Tensor<2, dim>> solution_gradient_loc(n_q);

  TensorUtils t_utils;

  auto cell_poisson = dof_handler_poisson.begin_active();

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned()) {
      ++cell_poisson;
      continue;
    }

    fe_values.reinit(cell);
    fe_values_poisson.reinit(cell_poisson);
    fe_values_poisson.get_function_values(lambda_ghost, lambda_values);
    fe_values_poisson.get_function_gradients(lambda_ghost, lambda_gradients);

    cell_rhs    = 0.0;
    cell_matrix = 0.0;
    cell->get_dof_indices(dof_indices);

    fe_values[vec_index].get_function_gradients(solution, solution_gradient_loc);

    Tensor<1, expected_dofs_per_cell, VecADNumberType> local_solution;
    for (unsigned int i = 0; i < expected_dofs_per_cell; ++i)
      local_solution[i] = solution(dof_indices[i]);

    auto local_rhs_assembler =
      [&](Tensor<1, expected_dofs_per_cell, VecADNumberType> &local_u) {

        for (unsigned int q = 0; q < n_q; ++q) {
          const double lam = lambda_values[q];

          Tensor<1, dim> e2 = lambda_gradients[q];
          e2 /= e2.norm();

          // Pick the coordinate axis least aligned with e2 to avoid
          // Gram-Schmidt singularity at the apex.
          Tensor<1, dim> ref;
          {
            unsigned int min_idx = 0;
            for (unsigned int d = 1; d < dim; ++d)
              if (std::abs(e2[d]) < std::abs(e2[min_idx]))
                min_idx = d;
            ref[min_idx] = 1.0;
          }
          Tensor<1, dim> e1 = ref - (ref * e2) * e2;
          e1 /= e1.norm();

          Tensor<1, dim> e0 = cross_product_3d(e1, e2);

          namespace cc  = cardiac::constants;
          namespace cch = cardiac::constants::holzapfel;
          const double alpha_endo = cc::FIBER_ALPHA_ENDO, alpha_epi = cc::FIBER_ALPHA_EPI;
          const double beta_endo  = cch::FIBER_BETA_ENDO, beta_epi  = cch::FIBER_BETA_EPI;

          const double alpha = (alpha_endo * (1.0 - lam) + alpha_epi * lam) * M_PI / 180.0;
          const double beta  = (beta_endo  * (1.0 - lam) + beta_epi  * lam) * M_PI / 180.0;

          Tensor<1, dim> f0    = std::cos(alpha) * e0 + std::sin(alpha) * e1;
          Tensor<1, dim> g_hat = -std::sin(alpha) * e0 + std::cos(alpha) * e1;
          Tensor<1, dim> s0   = std::cos(beta) * g_hat + std::sin(beta) * e2;
          Tensor<1, dim> n0   = cross_product_3d(f0, s0);

          Tensor<2, dim, VecADNumberType> F;
          F.clear();
          for (unsigned int i = 0; i < dim; ++i)
            F[i][i] = VecADNumberType(1.0);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            F += local_u[i] * fe_values[vec_index].gradient(i, q);

          Tensor<2, dim> F_d;
          for (unsigned int ii = 0; ii < dim; ++ii)
            for (unsigned int jj = 0; jj < dim; ++jj)
              F_d[ii][jj] = Sacado::ScalarValue<VecADNumberType>::eval(F[ii][jj]);

          const double det_F = determinant(F_d);
          AssertThrow(std::isfinite(det_F) && det_F > 1e-12,
                      ExcMessage("Non-positive or near-singular det(F) in volume quadrature"));

          Tensor<2, dim> P;
          Tensor<4, dim> dP_dF;
          t_utils.compute_tensors(F_d, P, dP_dF, f0, s0, n0);

          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            const Tensor<2, dim> grad_phi_i = fe_values[vec_index].gradient(i, q);
            cell_rhs(i) += scalar_product(P, grad_phi_i) * fe_values.JxW(q);

            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
              const Tensor<2, dim> grad_phi_j = fe_values[vec_index].gradient(j, q);
              Tensor<2, dim> dP;
              dP.clear();
              for (unsigned int a = 0; a < dim; ++a)
                for (unsigned int b = 0; b < dim; ++b)
                  for (unsigned int c = 0; c < dim; ++c)
                    for (unsigned int d = 0; d < dim; ++d)
                      dP[a][b] += dP_dF[a][b][c][d] * grad_phi_i[c][d];

              cell_matrix(i, j) += scalar_product(dP, grad_phi_j) * fe_values.JxW(q);
            }
          }
        }

        const double alpha_robin = cardiac::constants::holzapfel::ALPHA_ROBIN;

        if (cell->at_boundary()) {
          for (unsigned int f = 0; f < cell->n_faces(); ++f) {
            if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == cardiac::constants::BID_PRESSURE) {
              fe_face_values.reinit(cell, f);
              fe_face_values[vec_index].get_function_gradients(solution,
                                                               solution_gradient_loc_face);

              for (unsigned int q = 0; q < n_q_face; ++q) {
                Tensor<2, dim> Fh;
                Fh.clear();
                for (unsigned int d = 0; d < dim; ++d)
                  Fh[d][d] = 1.0;
                Fh += solution_gradient_loc_face[q];

                const double det_Fh = determinant(Fh);
                AssertThrow(std::isfinite(det_Fh) && det_Fh > 0.0,
                            ExcMessage("Non-positive or NaN det(Fh) on boundary face"));

                const Tensor<2, dim> aux_F = transpose(invert(Fh));
                const Tensor<2, dim> H     = det_Fh * aux_F;

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                  const Tensor<1, dim> term1 = H * fe_face_values.normal_vector(q);
                  const Tensor<1, dim> phi_i = fe_face_values[vec_index].value(i, q);
                  cell_rhs(i) += pressure * scalar_product(term1, phi_i) * fe_face_values.JxW(q);

                  for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    const Tensor<2, dim> grad_phi_j = fe_face_values[vec_index].gradient(j, q);
                    const double         prod        = scalar_product(grad_phi_j, aux_F);

                    Tensor<2, dim> term2;
                    for (unsigned int d = 0; d < dim; ++d)
                      term2[d][d] = prod;

                    const Tensor<2, dim> term3 = aux_F * transpose(grad_phi_j);
                    const Tensor<1, dim> term4 = (term2 - term3) * (H * fe_face_values.normal_vector(q));

                    cell_matrix(i, j) += pressure * scalar_product(term4, phi_i) * fe_face_values.JxW(q);
                  }
                }
              }
            }

            if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == cardiac::constants::BID_ROBIN) {
              fe_face_values.reinit(cell, f);
              fe_face_values[vec_index].get_function_values(solution, solution_values_face);

              for (unsigned int q = 0; q < n_q_face; ++q) {
                const Tensor<1, dim> u_q = solution_values_face[q];
                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                  const Tensor<1, dim> phi_i = fe_face_values[vec_index].value(i, q);
                  cell_rhs(i) += alpha_robin * scalar_product(u_q, phi_i) * fe_face_values.JxW(q);

                  for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    const Tensor<1, dim> phi_j = fe_face_values[vec_index].value(j, q);
                    cell_matrix(i, j) += alpha_robin * scalar_product(phi_i, phi_j) * fe_face_values.JxW(q);
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
    system_rhs.add(dof_indices, cell_rhs);
    ++cell_poisson;
  }
}

void LV::solve() {
  init_poisson();
  solve_loop(params.n_steps);
}

} // namespace holzapfel
