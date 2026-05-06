#include "tensor_utils.hpp"
#include "../constants.hpp"

namespace guccione {

// TensorUtils: hyperelastic material via automatic differentiation (AD)
//given deformation gradient F, compute
//    P  = dW/dF   (first Piola-Kirchhoff stress tensor)
//    dP/dF        (consistent material tangent for Newton)
//
// The code defines a scalar strain-energy density W(F) and then utilises deal.II AD to obtain:
//   gradient(W) -> P,  hessian(W) -> dP/dF.


TensorUtils::TensorUtils() : ad_helper(dim * dim) {
      P_flat = Vector<double>(dim * dim);
      C_flat = FullMatrix<double>(dim * dim, dim * dim);
    }

void TensorUtils::compute_tensors(Tensor<2, dim> F, const Point<dim>& p, Tensor<2, dim> &P, Tensor<4, dim> &C) {


  namespace ccg = cardiac::constants::guccione;
  namespace cc  = cardiac::constants;
  const double d = ccg::PROLATE_D;

  double x = p[0], y = p[1], z = p[2];
  
  double r1 = std::sqrt(x*x + y*y + (z-d)*(z-d)); //distance from upper focal point (d) in z
  double r2 = std::sqrt(x*x + y*y + (z+d)*(z+d)); //distance from lower focal point (-d) in z
  double cosh_xi = (r1 + r2) / (2.0 * d); 
  double xi = std::acosh(std::max(1.0, cosh_xi));

  
  double theta = std::acos(std::max(-1.0, std::min(1.0, (r2 - r1) / (2.0 * d))));
  double phi = std::atan2(y, x);

  double cosphi = std::cos(phi);
  double sinphi = std::sin(phi);

  const double r = std::sqrt(cosphi*cosphi + sinphi*sinphi);
  if (r > 1e-12) {
    cosphi /= r;
    sinphi /= r;
  } else {
    cosphi = 1.0;
    sinphi = 0.0;
  }


  
  Tensor<1, 3> g_xi, g_theta, g_phi; //tangent basis vectors in prolate spheroidal coordinates
  
  g_xi[0] = d * std::cosh(xi) * std::sin(theta) * cosphi;
  g_xi[1] = d * std::cosh(xi) * std::sin(theta) * sinphi;
  g_xi[2] = d * std::sinh(xi) * std::cos(theta);

  g_theta[0] = d * std::sinh(xi) * std::cos(theta) * cosphi;
  g_theta[1] = d * std::sinh(xi) * std::cos(theta) * sinphi;
  g_theta[2] = -d * std::cosh(xi) * std::sin(theta);

  g_phi[0] = -d * std::sinh(xi) * std::sin(theta) * sinphi;
  g_phi[1] =  d * std::sinh(xi) * std::sin(theta) * cosphi;
  g_phi[2] = 0.0;

  // Normalization
  Tensor<1, 3> e_xi    = g_xi / g_xi.norm();      // radial (transmural)
  Tensor<1, 3> e_theta = g_theta / g_theta.norm(); // longitudinal
  Tensor<1, 3> e_phi   = g_phi / g_phi.norm();     // circumferential

 
  const double alpha_endo = cc::FIBER_ALPHA_ENDO * M_PI / 180.0;
  const double alpha_epi  = cc::FIBER_ALPHA_EPI  * M_PI / 180.0;
  const double xi_endo    = ccg::PROLATE_XI_ENDO;
  const double xi_epi     = ccg::PROLATE_XI_EPI;
  double wall_fraction = (xi - xi_endo) / (xi_epi - xi_endo); 
  
  // Ensure wall between 0 and 1 to avoid extrapolation outside the wall
  wall_fraction = std::max(0.0, std::min(1.0, wall_fraction));

  const double alpha = alpha_endo + (alpha_epi - alpha_endo) * wall_fraction;

  Tensor<1, dim> n_vec = e_xi; 
  Tensor<1, dim> f_vec = std::cos(alpha) * e_phi + std::sin(alpha) * e_theta; 
  f_vec /= f_vec.norm();

  Tensor<1, dim> s_vec = cross_product_3d(n_vec, f_vec);
  s_vec /= s_vec.norm();

  ad_helper.reset();
  
      const FEValuesExtractors::Tensor<2> F_linearized(0);
      const FEValuesExtractors::Tensor<4> C_linearized(0);
      const typename AutoDiff::Types<ADNumberType>::tape_index tape_index = 0;
     
      // AD tape management:
        // On the first call, we "record" the computational graph of W(F).
  // On later calls we reuse the tape and only update the values of F.
      const bool is_recording =
          ad_helper.start_recording_operations(tape_index);
      if (is_recording) {
        // Register F as independent variables (inputs), compute W(F), and register W as dependent (output).
            // After recording, AD can produce derivatives of W with respect to entries of F.
        ad_helper.register_independent_variable(F, F_linearized);

        Tensor<2,dim,ADNumberType> F_AD = ad_helper.get_sensitive_variables(F_linearized);
        
        //ADNumberType W_ad = compute_W(F_AD);
        ADNumberType W_ad = compute_W(F_AD, f_vec, s_vec, n_vec);


        ad_helper.register_dependent_variable(W_ad);
        ad_helper.stop_recording_operations(false);
      } else {
        
        ad_helper.activate_recorded_tape(tape_index);
        ad_helper.set_independent_variable(F, F_linearized);
      }
      ad_helper.compute_gradient(P_flat);
      ad_helper.compute_hessian(C_flat);

      

      P = ad_helper.extract_gradient_component(P_flat, F_linearized);
      C = ad_helper.extract_hessian_component(C_flat, F_linearized,
                                           F_linearized);
    }

/*
typename TensorUtils::ADNumberType TensorUtils::compute_W(const Tensor<2, dim, ADNumberType> &F) const {
      const ADNumberType J_ad = determinant(F);
      const ADTensor2 C_ad = transpose(F) * F;
      const ADNumberType I1_ad = trace(C_ad);

      
      ADNumberType psi_ad =
          (mu_hook / 2.0) * (I1_ad * std::pow(J_ad, -2.0 / 3.0) - 3.0);
      psi_ad += (k_hook / 2.0) * std::pow(J_ad - 1.0, 2.0)
;

// //      using the isothropic formulation from the paper, pressure expressed in kPa
//       ADNumberType psi_ad = (a_iso / (2.0 * b_iso)) * (std::exp(b_iso * (I1_ad - 3.0)) - 1.0);

//       //adding this term as surrogate for orthothropic terms
//       psi_ad += (k_hook / 2.0) * std::pow(J_ad - 1.0, 2.0);
          
      return psi_ad;
    }
*/

typename TensorUtils::ADNumberType
TensorUtils::compute_W(const Tensor<2, dim, ADNumberType> &F,
                       const Tensor<1, dim> &f,
                       const Tensor<1, dim> &s,
                       const Tensor<1, dim> &n) const
{
  using ADTensor2 = Tensor<2, dim, ADNumberType>;

  // --- Kinematics ---
  const ADNumberType J_ad = determinant(F);
  const ADTensor2 C_ad = transpose(F) * F;

  // Green-Lagrange strain: E = 0.5 * (C - I)
  ADTensor2 I;
  for (unsigned int i = 0; i < dim; ++i) I[i][i] = 1.0;
  const ADTensor2 E_ad = 0.5 * (C_ad - I);

  // --- Project strain onto fiber coordinate system ---
  const ADNumberType E_ff = scalar_product(f, E_ad * f);
  const ADNumberType E_ss = scalar_product(s, E_ad * s);
  const ADNumberType E_nn = scalar_product(n, E_ad * n);

  const ADNumberType E_fs = scalar_product(f, E_ad * s);
  const ADNumberType E_fn = scalar_product(f, E_ad * n);
  const ADNumberType E_sn = scalar_product(s, E_ad * n);

  // --- Guccione exponential argument Q ---
  const ADNumberType Q =
      b_ff * E_ff * E_ff +
      b_ss * E_ss * E_ss +
      b_nn * E_nn * E_nn +
      2.0 * b_fs * E_fs * E_fs +
      2.0 * b_fn * E_fn * E_fn +
      2.0 * b_sn * E_sn * E_sn;

  // --- Strain energy (Guccione) ---
  ADNumberType psi_ad = (C_param / 2.0) * (exp(Q) - 1.0);

  // --- Volumetric penalty (nearly incompressible) ---
  psi_ad += (k_hook / 2.0) * pow(J_ad - 1.0, 2.0);

  return psi_ad;
}

TensorUtils::TensorUtils(const Params &p) : TensorUtils() {
  mu_hook = p.mu_hook; k_hook = p.k_hook;
  b_ff = p.b_ff; b_ss = p.b_ss; b_nn = p.b_nn;
  b_fs = p.b_fs; b_fn = p.b_fn; b_sn = p.b_sn;
  C_param = p.C_param;
}

void TensorUtils::compute_tensors(Tensor<2, dim>              F,
                                   Tensor<2, dim>              &P,
                                   Tensor<4, dim>              &C,
                                   const cardiac::MaterialInput &in) {
  compute_tensors(F, in.p, P, C);
}

} // namespace guccione


