#include "tensor_utils.hpp"

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

void TensorUtils::compute_tensors(Tensor<2, dim> F, Tensor<2, dim> &P,
                         Tensor<4, dim> &C,
                         const Tensor<1, dim> &f0,
                         const Tensor<1, dim> &s0,
                         const Tensor<1, dim> &n0) {

  ad_helper.reset();
  
      const FEValuesExtractors::Tensor<2> F_linearized(0);
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
        
        ADNumberType W_ad = compute_W(F_AD, f0, s0, n0);
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

typename TensorUtils::ADNumberType TensorUtils::compute_W(const Tensor<2, dim, ADNumberType> &F,
                         const Tensor<1, dim> &f0,
                         const Tensor<1, dim> &s0,
                         const Tensor<1, dim> &n0) const {
      const ADNumberType J_ad = determinant(F);
      const ADTensor2 C_ad = transpose(F) * F;
      const ADNumberType I1_ad = trace(C_ad);

      ADNumberType I4f_ad = 0.0;
      ADNumberType I4s_ad = 0.0;
      ADNumberType I8fs_ad = 0.0;

      for (unsigned int i = 0; i < dim; ++i) {
          for (unsigned int j = 0; j < dim; ++j) {
              I4f_ad += f0[i] * C_ad[i][j] * f0[j];
              I4s_ad += s0[i] * C_ad[i][j] * s0[j];
              I8fs_ad += f0[i] * C_ad[i][j] * s0[j];
          }
      }

      ADNumberType psi_ad = (a / (2.0 * b)) * (std::exp(b * (I1_ad - 3.0)) - 1.0);

      if (I4f_ad > 1.0) {
          psi_ad += (a_f / (2.0 * b_f)) * (std::exp(b_f * std::pow(I4f_ad - 1.0, 2.0)) - 1.0);
      }
      if (I4s_ad > 1.0) {
          psi_ad += (a_s / (2.0 * b_s)) * (std::exp(b_s * std::pow(I4s_ad - 1.0, 2.0)) - 1.0);
      }
      
      psi_ad += (a_fs / (2.0 * b_fs)) * (std::exp(b_fs * std::pow(I8fs_ad, 2.0)) - 1.0);

      psi_ad += (kappa / 2.0) * std::pow(J_ad - 1.0, 2.0);

// //      using the isothropic formulation from the paper, pressure expressed in kPa
//       ADNumberType psi_ad = (a_iso / (2.0 * b_iso)) * (std::exp(b_iso * (I1_ad - 3.0)) - 1.0);

//       //adding this term as surrogate for orthothropic terms
//       psi_ad += (k_hook / 2.0) * std::pow(J_ad - 1.0, 2.0);
          
      return psi_ad;
    }
