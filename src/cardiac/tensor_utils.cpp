#include "tensor_utils.hpp"

namespace cardiac {

TensorUtils::TensorUtils(const StrainEnergy &energy)
    : energy_(energy)
    , ad_helper(dim * dim)
{
  P_flat = Vector<double>(dim * dim);
  C_flat = FullMatrix<double>(dim * dim, dim * dim);
}

void TensorUtils::compute_tensors(Tensor<2, dim>       F,
                                   Tensor<2, dim>       &P,
                                   Tensor<4, dim>       &C,
                                   const Tensor<1, dim> &f0,
                                   const Tensor<1, dim> &s0,
                                   const Tensor<1, dim> &n0)
{
  ad_helper.reset();

  const FEValuesExtractors::Tensor<2> F_linearized(0);
  const typename AutoDiff::Types<ADNumberType>::tape_index tape_index = 0;

  const bool is_recording = ad_helper.start_recording_operations(tape_index);
  if (is_recording) {
    ad_helper.register_independent_variable(F, F_linearized);

    Tensor<2, dim, ADNumberType> F_AD = ad_helper.get_sensitive_variables(F_linearized);

    ADNumberType W_ad = energy_.compute_W(F_AD, f0, s0, n0);

    ad_helper.register_dependent_variable(W_ad);
    ad_helper.stop_recording_operations(false);
  } else {
    ad_helper.activate_recorded_tape(tape_index);
    ad_helper.set_independent_variable(F, F_linearized);
  }

  ad_helper.compute_gradient(P_flat);
  ad_helper.compute_hessian(C_flat);

  P = ad_helper.extract_gradient_component(P_flat, F_linearized);
  C = ad_helper.extract_hessian_component(C_flat, F_linearized, F_linearized);
}

} // namespace cardiac
