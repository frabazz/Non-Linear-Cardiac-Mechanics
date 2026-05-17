#pragma once

#include "strain_energy.hpp"
#include "../constants.hpp"
#include <deal.II/base/tensor.h>

using namespace dealii;

namespace cardiac {

class GuccioneEnergy : public StrainEnergy {
public:
  struct Params {
    double mu_hook = constants::guccione::MU_HOOK;
    double k_hook  = constants::guccione::K_HOOK;
    double b_ff    = constants::guccione::B_FF;
    double b_ss    = constants::guccione::B_SS;
    double b_nn    = constants::guccione::B_NN;
    double b_fs    = constants::guccione::B_FS;
    double b_fn    = constants::guccione::B_FN;
    double b_sn    = constants::guccione::B_SN;
    double C_param = constants::guccione::C_PARAM;
  };

  GuccioneEnergy() = default;
  explicit GuccioneEnergy(const Params &p)
  {
    mu_hook = p.mu_hook; k_hook  = p.k_hook;
    b_ff    = p.b_ff;    b_ss    = p.b_ss;    b_nn    = p.b_nn;
    b_fs    = p.b_fs;    b_fn    = p.b_fn;    b_sn    = p.b_sn;
    C_param = p.C_param;
  }

  ADNumberType compute_W(const Tensor<2, 3, ADNumberType> &F,
                          const Tensor<1, 3>               &f,
                          const Tensor<1, 3>               &s,
                          const Tensor<1, 3>               &n) const override
  {
    constexpr unsigned int dim = 3;
    using ADTensor2 = Tensor<2, dim, ADNumberType>;

    const ADNumberType J_ad = determinant(F);
    const ADTensor2    C_ad = transpose(F) * F;

    ADTensor2 I;
    for (unsigned int i = 0; i < dim; ++i) I[i][i] = 1.0;
    const ADTensor2 E_ad = 0.5 * (C_ad - I);

    const ADNumberType E_ff = scalar_product(f, E_ad * f);
    const ADNumberType E_ss = scalar_product(s, E_ad * s);
    const ADNumberType E_nn = scalar_product(n, E_ad * n);
    const ADNumberType E_fs = scalar_product(f, E_ad * s);
    const ADNumberType E_fn = scalar_product(f, E_ad * n);
    const ADNumberType E_sn = scalar_product(s, E_ad * n);

    const ADNumberType Q =
        b_ff * E_ff * E_ff +
        b_ss * E_ss * E_ss +
        b_nn * E_nn * E_nn +
        2.0 * b_fs * E_fs * E_fs +
        2.0 * b_fn * E_fn * E_fn +
        2.0 * b_sn * E_sn * E_sn;

    ADNumberType psi_ad = (C_param / 2.0) * (exp(Q) - 1.0);
    psi_ad += (k_hook / 2.0) * pow(J_ad - 1.0, 2.0);
    return psi_ad;
  }

private:
  double mu_hook = constants::guccione::MU_HOOK;
  double k_hook  = constants::guccione::K_HOOK;
  double b_ff    = constants::guccione::B_FF;
  double b_ss    = constants::guccione::B_SS;
  double b_nn    = constants::guccione::B_NN;
  double b_fs    = constants::guccione::B_FS;
  double b_fn    = constants::guccione::B_FN;
  double b_sn    = constants::guccione::B_SN;
  double C_param = constants::guccione::C_PARAM;
};

} // namespace cardiac
