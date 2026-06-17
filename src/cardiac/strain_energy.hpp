#pragma once

#include <deal.II/base/tensor.h>
#include <deal.II/differentiation/ad/ad_drivers.h>
#include <deal.II/differentiation/ad/ad_helpers.h>

#include "../common.hpp"

#include "../constants.hpp"
#include <deal.II/base/tensor.h>


namespace cardiac {

class StrainEnergy {
public:
  using ADHelper     = AutoDiff::ScalarFunction<3, ADTypeCode, double>;
  using ADNumberType = typename ADHelper::ad_type;

  virtual ~StrainEnergy() = default;

  virtual ADNumberType compute_W(const dealii::Tensor<2, 3, ADNumberType> &F,
                                  const dealii::Tensor<1, 3>               &f0,
                                  const dealii::Tensor<1, 3>               &s0,
                                  const dealii::Tensor<1, 3>               &n0) const = 0;
};

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

  ADNumberType compute_W(const dealii::Tensor<2, 3, ADNumberType> &F,
                         const dealii::Tensor<1, 3>               &f,
                         const dealii::Tensor<1, 3>               &s,
                         const dealii::Tensor<1, 3>               &n) const override
  {
    constexpr unsigned int dim = 3;
    using ADTensor2 = dealii::Tensor<2, dim, ADNumberType>;

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

class BeamEnergy : public StrainEnergy {
public:
  struct Params {
    double mu_hook = constants::beam::MU_HOOK;
    double k_hook  = constants::beam::K_HOOK;
    double b_ff    = constants::beam::B_FF;
    double b_ss    = constants::beam::B_SS;
    double b_nn    = constants::beam::B_NN;
    double b_fs    = constants::beam::B_FS;
    double b_fn    = constants::beam::B_FN;
    double b_sn    = constants::beam::B_SN;
    double C_param = constants::beam::C_PARAM;
  };

  BeamEnergy() = default;
  explicit BeamEnergy(const Params &p)
  {
    mu_hook = p.mu_hook; k_hook  = p.k_hook;
    b_ff    = p.b_ff;    b_ss    = p.b_ss;    b_nn    = p.b_nn;
    b_fs    = p.b_fs;    b_fn    = p.b_fn;    b_sn    = p.b_sn;
    C_param = p.C_param;
  }

  ADNumberType compute_W(const dealii::Tensor<2, 3, ADNumberType> &F,
                         const dealii::Tensor<1, 3>               &f,
                         const dealii::Tensor<1, 3>               &s,
                         const dealii::Tensor<1, 3>               &n) const override
  {
    constexpr unsigned int dim = 3;
    using ADTensor2 = dealii::Tensor<2, dim, ADNumberType>;

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
    b_ff  * E_ff * E_ff                    
  + b_sn  * (E_ss*E_ss + E_nn*E_nn + 2.0*E_sn*E_sn)  
  + b_fs * 2.0*(E_fs*E_fs + E_fn*E_fn);

    ADNumberType psi_ad = (C_param / 2.0) * (exp(Q) - 1.0);
    psi_ad += (k_hook / 2.0) * pow(J_ad - 1.0, 2.0);
    return psi_ad;
  }

private:
  double mu_hook = constants::beam::MU_HOOK;
  double k_hook  = constants::beam::K_HOOK;
  double b_ff    = constants::beam::B_FF;
  double b_ss    = constants::beam::B_SS;
  double b_nn    = constants::beam::B_NN;
  double b_fs    = constants::beam::B_FS;
  double b_fn    = constants::beam::B_FN;
  double b_sn    = constants::beam::B_SN;
  double C_param = constants::beam::C_PARAM;
};


class NeoHookeEnergy : public StrainEnergy {
public:
  struct Params {
    double mu    = constants::NeoHooke::MU_HOOK;
    double kappa = constants::NeoHooke::K_HOOK;
  };

  NeoHookeEnergy() = default;
  explicit NeoHookeEnergy(const Params &p) : mu(p.mu), kappa(p.kappa) {}

  ADNumberType compute_W(const dealii::Tensor<2, 3, ADNumberType> &F,
                         const dealii::Tensor<1, 3>               & /*f0*/,
                         const dealii::Tensor<1, 3>               & /*s0*/,
                         const dealii::Tensor<1, 3>               & /*n0*/) const override
  {
    const ADNumberType J_ad  = determinant(F);
    const ADNumberType I1_ad = trace(transpose(F) * F);

    ADNumberType psi_ad = (mu / 2.0) * (I1_ad * std::pow(J_ad, -2.0 / 3.0) - 3.0);
    psi_ad += (kappa / 2.0) * std::pow(J_ad - 1.0, 2.0);
    return psi_ad;
  }

private:
  double mu    = constants::NeoHooke::MU_HOOK;
  double kappa = constants::NeoHooke::K_HOOK;
};

class HolzapfelOgdenEnergy : public StrainEnergy {
public:
  struct Params {
    double a     = constants::holzapfel::A;
    double b     = constants::holzapfel::B;
    double a_f   = constants::holzapfel::A_F;
    double b_f   = constants::holzapfel::B_F;
    double a_s   = constants::holzapfel::A_S;
    double b_s   = constants::holzapfel::B_S;
    double a_fs  = constants::holzapfel::A_FS;
    double b_fs  = constants::holzapfel::B_FS;
    double kappa = constants::holzapfel::KAPPA;
  };

  HolzapfelOgdenEnergy() = default;
  explicit HolzapfelOgdenEnergy(const Params &p)
  {
    a = p.a; b = p.b;
    a_f = p.a_f; b_f = p.b_f;
    a_s = p.a_s; b_s = p.b_s;
    a_fs = p.a_fs; b_fs = p.b_fs;
    kappa = p.kappa;
  }

  ADNumberType compute_W(const dealii::Tensor<2, 3, ADNumberType> &F,
                         const dealii::Tensor<1, 3>               &f0,
                         const dealii::Tensor<1, 3>               &s0,
                         const dealii::Tensor<1, 3>               & /*n0*/) const override
  {
    constexpr unsigned int dim = 3;
    using ADTensor2 = dealii::Tensor<2, dim, ADNumberType>;

    const ADNumberType J_ad  = determinant(F);
    const ADTensor2    C_ad  = transpose(F) * F;
    const ADNumberType I1_ad = trace(C_ad);

    ADNumberType I4f_ad  = 0.0;
    ADNumberType I4s_ad  = 0.0;
    ADNumberType I8fs_ad = 0.0;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j) {
        I4f_ad  += f0[i] * C_ad[i][j] * f0[j];
        I4s_ad  += s0[i] * C_ad[i][j] * s0[j];
        I8fs_ad += f0[i] * C_ad[i][j] * s0[j];
      }

    ADNumberType psi_ad = (a / (2.0 * b)) * (std::exp(b * (I1_ad - 3.0)) - 1.0);

    if (I4f_ad > 1.0)
      psi_ad += (a_f / (2.0 * b_f)) * (std::exp(b_f * std::pow(I4f_ad - 1.0, 2.0)) - 1.0);
    if (I4s_ad > 1.0)
      psi_ad += (a_s / (2.0 * b_s)) * (std::exp(b_s * std::pow(I4s_ad - 1.0, 2.0)) - 1.0);

    psi_ad += (a_fs / (2.0 * b_fs)) * (std::exp(b_fs * std::pow(I8fs_ad, 2.0)) - 1.0);
    psi_ad += (kappa / 2.0) * std::pow(J_ad - 1.0, 2.0);

    return psi_ad;
  }

private:
  double a     = constants::holzapfel::A;
  double b     = constants::holzapfel::B;
  double a_f   = constants::holzapfel::A_F;
  double b_f   = constants::holzapfel::B_F;
  double a_s   = constants::holzapfel::A_S;
  double b_s   = constants::holzapfel::B_S;
  double a_fs  = constants::holzapfel::A_FS;
  double b_fs  = constants::holzapfel::B_FS;
  double kappa = constants::holzapfel::KAPPA;
};

  
} // namespace cardiac
