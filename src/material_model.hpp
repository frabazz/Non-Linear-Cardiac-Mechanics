#pragma once

#include "common.hpp"

namespace cardiac {

// Unified input for compute_tensors: each model uses what it needs.
// Holzapfel uses f0/s0/n0 (from Poisson); Guccione uses p (prolate coords).
struct MaterialInput {
  dealii::Tensor<1, dim> f0, s0, n0;
  dealii::Point<dim>     p;
};

class IMaterialModel {
public:
  virtual void compute_tensors(dealii::Tensor<2, dim>  F,
                                dealii::Tensor<2, dim> &P,
                                dealii::Tensor<4, dim> &C,
                                const MaterialInput    &in) = 0;
  virtual ~IMaterialModel() = default;
};

} // namespace cardiac
