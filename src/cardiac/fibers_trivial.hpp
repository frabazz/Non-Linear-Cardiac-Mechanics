#pragma once

#include "fiber_field.hpp"

namespace cardiac {

class TrivialFibers : public FiberField {
public:
  void frame_at(const dealii::DoFHandler<3>::active_cell_iterator & /*cell*/,
                 unsigned int                                        /*q_point*/,
                 const dealii::Point<3>                            & /*x*/,
                 dealii::Tensor<1, 3>                              &f0,
                 dealii::Tensor<1, 3>                              &s0,
                 dealii::Tensor<1, 3>                              &n0) const override
  {
    f0 = dealii::Tensor<1, 3>();  f0[0] = 1.0;
    s0 = dealii::Tensor<1, 3>();  s0[1] = 1.0;
    n0 = dealii::Tensor<1, 3>();  n0[2] = 1.0;
  }
};

} // namespace cardiac
