#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>

namespace cardiac {

class FiberField {
public:
  virtual ~FiberField() = default;

  virtual void frame_at(const dealii::DoFHandler<3>::active_cell_iterator &cell,
                         unsigned int                                       q_point,
                         const dealii::Point<3>                            &x,
                         dealii::Tensor<1, 3>                              &f0,
                         dealii::Tensor<1, 3>                              &s0,
                         dealii::Tensor<1, 3>                              &n0) const = 0;
};

} // namespace cardiac
