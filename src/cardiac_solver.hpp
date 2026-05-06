#pragma once

namespace cardiac {

class ISolver {
public:
  virtual void setup() = 0;
  virtual void solve() = 0;
  virtual ~ISolver() = default;
};

} // namespace cardiac
