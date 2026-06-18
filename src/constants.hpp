#pragma once

namespace cardiac {
namespace constants {

// boundary tags
constexpr unsigned int BID_DIRICHLET = 2;
constexpr unsigned int BID_PRESSURE  = 3;
constexpr unsigned int BID_ROBIN     = 4;

// Newton solver
constexpr unsigned int NEWTON_MAX_ITER  = 1000;
constexpr double       NEWTON_TOLERANCE = 1.0e-3;

// backtracking line search
constexpr unsigned int LS_MAX_BACKTRACKS   = 25;
constexpr double       LS_REL_DECREASE_EPS = 1e-8;
constexpr double       LS_ABS_DECREASE_EPS = 1e-14;
constexpr double       LS_ALPHA_MIN        = 1e-7;
constexpr double       LS_STEP_TOL_FACTOR  = 1e-12;

// fiber helix angles [degrees] — shared by both models
constexpr double FIBER_ALPHA_ENDO =  60.0;
constexpr double FIBER_ALPHA_EPI  = -60.0;

namespace holzapfel {
  constexpr double FIBER_BETA_ENDO = -30.0;
  constexpr double FIBER_BETA_EPI  =  30.0;
  constexpr double ALPHA_ROBIN     =  0.999918;

  constexpr unsigned int CONVERGENCE_STUDY_STEPS = 2;

  // pressure loading
  constexpr double P_START  = 0.1;
  constexpr double P_END    = 15.0;
  constexpr int    N_STEPS  = 50;

  // linear solver
  constexpr unsigned int GMRES_MAX_ITER    = 15000;
  constexpr double       GMRES_TOL_FACTOR  = 1.5e-4;

  constexpr unsigned int GMRES_RESTART     = 100;
  constexpr bool         AMG_ELLIPTIC      = true;
  constexpr double       AMG_AGG_THRESHOLD = 0.02;
  constexpr bool         AMG_CONST_MODES   = true;

  // Holzapfel-Ogden barbotta parameters
  constexpr double A     = 0.2;
  constexpr double B     = 4.610;
  constexpr double A_F   = 4.1907;
  constexpr double B_F   = 7.8565;
  constexpr double A_S   = 2.5640;
  constexpr double B_S   = 10.4460;
  constexpr double A_FS  = 0.1304;
  constexpr double B_FS  = 15.255;
  constexpr double KAPPA = 50.0;
}

namespace guccione {
  constexpr double       ALPHA_ROBIN             = 0.999918;
  constexpr double       PROLATE_D               = 2.91;
  constexpr double       PROLATE_XI_ENDO         = 0.6;
  constexpr double       PROLATE_XI_EPI          = 1.02;
  constexpr unsigned int CONVERGENCE_STUDY_STEPS = 25;

  // pressure loading
  constexpr double P_START = 1.0;
  constexpr double P_END   = 15.0;
  constexpr int    N_STEPS = 100;

  // linear solver
  constexpr unsigned int GMRES_MAX_ITER    = 15000;
  constexpr double       GMRES_TOL_FACTOR  = 1.5e-4;
  constexpr unsigned int GMRES_RESTART     = 30;
  constexpr bool         AMG_ELLIPTIC      = true;
  constexpr double       AMG_AGG_THRESHOLD = 1e-4;
  constexpr bool         AMG_CONST_MODES   = false;

  // Guccione material parameters
  constexpr double MU_HOOK = 4.0;
  constexpr double K_HOOK  = 50.0;
  constexpr double B_FF    = 8.0;
  constexpr double B_SS    = 2.0;
  constexpr double B_NN    = 2.0;
  constexpr double B_FS    = 4.0;
  constexpr double B_FN    = 4.0;
  constexpr double B_SN    = 2.0;
  constexpr double C_PARAM = 2.0;
}

namespace NeoHooke {
  constexpr double       ALPHA_ROBIN             = 0.999918;
  constexpr double       PROLATE_D               = 2.91;
  constexpr double       PROLATE_XI_ENDO         = 0.6;
  constexpr double       PROLATE_XI_EPI          = 1.02;
  constexpr unsigned int CONVERGENCE_STUDY_STEPS = 25;

  // pressure loading
  constexpr double P_START = 0.0;
  constexpr double P_END   = 3.0;
  constexpr int    N_STEPS = 20;

  // linear solver
  constexpr unsigned int GMRES_MAX_ITER    = 15000;
  constexpr double       GMRES_TOL_FACTOR  = 1.5e-4;
  constexpr unsigned int GMRES_RESTART     = 30;
  constexpr bool         AMG_ELLIPTIC      = true;
  constexpr double       AMG_AGG_THRESHOLD = 1e-4;
  constexpr bool         AMG_CONST_MODES   = false;

  // Guccione 1991 material parameters [kPa]
  constexpr double MU_HOOK = 4.0;
  constexpr double K_HOOK  = 50.0;

}

namespace beam {
  constexpr double       ALPHA_ROBIN             = 0.999918;
  constexpr double       PROLATE_D               = 2.91;
  constexpr double       PROLATE_XI_ENDO         = 0.6;
  constexpr double       PROLATE_XI_EPI          = 1.02;
  constexpr unsigned int CONVERGENCE_STUDY_STEPS = 25;

  // pressure loading
  constexpr double P_START = 0.0004;
  constexpr double P_END   = 0.004;
  constexpr int    N_STEPS = 30;

  // linear solver
  constexpr unsigned int GMRES_MAX_ITER    = 15000;
  constexpr double       GMRES_TOL_FACTOR  = 0.1;
  constexpr unsigned int GMRES_RESTART     = 30;
  constexpr bool         AMG_ELLIPTIC      = true;
  constexpr double       AMG_AGG_THRESHOLD = 1e-4;
  constexpr bool         AMG_CONST_MODES   = false;

  // Guccione material parameters
  constexpr double MU_HOOK = 4.0;
  constexpr double K_HOOK  = 30.0;

  constexpr double B_FF    = 8.0;
  constexpr double B_SS    = 2.0;
  constexpr double B_NN    = 2.0;
  
  constexpr double B_FS    = 4.0;
  constexpr double B_FN    = 4.0;
  constexpr double B_SN    = 2.0;
  constexpr double C_PARAM = 2.0;

  constexpr double       NEWTON_TOLERANCE = 1.0e-3;
}


} // namespace constants
} // namespace cardiac

