# Run with: $env:OMP_NUM_THREADS = 8
#           mpiexec -n 1 python dynamic_loop_doo_mf.py
'''
Implement the Multi-Fidelity Method from overleaf/3method.tex (Sec. "Multi-Fidelity
Method", Alg. "Multi-Fidelity Method").

The high-fidelity (dynamic FAST.Farm) simulator and the low-fidelity (quasi-static
FLORIS+MoorPy) simulator are probed jointly with the same perturbation direction
u_{n,k}. Their two-point bandit gradients are fused per partial derivative term,
anchored on the low-fidelity clean gradient with a drift-free correction:

    g = g^{l,cln} + c * (g^h - g^l)

where g^h / g^l are the drifted HF / LF gradients (base at wind w_0, probe at wind
w_i) and g^{l,cln} is the clean synchronous LF gradient (both probes at w_0).
Because the environmental drift is common-mode to both simulators it cancels in the
high-minus-low difference, so the correction (g^h - g^l) is drift-free.

The control-variate coefficient c is estimated online. For the LF-anchored estimator
g = A + c*D with anchor A = g^{l,cln} and correction D = g^h - g^l, the
variance-minimizing coefficient is

    c* = -Cov(A, D) / Var(D),

maintained with an exponential moving average of the moments E[A*D], E[A], E[D],
E[D^2] (cf. the correlation-based EMA of liu2025multi), and clamped to [0, 1]
(c = 0 -> low-fidelity only; c = 1 -> drift-corrected high-fidelity).

The dynamic interface exposes one power measurement per ``interface.step()`` (the
measurement belongs to the yaw command sent one step earlier), so a single bandit
update spans N+1 consecutive steps (base + one probe per turbine), exactly as in
``dynamic_loop_doo_hf.py``.
'''
import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ffconnect.interface import FastFarmInterface
from ffconnect.simul_utils import get_inflow_file_path
from openfast_toolbox.io.fast_input_file import FASTInputFile
from floris import FlorisModel
from static_simulation_interface import FLORISFarmSim
from dynamic_wind_profile import read_wind_profile, wind_at_time

_SCRIPT_DIR  = Path(__file__).resolve().parent
FLORIS_FILE  = str(_SCRIPT_DIR / 'static_simulation_inputs' / 'gch_floating.yaml')
SIM_CONFIG   = str(_SCRIPT_DIR / 'static_simulation_inputs' / 'sim_cfg.YAML')
YAW_BOUND    = 30.0                       # yaw limit (deg) — a control bound, not a dynamic-case value


def build_quasi_static(fstf_path):
    '''Read the dynamic FAST.Farm case and build the matching FLORIS+MoorPy simulator.'''
    fstf   = FASTInputFile(fstf_path)
    inflow_path = get_inflow_file_path(fstf_path)
    inflow = FASTInputFile(inflow_path)

    # turbine positions from the .fstf WindTurbines table (cols: WT_X, WT_Y, WT_Z, ...)
    wt = np.asarray(fstf["WindTurbines"], dtype=object)
    neutral_positions = np.array([
        wt[:, 0].astype(float),            # WT_X
        wt[:, 1].astype(float),            # WT_Y
    ])                                     # shape (2, N)

    # wind from the InflowWind file (steady or time-varying)
    wind_profile = read_wind_profile(inflow, inflow_path)
    wind_speed, wind_dir = wind_at_time(wind_profile, 0.0)

    fmodel = FlorisModel(FLORIS_FILE)
    SI = FLORISFarmSim(neutral_positions, 'FLORIS', 'MoorPy', SIM_CONFIG, fmodel)
    SI.reset_simulator(wind_direction=wind_dir, wind_speed=wind_speed,
                       available_turbines=np.ones(neutral_positions.shape[1], dtype=bool))
    SI.wind_profile = wind_profile
    return SI


def rated_power(SI):
    rated_per_turbine = max(SI.sim.core.farm.turbine_definitions[0]['power_thrust_table']['power'])  # kW
    return rated_per_turbine * SI.n_turbines * 1e3   # W


class TurbineAgent:
    '''Local state for one turbine: owns its own yaw and its own feasible box.'''
    def __init__(self, idx, neighbors, x_lo, x_hi, gamma=0.0):
        self.idx       = idx        # turbine index (0..N-1)
        self.neighbors = neighbors  # indices j in N_i (i-2, i-1, i+1, i+2)
        self.x_lo      = x_lo       # lower bound of X_i [deg]
        self.x_hi      = x_hi       # upper bound of X_i [deg]
        self.gamma     = gamma      # committed yaw gamma_i [deg]


# ================================= Initialization =================================
save_results = True
save_figure_option = True    # True → save all figures to dynamic_Data/<run_name>/  |  False → interactive plt.show()
print_result = False
visualization = True
DATA_DIR = "dynamic_Data"                            # all .pkl / figures live here
result_file_name = "mf_tpbe-1x2-timeVaryingWind.pkl"
# --- User-defined paths ---
FFexePath = "D:\\2_PhD_UBC\\Code\\FASTv355\\FAST.Farm_x64_OMP_v355.exe"                              # parallel executable (with SC_DLL_Full variant)
FFsimPath = "D:\\2_PhD_UBC\\Code\\DOO-FOWF\\dynamic_CaseCtrl\\1x2.fstf"
lift_constraint = True                              # To see whether nonconvexity influence convergence

# --- Initialize interface from .fstf file ---
interface = FastFarmInterface.from_file(FFsimPath, fast_farm_executable=FFexePath, per_turbine_flags=True)  # dynamic FAST.Farm handle (MPI super-controller)
interface.init()

n_turbines = interface.num_turbines           # number of turbines (read from the .fstf)
max_iter   = interface.max_iter               # number of timesteps (300 s / 3 s = 100)
dt         = interface.dt                     # dynamic timestep [s] (DT_Low = 3.0)

# --- Quasi-static low-fidelity oracle (built from the dynamic case) ---
SI = build_quasi_static(FFsimPath)
P_rate = rated_power(SI)                       # shared normalization for both fidelities

# --- Algorithm 1 parameters (N-turbine row, streamwise ordering) ---
GAMMA_MAX = YAW_BOUND                          # 30 deg
DELTA     = 2.0                                # exploration delta_t (deg)
XI        = 0.1                                # shrinkage xi_t -> effective bound 27 deg
ALPHA0    = 500.0                              # base stepsize alpha_0 (deg^2) -- TUNE ME
WARM_START = True                              # True -> warm-start quasi-static solver from measured layout
CV_EMA_ALPHA = 0.95                            # EMA smoothing for the online control-variate coefficient c
CV_STD_EPS   = 1e-8                            # guard: fall back to c=0 when Var(g^h - g^l) is below this

# adjacency: turbine i is coupled to its up/down-stream neighbors i-2, i-1, i+1, i+2
NEIGHBORS = [
    [j for j in (i - 2, i - 1, i + 1, i + 2) if 0 <= j < n_turbines]
    for i in range(n_turbines)
]

if lift_constraint:
    # no sign constraint: every turbine free on the full box
    X_LO = np.full(n_turbines, -GAMMA_MAX)
    X_HI = np.full(n_turbines,  GAMMA_MAX)
else:
    # opposite-yaw constraint: alternate sign quadrants along the row
    neg  = np.arange(n_turbines) % 2 == 0      # T0 neg, T1 pos, T2 neg, ...
    X_LO = np.where(neg, -GAMMA_MAX, 0.0)
    X_HI = np.where(neg, 0.0, GAMMA_MAX)

rng    = np.random.default_rng(42)             # seeded RNG for the +/-1 probe directions
AGENTS = [                                      # one local agent per turbine (each owns its own yaw)
    TurbineAgent(i, NEIGHBORS[i], X_LO[i], X_HI[i], gamma=0.0)
    for i in range(n_turbines)
]


# ================================= Output directory =================================
run_name = os.path.splitext(result_file_name)[0]        # e.g. "mf_tpbe-1x2-steadyWind"
run_dir  = os.path.join(DATA_DIR, run_name)             # e.g. "dynamic_Data/mf_tpbe-1x2-steadyWind"
os.makedirs(run_dir, exist_ok=True)


def distributed_doo_step_mf(agents, alpha, delta, xi, us, cv_coefficient,
                            c_base_hf, c_probe_hf,
                            c_base_lf, c_probe_lf, c_probe_lf_cln):
    '''Assemble the control-variate fused bandit gradient and update the per-turbine
    yaws (message-passing + projected-update tail of the Multi-Fidelity Algorithm).

    Each partial derivative term (own partial df_i/dgamma_i and cross partials
    df_j/dgamma_i for j in N_i) is fused independently, anchored on the clean LF
    gradient:

        g = g_lf_cln + c * (g_hf - g_lf)

    before being assembled into the full gradient for turbine i. The coefficient
    ``cv_coefficient`` is an online scalar shared by all partial terms.

    Parameters
    ----------
    agents       : list[TurbineAgent], one agent per turbine (owns gamma_i and box X_i).
    alpha        : float, stepsize alpha_k [deg^2].
    delta        : float, exploration delta_k [deg].
    xi           : float, shrinkage xi_k in (0,1); projection box is (1-xi)*X_i.
    us           : dict, {turb_idx: +1/-1} probe directions of the current cycle.
    cv_coefficient : float, online control-variate coefficient in [0, 1].
    c_base_hf    : (N,) float, HF cost vector f^h_k(gamma) at w_0.
    c_probe_hf   : dict, {i: (N,) float} HF cost at (gamma_i + delta*u_i, gamma_{-i}), w_i.
    c_base_lf    : (N,) float, LF cost vector f^l_k(gamma) at w_0.
    c_probe_lf   : dict, {i: (N,) float} LF drifted cost at (gamma_i + delta*u_i), w_i.
    c_probe_lf_cln : dict, {i: (N,) float} LF clean cost at (gamma_i + delta*u_i), w_0.

    Returns
    -------
    gamma        : (N,) float, updated yaws gamma_{i,k+1}.
    grad         : (N,) float, assembled fused gradient estimate ghat_{i,k}.
    grad_hf      : (N,) float, assembled HF gradient (diagnostic).
    grad_lf      : (N,) float, assembled drifted LF gradient (diagnostic).
    grad_lf_cln  : (N,) float, assembled clean LF gradient (diagnostic).
    c_base_hf    : (N,) float, HF base cost (unchanged, for logging).
    A            : (M,) float, anchor partials g_lf_cln (for the online c update).
    D            : (M,) float, correction partials g_hf - g_lf (for the online c update).
    '''
    N = len(agents)

    grad        = np.zeros(N)
    grad_hf     = np.zeros(N)
    grad_lf     = np.zeros(N)
    grad_lf_cln = np.zeros(N)
    A_vals      = []
    D_vals      = []

    # ---- per-turbine fused partials + message passing ----
    for a in agents:
        i  = a.idx
        u  = us[i]

        # own partial: df_i/dgamma_i (probe i, cost i)
        gh   = (c_probe_hf[i][i]     - c_base_hf[i]) / delta * u
        gl   = (c_probe_lf[i][i]     - c_base_lf[i]) / delta * u
        glcn = (c_probe_lf_cln[i][i] - c_base_lf[i]) / delta * u
        grad[i]        += glcn + cv_coefficient * (gh - gl)
        grad_hf[i]     += gh
        grad_lf[i]     += gl
        grad_lf_cln[i] += glcn
        A_vals.append(glcn)
        D_vals.append(gh - gl)

        # cross partials: df_j/dgamma_i for neighbor j -> sends to i (probe i, cost j)
        for j in a.neighbors:
            gh   = (c_probe_hf[i][j]     - c_base_hf[j]) / delta * u
            gl   = (c_probe_lf[i][j]     - c_base_lf[j]) / delta * u
            glcn = (c_probe_lf_cln[i][j] - c_base_lf[j]) / delta * u
            grad[i]        += glcn + cv_coefficient * (gh - gl)
            grad_hf[i]     += gh
            grad_lf[i]     += gl
            grad_lf_cln[i] += glcn
            A_vals.append(glcn)
            D_vals.append(gh - gl)

    # ---- local projected update (each agent updates only itself) ----
    for a in agents:
        lo = (1.0 - xi) * a.x_lo                   # shrunken box keeps the next probe feasible
        hi = (1.0 - xi) * a.x_hi
        a.gamma = float(np.clip(a.gamma - alpha * grad[a.idx], lo, hi))  # projected gradient descent

    gamma = np.array([a.gamma for a in agents])
    return (gamma, grad, grad_hf, grad_lf, grad_lf_cln, c_base_hf,
            np.asarray(A_vals), np.asarray(D_vals))


class MFBanditController:
    '''Stateful multi-fidelity two-point bandit controller.

    Mirrors ``HFBanditController``'s N+1-step probe cycle (base + one probe per
    turbine), but at every high-fidelity query instant also evaluates the
    low-fidelity model at the same time/wind/yaw with the same perturbation
    direction ``u``:

        step 0  -> base    (all yaws at committed gamma, wind w_0)
        step i  -> probe_i (turbine i at gamma_i + delta*u_i, wind w_i), i = 1..N

    At the base step the clean LF base cost f^l(gamma, w_0) is recorded; at each
    probe step the drifted LF cost f^l(gamma + delta*u_i, w_i) and the clean LF cost
    f^l(gamma + delta*u_i, w_0) are recorded. On cycle completion the fused gradient
    is assembled with the online control-variate coefficient ``cv_coefficient``, which is then
    updated from the anchor/correction partials (A, D) of that cycle.
    '''
    def __init__(self, agents, SI, P_rate, alpha0, delta, xi, gamma_max, rng,
                 warm_start=True, cv_ema_alpha=0.95, cv_std_eps=1e-8):
        self.agents      = agents
        self.SI          = SI
        self.P_rate      = P_rate
        self.alpha0      = alpha0
        self.delta       = delta
        self.xi          = xi
        self.gamma_max   = gamma_max
        self.rng         = rng
        self.warm_start  = warm_start
        self.cv_ema_alpha = cv_ema_alpha
        self.cv_std_eps   = cv_std_eps
        self.N           = len(agents)
        self.k           = 0          # number of completed algorithm updates
        self.phase       = 0          # 0 -> base active; i in 1..N -> probe_{i-1} active
        self.us          = {}         # turb_idx -> +/-1 probe direction of current cycle

        # online control-variate coefficient (LF-anchored, in [0, 1])
        self.cv_coefficient = 0.0    # start from pure LF before any correlation data
        self._ema_cnt    = 0          # number of EMA updates so far
        self._ema_AD     = 0.0        # E[A * D]
        self._ema_A      = 0.0        # E[A]
        self._ema_D      = 0.0        # E[D]
        self._ema_D2     = 0.0        # E[D^2]

        # high-fidelity costs (dynamic simulator)
        self.c_base_hf   = None
        self.c_probe_hf  = {}
        # low-fidelity costs (quasi-static simulator)
        self.c_base_lf   = None
        self.c_probe_lf  = {}         # drifted: probe at w_i
        self.c_probe_lf_cln = {}      # clean:   probe at w_0
        # base-step wind + layout (for the clean synchronous LF evals)
        self.w0          = None
        self.base_layout = None

        # logging of the last completed update
        self.last_grad         = np.full(self.N, np.nan)
        self.last_grad_hf      = np.full(self.N, np.nan)
        self.last_grad_lf      = np.full(self.N, np.nan)
        self.last_grad_lf_cln  = np.full(self.N, np.nan)
        self.last_c_base       = np.full(self.N, np.nan)
        self.last_c_base_lf    = np.full(self.N, np.nan)

    def _gamma(self):
        return np.array([a.gamma for a in self.agents])

    def _probe_yaw(self, i):
        '''Reconstruct the yaw vector that probed turbine i in the current cycle.'''
        gamma = self._gamma()
        gp = gamma.copy()
        gp[i] = np.clip(gamma[i] + self.delta * self.us[i], -self.gamma_max, self.gamma_max)
        return gp

    def _lf_cost(self, yaws, wind, layout):
        '''Evaluate the low-fidelity cost f^l(yaws; wind) = -P^l/P_rate, seeded from
        ``layout`` (the measured dynamic layout).'''
        ws, wd = wind
        self.SI.sim.set(wind_speeds=[ws], wind_directions=[wd])
        self.SI.take_step(np.asarray(yaws, dtype=float),
                          warm_start=self.warm_start, initial_layout=layout)
        return -self.SI.get_powers().squeeze() / self.P_rate

    def record(self, powers, wind, layout):
        '''Store the cost of the just-completed high-fidelity step and evaluate the
        low-fidelity model at the same instant (same time/wind/yaw).'''
        c_hf = -np.asarray(powers, dtype=float) / self.P_rate
        gamma = self._gamma()
        if self.phase == 0:
            self.c_base_hf   = c_hf
            self.w0          = wind
            self.base_layout = layout.copy()
            self.c_base_lf   = self._lf_cost(gamma, self.w0, self.base_layout)
        else:
            i = self.phase - 1
            self.c_probe_hf[i]     = c_hf
            self.c_probe_lf[i]     = self._lf_cost(self._probe_yaw(i), wind, layout)
            self.c_probe_lf_cln[i] = self._lf_cost(self._probe_yaw(i), self.w0, self.base_layout)

    def next_command(self):
        '''Return the yaw command for the next step; every N+1 steps also returns the update.'''
        gamma = self._gamma()

        if self.phase == 0:
            # base just measured -> issue probe of turbine 0
            u = 1.0 if self.rng.uniform() < 0.5 else -1.0
            self.us[0] = u
            yaw = gamma.copy()
            yaw[0] = np.clip(gamma[0] + self.delta * u, -self.gamma_max, self.gamma_max)
            self.phase = 1
            return yaw, None

        if self.phase < self.N:
            # probe (phase-1) just measured -> issue probe of turbine (phase)
            i = self.phase
            u = 1.0 if self.rng.uniform() < 0.5 else -1.0
            self.us[i] = u
            yaw = gamma.copy()
            yaw[i] = np.clip(gamma[i] + self.delta * u, -self.gamma_max, self.gamma_max)
            self.phase += 1
            return yaw, None

        # last probe just measured -> assemble fused gradient and update
        alpha = self.alpha0 / np.sqrt(self.k + 1.0)
        (gamma_new, grad, grad_hf, grad_lf, grad_lf_cln, c_base_hf,
         A_vals, D_vals) = distributed_doo_step_mf(
            self.agents, alpha, self.delta, self.xi, self.us, self.cv_coefficient,
            self.c_base_hf, self.c_probe_hf,
            self.c_base_lf, self.c_probe_lf, self.c_probe_lf_cln,
        )
        self._update_cv(A_vals, D_vals)
        self.last_grad        = grad
        self.last_grad_hf     = grad_hf
        self.last_grad_lf     = grad_lf
        self.last_grad_lf_cln = grad_lf_cln
        self.last_c_base      = c_base_hf
        self.last_c_base_lf   = self.c_base_lf
        self.k += 1
        self.phase       = 0
        self.us          = {}
        self.c_base_hf   = None
        self.c_probe_hf  = {}
        self.c_base_lf   = None
        self.c_probe_lf  = {}
        self.c_probe_lf_cln = {}
        self.w0          = None
        self.base_layout = None
        return gamma_new, (gamma_new, grad, c_base_hf)

    def _update_cv(self, A, D):
        '''Update the online control-variate coefficient from the anchor A = g_lf_cln
        and correction D = g_hf - g_lf partials of the completed cycle.

        The variance-minimizing coefficient for the LF-anchored estimator
        g = A + c*D is c = -Cov(A, D) / Var(D). Cov/Var are maintained with an
        exponential moving average of the moments E[A*D], E[A], E[D], E[D^2]
        (cf. liu2025multi), and c is clamped to [0, 1].
        '''
        A = np.asarray(A, dtype=float)
        D = np.asarray(D, dtype=float)
        if A.size == 0:
            return
        mean_AD = float(np.mean(A * D))
        mean_A  = float(np.mean(A))
        mean_D  = float(np.mean(D))
        mean_D2 = float(np.mean(D * D))

        if self._ema_cnt == 0:
            self._ema_AD  = mean_AD
            self._ema_A   = mean_A
            self._ema_D   = mean_D
            self._ema_D2  = mean_D2
        else:
            b = self.cv_ema_alpha
            self._ema_AD  = b * self._ema_AD  + (1.0 - b) * mean_AD
            self._ema_A   = b * self._ema_A   + (1.0 - b) * mean_A
            self._ema_D   = b * self._ema_D   + (1.0 - b) * mean_D
            self._ema_D2  = b * self._ema_D2  + (1.0 - b) * mean_D2
        self._ema_cnt += 1

        cov = self._ema_AD - self._ema_A * self._ema_D
        var = self._ema_D2 - self._ema_D * self._ema_D
        if var < self.cv_std_eps:
            self.cv_coefficient = 0.0
        else:
            self.cv_coefficient = float(np.clip(-cov / var, 0.0, 1.0))


# --- Multi-fidelity bandit controller (stateful; drives the joint HF/LF probe cycle) ---
mf_ctrl = MFBanditController(AGENTS, SI, P_rate, ALPHA0, DELTA, XI, GAMMA_MAX, rng,
                             warm_start=WARM_START,
                             cv_ema_alpha=CV_EMA_ALPHA, cv_std_eps=CV_STD_EPS)

# --- Simulation loop ---
results = []
done = False
run_start = time.perf_counter()

for t in range(max_iter):
    # Advance one step FIRST, then read the freshly computed measures.
    # current_measures is only populated inside step(), so stepping first
    # guarantees a valid (non-NaN) read — no spin-up NaN to filter around.
    done = interface.step()

    step_result = {"t": (t + 1) * dt}
    total_power = 0.0

    # 1. Get measurements and store for all WTs
    for i in range(n_turbines):
        power_i   = interface.get_measure("power",        turb_idx=i)
        pitch_b1  = interface.get_measure("pitch",        turb_idx=i)
        pitch_b2  = interface.get_measure("pitch_blade2", turb_idx=i)
        pitch_b3  = interface.get_measure("pitch_blade3", turb_idx=i)
        yaw_i     = interface.get_measure("yaw",          turb_idx=i)
        loads_i   = interface.get_measure("load",         turb_idx=i)
        tdx_i     = interface.get_measure("platform_tdx", turb_idx=i)  # platform surge translational DISPLACEMENT (X),     unit: m
        tdy_i     = interface.get_measure("platform_tdy", turb_idx=i)  # platform sway  translational DISPLACEMENT (Y),     unit: m
        tdz_i     = interface.get_measure("platform_tdz", turb_idx=i)  # platform heave translational DISPLACEMENT (Z),     unit: m
        rdx_i     = interface.get_measure("platform_rdx", turb_idx=i)  # platform roll  rotational DISPLACEMENT (about X), unit: rad (converted to deg before storing)
        rdy_i     = interface.get_measure("platform_rdy", turb_idx=i)  # platform pitch rotational DISPLACEMENT (about Y), unit: rad (converted to deg before storing)
        rdz_i     = interface.get_measure("platform_rdz", turb_idx=i)  # platform yaw   rotational DISPLACEMENT (about Z), unit: rad (converted to deg before storing)
        azimuth_i = interface.get_measure("azimuth",      turb_idx=i)  # radians, blade 1
        wind_spd_i = interface.get_measure("wind_speed",     turb_idx=i)
        wind_dir_i = interface.get_measure("wind_direction",  turb_idx=i)

        # 2. Store results
        step_result[f"power_T{i}"]          = power_i.item()
        step_result[f"pitch_b1_T{i}"]       = np.degrees(pitch_b1.item())
        step_result[f"pitch_b2_T{i}"]       = np.degrees(pitch_b2.item())
        step_result[f"pitch_b3_T{i}"]       = np.degrees(pitch_b3.item())
        step_result[f"yaw_T{i}"]            = np.degrees(yaw_i.item())                  # current yaw angle (deg)
        step_result[f"loads_T{i}"]          = loads_i[:3].copy()
        step_result[f"azimuth_T{i}"]        = np.degrees(azimuth_i.item())
        step_result[f"platform_tdx_T{i}"]   = tdx_i.item()
        step_result[f"platform_tdy_T{i}"]   = tdy_i.item()
        step_result[f"platform_tdz_T{i}"]   = tdz_i.item()
        step_result[f"platform_rdx_T{i}"]   = np.degrees(rdx_i.item())
        step_result[f"platform_rdy_T{i}"]   = np.degrees(rdy_i.item())
        step_result[f"platform_rdz_T{i}"]   = np.degrees(rdz_i.item())
        step_result[f"wind_speed_T{i}"]     = wind_spd_i.item()
        step_result[f"wind_direction_T{i}"] = wind_dir_i.item()
        total_power += power_i.item()
    results.append(step_result)

    # 3. Joint multi-fidelity two-point bandit update. The wind and dynamic layout
    #    are sampled at the current high-fidelity instant so the low-fidelity model
    #    sees the same time/wind/yaw as the high-fidelity simulator.
    ws_t, wd_t = wind_at_time(SI.wind_profile, (t + 1) * dt)
    step_result["wind_speed_SI"]     = ws_t
    step_result["wind_direction_SI"] = wd_t

    # Seed the LF gradient probes from the actual measured farm layout (reality),
    # not from the quasi-static simulator's own converged layout.
    surge = np.array([step_result[f"platform_tdx_T{i}"] for i in range(n_turbines)])
    sway  = np.array([step_result[f"platform_tdy_T{i}"] for i in range(n_turbines)])
    dynamic_layout = np.vstack([
        SI.neutral_positions[0] + surge,
        SI.neutral_positions[1] + sway,
    ])

    t0 = time.perf_counter()
    powers = np.array([step_result[f"power_T{i}"] for i in range(n_turbines)])
    mf_ctrl.record(powers, (ws_t, wd_t), dynamic_layout)
    yaw_cmd, _ = mf_ctrl.next_command()

    # 4. Send commands (buffered, consumed by the NEXT step())
    for i in range(n_turbines):
        interface.set_command(yaw=float(yaw_cmd[i]), ipc=None, torque=None, turb_idx=i)
        step_result[f"ctrl_yaw_cmd_T{i}"] = float(yaw_cmd[i])        # base or probe command
        step_result[f"gamma_T{i}"]        = float(AGENTS[i].gamma)   # committed yaw anchor

    # grad / cost / coefficient reflect the most recent completed update (NaN until the first)
    step_result["grad"]         = mf_ctrl.last_grad.copy()
    step_result["grad_hf"]      = mf_ctrl.last_grad_hf.copy()
    step_result["grad_lf"]      = mf_ctrl.last_grad_lf.copy()
    step_result["grad_lf_cln"]  = mf_ctrl.last_grad_lf_cln.copy()
    step_result["cost_base"]    = mf_ctrl.last_c_base.copy()
    step_result["qs_cost"]      = mf_ctrl.last_c_base_lf.copy()
    step_result["cv_coefficient"] = float(mf_ctrl.cv_coefficient)
    step_result["ctrl_compute_time"] = time.perf_counter() - t0

    if print_result:
        yaw_cmds = "  ".join(
            f"yaw_cmd_T{i}={step_result[f'ctrl_yaw_cmd_T{i}']:+6.2f}deg"
            f" (meas {step_result[f'yaw_T{i}']:+6.2f}deg)"
            for i in range(n_turbines)
        )
        print(f"t={(t+1)*dt:6.1f}s  power={total_power/1e6:.3f} MW  |  {yaw_cmds}", flush=True)

    if done:
        break

print(f"Total wall-clock: {time.perf_counter() - run_start:.3f}s")

# ================================= Save Results =================================
if save_results:
    out_path = os.path.join(run_dir, result_file_name)
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {out_path} ({len(results)} steps)", flush=True)

# ================================= Plotting =================================
t_arr = np.array([r["t"] for r in results])

def _savefig(fig, name):
    if save_figure_option:
        path = os.path.join(run_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    if visualization:
        plt.show()
    else:
        plt.close(fig)

if visualization or save_figure_option:
    farm_power = np.array(
        [sum(r[f"power_T{i}"] for i in range(n_turbines)) for r in results]
    ) / 1e6                                    # W -> MW

    # 1. Power: farm + individual turbines
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(t_arr, farm_power, label="Farm", lw=2)
    for i in range(n_turbines):
        ax1.plot(t_arr, [r[f"power_T{i}"] / 1e6 for r in results],
                 label=f"T{i}", ls="--")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Power (MW)")
    ax1.set_title("Farm and individual turbine power")
    ax1.legend(); ax1.grid(True); fig1.tight_layout()
    _savefig(fig1, "power")

    # 2. Yaw: commanded + measured per turbine
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    for i in range(n_turbines):
        ax2.plot(t_arr, [r[f"ctrl_yaw_cmd_T{i}"] for r in results],
                 label=f"T{i} cmd", marker="o", ms=3)
        ax2.plot(t_arr, [r[f"yaw_T{i}"] for r in results],
                 label=f"T{i} meas", ls="--")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Yaw (deg)")
    ax2.set_title("Yaw command vs. measured yaw")
    ax2.legend(); ax2.grid(True); fig2.tight_layout()
    _savefig(fig2, "yaw")

    # 3. Cost: realized dynamic + HF base (high-fidelity) + LF base (quasi-static)
    cost_realized = -farm_power * 1e6 / P_rate           # f_t = -P_real/P_rate
    cost_base     = np.array([r["cost_base"].sum() for r in results])
    cost_qs       = np.array([r["qs_cost"].sum() for r in results])
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(t_arr, cost_realized,
             label=r"$f_t=-P_{\mathrm{real}}/P_{\mathrm{rate}}$", lw=2)
    ax3.plot(t_arr, cost_base, label="base (high-fidelity)", ls="--")
    ax3.plot(t_arr, cost_qs,   label="base (low-fidelity)", ls=":")
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Cost")
    ax3.set_title("Cost function value")
    ax3.legend(); ax3.grid(True); fig3.tight_layout()
    _savefig(fig3, "cost")

    # 4. Fused gradient estimate per turbine
    grad_arr = np.array([r["grad"] for r in results])    # (T, N)
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    for i in range(n_turbines):
        ax4.plot(t_arr, grad_arr[:, i], label=f"T{i}")
    ax4.axhline(0, color="k", lw=0.5)
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Gradient (1/deg)")
    ax4.set_title("Fused gradient estimate")
    ax4.legend(); ax4.grid(True); fig4.tight_layout()
    _savefig(fig4, "gradient")

    # 5. Gradient components (fused / HF / drifted LF / clean LF) per turbine
    grad_hf_arr     = np.array([r["grad_hf"] for r in results])
    grad_lf_arr     = np.array([r["grad_lf"] for r in results])
    grad_lf_cln_arr = np.array([r["grad_lf_cln"] for r in results])
    grad_fused_arr  = np.array([r["grad"] for r in results])
    fig5, axes5 = plt.subplots(n_turbines, 1, sharex=True, figsize=(8, 3 * n_turbines))
    axes5 = np.atleast_1d(axes5)
    for i in range(n_turbines):
        ax = axes5[i]
        ax.plot(t_arr, grad_fused_arr[:, i],  label="Fused",    lw=2)
        ax.plot(t_arr, grad_hf_arr[:, i],     label="HF",       ls="-")
        ax.plot(t_arr, grad_lf_arr[:, i],     label="LF drift", ls="--")
        ax.plot(t_arr, grad_lf_cln_arr[:, i], label="LF clean", ls=":")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(f"T{i} (1/deg)")
        ax.legend(); ax.grid(True)
    axes5[-1].set_xlabel("Time (s)")
    fig5.suptitle("Gradient components per turbine")
    fig5.tight_layout()
    _savefig(fig5, "gradient_components")

    # 6. Online control-variate coefficient over time
    cv_coeff_arr = np.array([r["cv_coefficient"] for r in results])
    fig6, ax6 = plt.subplots(figsize=(8, 4))
    ax6.plot(t_arr, cv_coeff_arr, label="cv_coefficient", lw=2)
    ax6.axhline(0.0, color="k", lw=0.5, ls="--")
    ax6.axhline(1.0, color="k", lw=0.5, ls="--")
    ax6.set_xlabel("Time (s)"); ax6.set_ylabel("cv_coefficient")
    ax6.set_title("Online control-variate coefficient")
    ax6.set_ylim(-0.05, 1.05)
    ax6.legend(); ax6.grid(True); fig6.tight_layout()
    _savefig(fig6, "cv_coefficient")
