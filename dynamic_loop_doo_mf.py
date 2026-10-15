# Run with: $env:OMP_NUM_THREADS = 8
#           mpiexec -n 1 python dynamic_loop_doo_mf.py
'''
Implement the Multi-Fidelity Method from overleaf/3method.tex (Sec. "Multi-Fidelity
Method", Eq. mf-gradient, Alg. "Multi-Fidelity Method").

The high-fidelity (dynamic FAST.Farm) simulator and the low-fidelity (quasi-static
FLORIS+MoorPy) simulator are probed jointly with the same perturbation direction
u_{n,k} and at the same wind w_i (one-point bandit, no committed-point query), so
there is no time mismatch between the two fidelities. Their one-point bandit
gradients are fused per partial derivative term with a high-fidelity-anchored
control variate:

    g = g^h + c * (g^l - mu^l)

where g^h is the HF probe gradient (the anchor) and (g^l - mu^l) is the LF
correction that reduces variance, with mu^l = E[g^l] the low-fidelity gradient
mean. Since g^l is correlated with g^h (both share the probe direction u), the
correction cancels part of the HF variance while leaving the estimator unbiased.

mu^l is estimated from a batch of cheap LF probes: M independent Rademacher
directions are drawn (independent of the HF probe direction) and their one-point
LF gradient estimates are averaged. M = LF_MEAN_SAMPLES (M = 2 with the fixed
(+1, -1) pair reduces to the central-difference estimate).

The control-variate coefficient is a constant for now (CV_COEFF = 1.0) for
testing. The variance-minimizing coefficient is c* = -Cov(g^h, g^l) / Var(g^l),
which is typically negative here (positively correlated probes), so no [0, 1]
clamp is used.

The dynamic interface exposes one power measurement per ``interface.step()`` (the
measurement belongs to the yaw command sent one step earlier), so a single bandit
update spans N consecutive steps (one probe per turbine), exactly as in
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
result_file_name = "mf_opbe-1x2-timeVaryingWind.pkl"
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
ALPHA0    = 10.0                               # base stepsize alpha_0 (deg^2) -- TUNE ME (one-point bandit)
WARM_START = True                              # True -> warm-start quasi-static solver from measured layout
CV_COEFF         = 1.0                         # control-variate coefficient c (constant, for testing)
LF_MEAN_SAMPLES  = 10                          # M: cheap LF probes per step used to estimate mu^l = E[g^l]
MU_SEED          = 12345                       # dedicated RNG seed for the mu^l LF probe directions

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
                            c_probe_hf, c_probe_lf, mu_l):
    '''Assemble the high-fidelity-anchored control-variate fused one-point bandit
    gradient and update the per-turbine yaws (message-passing + projected-update
    tail of the Multi-Fidelity Algorithm).

    Each partial derivative term (own partial df_i/dgamma_i and cross partials
    df_j/dgamma_i for j in N_i) is fused independently, anchored on the HF gradient
    with an LF variance-reduction correction:

        g = g_hf + c * (g_lf - mu_l)

    before being assembled into the full gradient for turbine i. Every partial is a
    one-point bandit estimate formed directly from its probe cost (no base query),
    e.g. g_hf = f^h(gamma + delta*u_i, w_i) / delta * u_i. The coefficient
    ``cv_coefficient`` is a scalar shared by all partial terms (constant for testing).

    Parameters
    ----------
    agents       : list[TurbineAgent], one agent per turbine (owns gamma_i and box X_i).
    alpha        : float, stepsize alpha_k [deg^2].
    delta        : float, exploration delta_k [deg].
    xi           : float, shrinkage xi_k in (0,1); projection box is (1-xi)*X_i.
    us           : dict, {turb_idx: +1/-1} probe directions of the current cycle.
    cv_coefficient : float, control-variate coefficient c (constant for testing).
    c_probe_hf   : dict, {i: (N,) float} HF cost at (gamma_i + delta*u_i, gamma_{-i}), w_i.
    c_probe_lf   : dict, {i: (N,) float} LF cost at (gamma_i + delta*u_i), w_i (same u as HF).
    mu_l         : dict, {i: (N,) float} LF gradient mean E[g^l] partials at w_i, from a batch
                   of cheap LF probes.

    Returns
    -------
    gamma        : (N,) float, updated yaws gamma_{i,k+1}.
    grad         : (N,) float, assembled fused gradient estimate ghat_{i,k}.
    grad_hf      : (N,) float, assembled HF gradient (diagnostic).
    grad_lf      : (N,) float, assembled LF gradient at the HF probe direction (diagnostic).
    grad_lf_mean : (N,) float, assembled LF gradient mean mu^l (diagnostic).
    '''
    N = len(agents)

    grad         = np.zeros(N)
    grad_hf      = np.zeros(N)
    grad_lf      = np.zeros(N)
    grad_lf_mean = np.zeros(N)

    # ---- per-turbine fused partials + message passing ----
    for a in agents:
        i  = a.idx
        u  = us[i]

        # own partial: df_i/dgamma_i (probe i, cost i)
        gh   = c_probe_hf[i][i] / delta * u
        gl   = c_probe_lf[i][i] / delta * u
        mul  = mu_l[i][i]
        grad[i]         += gh + cv_coefficient * (gl - mul)
        grad_hf[i]      += gh
        grad_lf[i]      += gl
        grad_lf_mean[i] += mul

        # cross partials: df_j/dgamma_i for neighbor j -> sends to i (probe i, cost j)
        for j in a.neighbors:
            gh   = c_probe_hf[i][j] / delta * u
            gl   = c_probe_lf[i][j] / delta * u
            mul  = mu_l[i][j]
            grad[i]         += gh + cv_coefficient * (gl - mul)
            grad_hf[i]      += gh
            grad_lf[i]      += gl
            grad_lf_mean[i] += mul

    # ---- local projected update (each agent updates only itself) ----
    for a in agents:
        lo = (1.0 - xi) * a.x_lo                   # shrunken box keeps the next probe feasible
        hi = (1.0 - xi) * a.x_hi
        a.gamma = float(np.clip(a.gamma - alpha * grad[a.idx], lo, hi))  # projected gradient descent

    gamma = np.array([a.gamma for a in agents])
    return gamma, grad, grad_hf, grad_lf, grad_lf_mean


class MFBanditController:
    '''Stateful multi-fidelity one-point bandit controller.

    Mirrors ``HFBanditController``'s N-step probe cycle (one probe per turbine),
    but at every high-fidelity query instant also evaluates the low-fidelity model
    at the same time/wind/yaw with the same perturbation direction ``u``:

        step i  -> probe_i (turbine i at gamma_i + delta*u_i, wind w_i), i = 0..N-1

    At each probe step the LF cost f^l(gamma + delta*u_i, w_i) (same u as HF) and the
    LF gradient mean mu^l at w_i (from LF_MEAN_SAMPLES cheap LF probes with fresh
    Rademacher directions) are recorded. On cycle completion the fused gradient is
    assembled with the constant control-variate coefficient ``cv_coefficient``.
    '''
    def __init__(self, agents, SI, P_rate, alpha0, delta, xi, gamma_max, rng,
                 warm_start=True, cv_coefficient=1.0, lf_mean_samples=10, mu_seed=12345):
        self.agents      = agents
        self.SI          = SI
        self.P_rate      = P_rate
        self.alpha0      = alpha0
        self.delta       = delta
        self.xi          = xi
        self.gamma_max   = gamma_max
        self.rng         = rng
        self.warm_start  = warm_start
        self.N           = len(agents)
        self.k           = 0          # number of completed algorithm updates
        self.phase       = -1         # -1 -> nothing measured yet; i in 0..N-1 -> probe_i active
        self.us          = {}         # turb_idx -> +/-1 probe direction of current cycle

        # control-variate coefficient (constant for testing; no [0, 1] clamp when online)
        self.cv_coefficient = cv_coefficient

        # low-fidelity gradient mean mu^l = E[g^l] via cheap LF probes
        self.lf_mean_samples = lf_mean_samples
        self.rng_mu          = np.random.default_rng(mu_seed)   # independent of the HF probe RNG

        # high-fidelity probe costs (dynamic simulator)
        self.c_probe_hf  = {}
        # low-fidelity probe costs (quasi-static simulator), same u as the HF probe
        self.c_probe_lf  = {}
        # low-fidelity gradient means mu^l per partial (cheap LF batch)
        self.mu_l        = {}

        # logging of the last completed update
        self.last_grad         = np.full(self.N, np.nan)
        self.last_grad_hf      = np.full(self.N, np.nan)
        self.last_grad_lf      = np.full(self.N, np.nan)
        self.last_grad_lf_mean = np.full(self.N, np.nan)

    def _gamma(self):
        return np.array([a.gamma for a in self.agents])

    def _probe_yaw(self, i):
        '''Reconstruct the yaw vector that probed turbine i in the current cycle.'''
        gamma = self._gamma()
        gp = gamma.copy()
        gp[i] = np.clip(gamma[i] + self.delta * self.us[i], -self.gamma_max, self.gamma_max)
        return gp

    def _probe_yaw_signed(self, i, sign):
        '''Yaw vector perturbing turbine i by ``sign * delta`` (keeps others committed).'''
        gamma = self._gamma()
        gp = gamma.copy()
        gp[i] = np.clip(gamma[i] + sign * self.delta, -self.gamma_max, self.gamma_max)
        return gp

    def _issue_probe(self, i):
        '''Draw the probe direction for turbine i and return the corresponding yaw command.'''
        u = 1.0 if self.rng.uniform() < 0.5 else -1.0
        self.us[i] = u
        return self._probe_yaw(i)

    def _lf_cost(self, yaws, wind, layout):
        '''Evaluate the low-fidelity cost f^l(yaws; wind) = -P^l/P_rate, seeded from
        ``layout`` (the measured dynamic layout).'''
        ws, wd = wind
        self.SI.sim.set(wind_speeds=[ws], wind_directions=[wd])
        self.SI.take_step(np.asarray(yaws, dtype=float),
                          warm_start=self.warm_start, initial_layout=layout)
        return -self.SI.get_powers().squeeze() / self.P_rate

    def _lf_grad_mean(self, i, wind, layout):
        '''Estimate mu^l for probe i (all partials) at the given wind/layout from a
        batch of ``lf_mean_samples`` cheap LF one-point probes with fresh Rademacher
        directions (independent of the HF probe direction). Returns an (N,) vector of
        gradient-mean partials: mu^l[j] = E[(1/delta) f^l(gamma + delta*u_i e_i)[j] * u_i].
        '''
        acc = np.zeros(self.N)
        for _ in range(self.lf_mean_samples):
            u = 1.0 if self.rng_mu.uniform() < 0.5 else -1.0
            cost = self._lf_cost(self._probe_yaw_signed(i, u), wind, layout)
            acc += cost / self.delta * u
        return acc / self.lf_mean_samples

    def record(self, powers, wind, layout):
        '''Store the cost of the just-completed high-fidelity step and evaluate the
        low-fidelity model at the same instant (same time/wind/yaw/direction).'''
        if self.phase < 0:
            return                                  # initial default-command measurement is discarded
        i = self.phase
        self.c_probe_hf[i] = -np.asarray(powers, dtype=float) / self.P_rate
        # LF cost at the same yaw as the HF probe (same u, same w_i)
        self.c_probe_lf[i] = self._lf_cost(self._probe_yaw(i), wind, layout)
        # LF gradient mean mu^l at the same w_i (batch of cheap LF probes)
        self.mu_l[i] = self._lf_grad_mean(i, wind, layout)

    def next_command(self):
        '''Return the yaw command for the next step; every N steps also returns the update.'''
        if self.phase < 0:
            # first call -> start the cycle with probe 0
            self.phase = 0
            return self._issue_probe(0), None

        if self.phase < self.N - 1:
            # probe (phase) just measured -> issue probe (phase+1)
            self.phase += 1
            return self._issue_probe(self.phase), None

        # last probe just measured -> assemble fused gradient and update, then start next cycle
        alpha = self.alpha0 / np.sqrt(self.k + 1.0)
        (gamma_new, grad, grad_hf, grad_lf, grad_lf_mean) = distributed_doo_step_mf(
            self.agents, alpha, self.delta, self.xi, self.us, self.cv_coefficient,
            self.c_probe_hf, self.c_probe_lf, self.mu_l,
        )
        self.last_grad         = grad
        self.last_grad_hf      = grad_hf
        self.last_grad_lf      = grad_lf
        self.last_grad_lf_mean = grad_lf_mean
        self.k += 1
        self.us         = {}
        self.c_probe_hf = {}
        self.c_probe_lf = {}
        self.mu_l       = {}
        self.phase      = 0
        return self._issue_probe(0), (gamma_new, grad, None)


# --- Multi-fidelity bandit controller (stateful; drives the joint HF/LF probe cycle) ---
mf_ctrl = MFBanditController(AGENTS, SI, P_rate, ALPHA0, DELTA, XI, GAMMA_MAX, rng,
                             warm_start=WARM_START,
                             cv_coefficient=CV_COEFF,
                             lf_mean_samples=LF_MEAN_SAMPLES, mu_seed=MU_SEED)

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

    # 3. Joint multi-fidelity one-point bandit update. The wind and dynamic layout
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
        step_result[f"ctrl_yaw_cmd_T{i}"] = float(yaw_cmd[i])        # probe command
        step_result[f"gamma_T{i}"]        = float(AGENTS[i].gamma)   # committed yaw anchor

    # grad / coefficient reflect the most recent completed update (NaN until the first)
    step_result["grad"]           = mf_ctrl.last_grad.copy()
    step_result["grad_hf"]        = mf_ctrl.last_grad_hf.copy()
    step_result["grad_lf"]        = mf_ctrl.last_grad_lf.copy()
    step_result["grad_lf_mean"]   = mf_ctrl.last_grad_lf_mean.copy()
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

    # 3. Cost: realized dynamic cost
    cost_realized = -farm_power * 1e6 / P_rate           # f_t = -P_real/P_rate
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(t_arr, cost_realized,
             label=r"$f_t=-P_{\mathrm{real}}/P_{\mathrm{rate}}$", lw=2)
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

    # 5. Gradient components (fused / HF anchor / LF probe / LF mean) per turbine
    grad_hf_arr      = np.array([r["grad_hf"] for r in results])
    grad_lf_arr      = np.array([r["grad_lf"] for r in results])
    grad_lf_mean_arr = np.array([r["grad_lf_mean"] for r in results])
    grad_fused_arr   = np.array([r["grad"] for r in results])
    fig5, axes5 = plt.subplots(n_turbines, 1, sharex=True, figsize=(8, 3 * n_turbines))
    axes5 = np.atleast_1d(axes5)
    for i in range(n_turbines):
        ax = axes5[i]
        ax.plot(t_arr, grad_fused_arr[:, i],   label="Fused",   lw=2)
        ax.plot(t_arr, grad_hf_arr[:, i],      label="HF anchor", ls="-")
        ax.plot(t_arr, grad_lf_arr[:, i],      label="LF probe",  ls="--")
        ax.plot(t_arr, grad_lf_mean_arr[:, i], label="LF mean",   ls=":")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(f"T{i} (1/deg)")
        ax.legend(); ax.grid(True)
    axes5[-1].set_xlabel("Time (s)")
    fig5.suptitle("Gradient components per turbine")
    fig5.tight_layout()
    _savefig(fig5, "gradient_components")
