# Run with: $env:OMP_NUM_THREADS = 8
#           mpiexec -n 1 python dynamic_loop_doo_hf.py
'''
Implement Algorithm 1 from overleaf/main.tex (distributed online optimization with bandit feedback),
high-fidelity variant: the dynamic FAST.Farm simulator is probed directly for the one-point
bandit gradient estimate (no committed-point query).
'''
import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from ffconnect.interface import FastFarmInterface

YAW_BOUND    = 30.0                       # yaw limit (deg) — a control bound, not a dynamic-case value
P_RATED_PER_TURBINE = 5.0e6               # rated power per turbine [W] (NREL 5-MW baseline)


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
result_file_name = "hf_opbe-1x3-steadyWind.pkl"
# --- User-defined paths ---
FFexePath = "D:\\2_PhD_UBC\\Code\\FASTv355\\FAST.Farm_x64_OMP_v355.exe"                              # parallel executable (with SC_DLL_Full variant)
FFsimPath = "D:\\2_PhD_UBC\\Code\\DOO-FOWF\\dynamic_CaseCtrl\\1x3.fstf"  
lift_constraint = True                              # To see whether nonconvexity influence convergence

# --- Initialize interface from .fstf file ---
interface = FastFarmInterface.from_file(FFsimPath, fast_farm_executable=FFexePath, per_turbine_flags=True)  # dynamic FAST.Farm handle (MPI super-controller)
interface.init()

n_turbines = interface.num_turbines           # number of turbines (read from the .fstf)
max_iter   = interface.max_iter               # number of timesteps (1000 s / 5 s = 200)
dt         = interface.dt                     # dynamic timestep [s] (DT_Low = 5.0)

P_rate = P_RATED_PER_TURBINE * n_turbines     # rated farm power [W], normalizes the cost f = -P/P_rate

# --- Algorithm 1 parameters (N-turbine row, streamwise ordering) ---
GAMMA_MAX = YAW_BOUND                          # 30 deg
DELTA     = 2.0                                # exploration delta_t (deg)
XI        = 0.1                                # shrinkage xi_t -> effective bound 27 deg
ALPHA0    = 10.0                               # base stepsize alpha_0 (deg^2) -- TUNE ME (one-point bandit)

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
run_name = os.path.splitext(result_file_name)[0]        # e.g. "warm-start-hftpde-1x2-steadyWind"
run_dir  = os.path.join(DATA_DIR, run_name)             # e.g. "dynamic_Data/warm-start-hftpde-1x2-steadyWind"
os.makedirs(run_dir, exist_ok=True)


def distributed_doo_step_hf_one_point_bandit(agents, alpha, delta, xi, us, c_probe):
    '''Assemble the one-point bandit gradient from high-fidelity cost samples and
    update the per-turbine yaws.

    This is the message-passing + projected-update tail of Algorithm 1: each agent
    estimates its own partial derivative plus the cross-partials reported by its
    neighbors, then takes a projected gradient step. The probe cost vectors
    ``c_probe`` come from the high-fidelity dynamic simulator (collected over ``N``
    consecutive ``interface.step()`` calls by ``HFBanditController``), so this function
    never touches a simulator.

    The one-point estimator never queries the committed point; each partial is formed
    directly from the probe cost,
        g_i = (1/delta) * f_i(gamma + delta*u_i) * u_i,
    so only ``N`` high-fidelity evaluations are needed per update.

    Parameters
    ----------
    agents  : list[TurbineAgent], one agent per turbine (owns gamma_i and box X_i).
    alpha   : float, stepsize alpha_k [deg^2].
    delta   : float, exploration delta_k [deg].
    xi      : float, shrinkage xi_k in (0,1); projection box is (1-xi)*X_i.
    us      : dict, {turb_idx: +1/-1} probe directions for the current cycle.
    c_probe : dict, {turb_idx: (N,) float} cost at (gamma_i + delta*u_i, gamma_{-i}).

    Returns
    -------
    gamma  : (N,) float, updated yaws gamma_{i,k+1}.
    grad   : (N,) float, assembled bandit gradient estimate ghat_{i,k}.
    '''
    N = len(agents)

    # ---- message passing: own partial + cross-partials from neighbors ----
    grad = np.zeros(N)
    for a in agents:
        i = a.idx
        g = c_probe[i][i] / delta * us[i]                     # own partial: df_i/dgamma_i
        for j in a.neighbors:                                 # neighbor j -> sends to i
            g += c_probe[i][j] / delta * us[i]                # cross partial: df_j/dgamma_i
        grad[i] = g

    # ---- local projected update (each agent updates only itself) ----
    for a in agents:
        lo = (1.0 - xi) * a.x_lo                   # shrunken box keeps the next probe feasible
        hi = (1.0 - xi) * a.x_hi
        a.gamma = float(np.clip(a.gamma - alpha * grad[a.idx], lo, hi))  # projected gradient descent

    return np.array([a.gamma for a in agents]), grad


class HFBanditController:
    '''Stateful high-fidelity one-point bandit controller.

    Because the dynamic interface exposes only one power measurement per
    ``interface.step()`` (the measurement belongs to the yaw command sent one step
    earlier), a single bandit update spans ``N`` consecutive steps (one probe per
    turbine); the one-point estimator never queries the committed point:

        step i -> probe_i (only turbine i at gamma_i + delta*u_i), i = 0..N-1

    The simulation loop calls ``record(powers)`` right after stepping (storing the
    freshly measured cost), then ``next_command()`` (returning the yaw command for the
    *next* step, and, every ``N`` steps, the completed update).
    '''
    def __init__(self, agents, P_rate, alpha0, delta, xi, gamma_max, rng):
        self.agents    = agents
        self.P_rate    = P_rate
        self.alpha0    = alpha0
        self.delta     = delta
        self.xi        = xi
        self.gamma_max = gamma_max
        self.rng       = rng
        self.N         = len(agents)
        self.k         = 0          # number of completed algorithm updates
        self.phase     = -1         # -1 -> nothing measured yet; i in 0..N-1 -> probe_i active
        self.us        = {}         # turb_idx -> +/-1 probe direction of current cycle
        self.c_probe   = {}
        self.last_grad    = np.full(self.N, np.nan)   # last completed update's gradient

    def _gamma(self):
        return np.array([a.gamma for a in self.agents])

    def _probe_yaw(self, i):
        '''Yaw vector that probes turbine i alone, relative to the committed gamma.'''
        gamma = self._gamma()
        gp = gamma.copy()
        gp[i] = np.clip(gamma[i] + self.delta * self.us[i], -self.gamma_max, self.gamma_max)
        return gp

    def _issue_probe(self, i):
        '''Draw the probe direction for turbine i and return the corresponding yaw command.'''
        u = 1.0 if self.rng.uniform() < 0.5 else -1.0
        self.us[i] = u
        return self._probe_yaw(i)

    def record(self, powers):
        '''Store the cost of the just-completed step under the currently active probe.'''
        if self.phase < 0:
            return                                  # initial default-command measurement is discarded
        self.c_probe[self.phase] = -np.asarray(powers, dtype=float) / self.P_rate

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

        # last probe just measured -> assemble gradient and update, then start next cycle
        alpha = self.alpha0 / np.sqrt(self.k + 1.0)
        gamma_new, grad = distributed_doo_step_hf_one_point_bandit(
            self.agents, alpha, self.delta, self.xi, self.us, self.c_probe,
        )
        self.last_grad = grad
        self.k += 1
        self.us      = {}
        self.c_probe = {}
        self.phase   = 0
        return self._issue_probe(0), (gamma_new, grad)

# --- High-fidelity bandit controller (stateful; drives the probe cycle across steps) ---
hf_ctrl = HFBanditController(AGENTS, P_rate, ALPHA0, DELTA, XI, GAMMA_MAX, rng)

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

    # 3. One-point bandit update (high-fidelity): the dynamic simulator is probed
    #    directly. Each step's measured power belongs to the yaw command sent one step
    #    earlier (a probe); HFBanditController pieces the N probe samples into one
    #    bandit update. gamma is the committed yaw and grad the gradient estimate.
    t0 = time.perf_counter()
    powers = np.array([step_result[f"power_T{i}"] for i in range(n_turbines)])
    hf_ctrl.record(powers)
    yaw_cmd, _ = hf_ctrl.next_command()

    # 4. Send commands (buffered, consumed by the NEXT step())
    for i in range(n_turbines):
        interface.set_command(yaw=float(yaw_cmd[i]), ipc=None, torque=None, turb_idx=i)
        step_result[f"ctrl_yaw_cmd_T{i}"] = float(yaw_cmd[i])        # probe command
        step_result[f"gamma_T{i}"]        = float(AGENTS[i].gamma)   # committed yaw anchor

    # grad reflects the most recent completed update (NaN until the first)
    step_result["grad"]      = hf_ctrl.last_grad.copy()
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

    # 4. Gradient estimate per turbine
    grad_arr = np.array([r["grad"] for r in results])    # (T, N)
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    for i in range(n_turbines):
        ax4.plot(t_arr, grad_arr[:, i], label=f"T{i}")
    ax4.axhline(0, color="k", lw=0.5)
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Gradient (1/deg)")
    ax4.set_title("Estimated gradient")
    ax4.legend(); ax4.grid(True); fig4.tight_layout()
    _savefig(fig4, "gradient")
