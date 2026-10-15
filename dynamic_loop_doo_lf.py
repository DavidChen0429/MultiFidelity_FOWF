# Run with: $env:OMP_NUM_THREADS = 8
#           mpiexec -n 1 python dynamic_loop_doo_lf.py
'''
Implement low-fidelity methods in overleaf/3Method.tex (distributed online optimization with bandit feedback)
'''
import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from ffconnect.interface import FastFarmInterface
from pathlib import Path
from floris import FlorisModel
from static_simulation_interface import FLORISFarmSim
from ffconnect.simul_utils import get_inflow_file_path
from openfast_toolbox.io.fast_input_file import FASTInputFile
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
save_snapshot_option = False   # True -> save FLORIS wake-image snapshots every SNAPSHOT_INTERVAL s
SNAPSHOT_INTERVAL = 50.0       # seconds between snapshots
DATA_DIR = "dynamic_Data"                            # all .pkl / figures live here
result_file_name = "debug-1x2-steadyWind.pkl"
# --- User-defined paths ---
FFexePath = "D:\\2_PhD_UBC\\Code\\FASTv355\\FAST.Farm_x64_OMP_v355.exe"                              # parallel executable (with SC_DLL_Full variant)
FFsimPath = "D:\\2_PhD_UBC\\Code\\DOO-FOWF\\dynamic_CaseCtrl\\1x2-debug.fstf"  
lift_constraint = True                              # To see whether nonconvexity influence convergence

# --- Initialize interface from .fstf file ---
interface = FastFarmInterface.from_file(FFsimPath, fast_farm_executable=FFexePath, per_turbine_flags=True)  # dynamic FAST.Farm handle (MPI super-controller)
interface.init()

n_turbines = interface.num_turbines           # number of turbines (read from the .fstf)
max_iter   = interface.max_iter               # number of timesteps (300 s / 3 s = 100)
dt         = interface.dt                     # dynamic timestep [s] (DT_Low = 3.0)

# --- Quasi-static gradient oracle (built from the dynamic case) ---
SI = build_quasi_static(FFsimPath)
P_rate = rated_power(SI)

# --- Algorithm 1 parameters (N-turbine row, streamwise ordering) ---
GAMMA_MAX = YAW_BOUND                          # 30 deg
DELTA     = 2.0                                # exploration delta_t (deg)
XI        = 0.1                                # shrinkage xi_t -> effective bound 27 deg
ALPHA0    = 10.0                               # base stepsize alpha_0 (deg^2) -- TUNE ME (one-point bandit)
WARM_START = True                              # True -> warm-start quasi-static solver from baseline layout

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


def save_floris_snapshot(SI, t_seconds, snapshot_dir):
    '''Save a PNG image of the FLORIS waked flow at the committed yaw.'''
    fig, ax = plt.subplots(dpi=120, figsize=(8, 5), layout="constrained")
    ax.set_axis_off()
    SI.get_wake_snapshot(title=f"t = {t_seconds:.0f} s", ax=ax)
    path = os.path.join(snapshot_dir, f"snapshot_t{t_seconds:03.0f}s.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved FLORIS snapshot: {path}")


# ================================= Output directory =================================
run_name = os.path.splitext(result_file_name)[0]        # e.g. "warm-start-cde-1x2-timeVaryingWind"
run_dir  = os.path.join(DATA_DIR, run_name)             # e.g. "dynamic_Data/warm-start-cde-1x2-timeVaryingWind"
os.makedirs(run_dir, exist_ok=True)
snapshot_dir = os.path.join(run_dir, "quasi-static-snapshot")
os.makedirs(snapshot_dir, exist_ok=True)


def distributed_doo_step_lf(agents, SI, P_rate, alpha, delta, xi, gamma_max, rng, warm_start=True):
    ''' One-point bandit gradient estimator:
            g_i = (1/delta) * f_i(gamma + delta*u_i) * u_i,

    Parameters
    ----------
    agents    : list[TurbineAgent], one agent per turbine (owns gamma_i and box X_i).
    SI        : FLORISFarmSim, quasi-static simulator used as the (whole-farm) cost oracle.
    P_rate    : float, rated farm power [W]; cost is f = -P/P_rate.
    alpha     : float, stepsize alpha_t [deg^2].
    delta     : float, exploration delta_t [deg].
    xi        : float, shrinkage xi_t in (0,1); projection box is (1-xi)*X_i.
    gamma_max : float, physical yaw bound [deg] (clips probes only).
    rng       : numpy.Generator, seeded RNG for the +/-1 probe directions.
    warm_start: bool, warm-start the quasi-static solver from the baseline layout.

    Returns
    -------
    gamma     : (N,) float, updated yaws gamma_{i,t+1}.
    grad      : (N,) float, assembled one-point bandit gradient estimate ghat_{i,t}.
    qs_cost   : None, the one-point estimator never evaluates the committed point.
    '''
    def cost(yaws, initial_layout=None):
        # f(yaws) = -P(yaws)/P_rate, shape (N,) -- oracle computes the whole farm
        SI.take_step(np.asarray(yaws, dtype=float),
                     warm_start=warm_start, initial_layout=initial_layout)
        return -SI.get_powers().squeeze() / P_rate

    N = len(agents)
    gamma = np.array([a.gamma for a in agents])     # gather committed yaws (for the oracle only)

    base_layout = SI.warm_layout.copy() if warm_start else None   # seed layout (measured reality)

    # ---- sequential probe phase (one probe per turbine, no base query) ----
    c_probe = {}                                    # i -> cost vector at (gamma_i + delta*u_i, gamma_{-i})
    us      = {}                                    # probe direction
    for a in agents:                                # Sequentially probe all turbines
        u = 1.0 if rng.uniform() < 0.5 else -1.0    # Rademacher probe direction (+1 / -1)
        gp = gamma.copy()
        gp[a.idx] = np.clip(a.gamma + delta * u, -gamma_max, gamma_max)  # perturb only turbine i
        c_probe[a.idx] = cost(gp, initial_layout=base_layout)            # cost of the entire FOWF         
        us[a.idx] = u

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

    return np.array([a.gamma for a in agents]), grad, None


# --- Simulation loop ---
results = []
done = False
run_start = time.perf_counter()
next_snapshot_time = SNAPSHOT_INTERVAL

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

    # 3. Derive optimal yaw (future yaw at t+1) based on Algorithm 1
    #    gamma  = new committed yaws gamma_{i,t+1},
    #    grad   = bandit gradient estimate ghat_{i,t},
    #    qs_cost = cost vector f_k(gamma) before the update (None for one-point).
    #    Set the quasi-static oracle wind to the current simulation time first.
    ws_t, wd_t = wind_at_time(SI.wind_profile, (t + 1) * dt)
    SI.sim.set(wind_speeds=[ws_t], wind_directions=[wd_t])
    step_result["wind_speed_SI"]     = ws_t
    step_result["wind_direction_SI"] = wd_t

    # Seed the gradient probes from the actual measured farm layout (reality),
    # not from the quasi-static simulator's own converged layout.
    surge = np.array([step_result[f"platform_tdx_T{i}"] for i in range(n_turbines)])
    sway  = np.array([step_result[f"platform_tdy_T{i}"] for i in range(n_turbines)])
    dynamic_layout = np.vstack([
        SI.neutral_positions[0] + surge,
        SI.neutral_positions[1] + sway,
    ])
    SI.warm_layout = dynamic_layout

    # Record the seed actually handed to the quasi-static simulator (displacement form)
    for i in range(n_turbines):
        step_result[f"qs_probe_start_tdx_T{i}"] = float(SI.warm_layout[0, i] - SI.neutral_positions[0, i])
        step_result[f"qs_probe_start_tdy_T{i}"] = float(SI.warm_layout[1, i] - SI.neutral_positions[1, i])

    alpha_t = ALPHA0 / np.sqrt(t + 1.0)        # decaying stepsize (t 0-indexed)
    t0 = time.perf_counter()
    gamma, grad, qs_cost = distributed_doo_step_lf(
        AGENTS, SI, P_rate, alpha_t, DELTA, XI, GAMMA_MAX, rng, WARM_START,
    )
    for i in range(n_turbines):
        step_result[f"ctrl_yaw_cmd_T{i}"] = float(gamma[i])
        step_result[f"gamma_T{i}"]        = float(gamma[i])
    step_result["grad"]    = grad.copy()
    step_result["qs_cost"] = None if qs_cost is None else qs_cost.copy()
    step_result["ctrl_compute_time"] = time.perf_counter() - t0

    # 4. Send commands (buffered, consumed by the NEXT step())
    for i in range(n_turbines):
        cmd  = step_result[f"ctrl_yaw_cmd_T{i}"]
        interface.set_command(
            yaw=cmd,
            ipc=None, torque=None, turb_idx=i,
        )

    # 5. Periodically save a FLORIS wake snapshot at the committed yaw
    if save_snapshot_option and step_result["t"] >= next_snapshot_time:
        SI.take_step(np.asarray(gamma, dtype=float), warm_start=True, initial_layout=dynamic_layout)
        save_floris_snapshot(SI, step_result["t"], snapshot_dir)
        next_snapshot_time += SNAPSHOT_INTERVAL

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

    # 3. Cost: realized dynamic + quasi-static oracle
    cost_realized = -farm_power * 1e6 / P_rate           # f_t = -P_real/P_rate
    has_qs        = all(r["qs_cost"] is not None for r in results)  # one-point estimator has no base query
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(t_arr, cost_realized,
             label=r"$f_t=-P_{\mathrm{real}}/P_{\mathrm{rate}}$", lw=2)
    if has_qs:
        cost_qs = np.array([r["qs_cost"].sum() for r in results])
        ax3.plot(t_arr, cost_qs, label="quasi-static", ls="--")
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
