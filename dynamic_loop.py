# Run with: $env:OMP_NUM_THREADS = 8
#           mpiexec -n 1 python dynamic_loop.py

import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from ffconnect.interface import FastFarmInterface
from ffconnect.simul_utils import get_inflow_file_path
from openfast_toolbox.io.fast_input_file import FASTInputFile
from dynamic_wind_profile import read_wind_profile, wind_at_time

# ================================= Initialization =================================
save_results = True
save_figure_option = True    # True → save all figures to dynamic_Data/<run_name>/  |  False → interactive plt.show()
print_result = False
visualization = True
DATA_DIR = "dynamic_Data"                            # all .pkl / figures live here
result_file_name = "no-control-1x3-steadyWind.pkl"
# --- User-defined paths ---
FFexePath = "D:\\2_PhD_UBC\\Code\\FASTv355\\FAST.Farm_x64_OMP_v355.exe"                              # parallel executable (with SC_DLL_Full variant)
FFsimPath = "D:\\2_PhD_UBC\\Code\\DOO-FOWF\\dynamic_CaseCtrl\\1x3.fstf"  

# --- Initialize interface from .fstf file ---
interface = FastFarmInterface.from_file(FFsimPath, fast_farm_executable=FFexePath, per_turbine_flags=True)
interface.init()

n_turbines = interface.num_turbines
max_iter   = interface.max_iter
dt         = interface.dt

# --- Ambient wind profile (steady or time-varying, from the InflowWind file) ---
inflow_path  = get_inflow_file_path(FFsimPath)
inflow       = FASTInputFile(inflow_path)
WIND_PROFILE = read_wind_profile(inflow, inflow_path)

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
    ws_t, wd_t = wind_at_time(WIND_PROFILE, (t + 1) * dt)
    step_result["wind_speed_SI"]     = ws_t
    step_result["wind_direction_SI"] = wd_t
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
        step_result[f"yaw_T{i}"]            = np.degrees(yaw_i.item())
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

    # 2. Store the computed total farm power
    step_result["total_power"] = total_power
    step_result["ctrl_compute_time"] = 0.0    # no control computation in this loop

    results.append(step_result)

    if print_result:
        print(f"t={(t+1)*dt:6.1f}s  power={total_power/1e6:.3f} MW")

    if done:
        break

print(f"Total wall-clock: {time.perf_counter() - run_start:.3f}s")

# ================================= Output directory =================================
run_name = os.path.splitext(result_file_name)[0]        # e.g. "Myaw-none0"
run_dir  = os.path.join(DATA_DIR, run_name)             # e.g. "dynamic_Data/Myaw-none0"
os.makedirs(run_dir, exist_ok=True)

# ================================= Save Results =================================
if save_results:
    out_path = os.path.join(run_dir, result_file_name)
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {out_path} ({len(results)} steps)")

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

    # 2. Yaw: measured per turbine (no control command)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    for i in range(n_turbines):
        ax2.plot(t_arr, [r[f"yaw_T{i}"] for r in results],
                 label=f"T{i} meas", ls="--")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Yaw (deg)")
    ax2.set_title("Measured yaw (no control)")
    ax2.legend(); ax2.grid(True); fig2.tight_layout()
    _savefig(fig2, "yaw")
