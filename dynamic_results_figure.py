import os
import pickle

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "figure.titlesize": 20,
})

DATA_DIR = "dynamic_Data"

CASES = {
    "baseline": ("no-control-1x3-na-timeVaryingWind", "no-control", "tab:blue"),
    "cde": ("lf_warm-start-cde-1x3-na-timeVaryingWind", "central diff.", "tab:orange"),
    "tpbe": ("lf_warm-start-tpbe-1x3-na-timeVaryingWind", "two-point bandit", "tab:green"),
}

CASE_LS = {"no-control": "--", "central diff.": "-", "two-point bandit": "-."}
TURB_COLOR = {i: f"C{i}" for i in range(10)}
LW = 3


def n_turbines(data):
    return sum(1 for k in data[0] if k.startswith("power_T"))


def load_case(name):
    folder, _, _ = CASES[name]
    path = os.path.join(DATA_DIR, folder, f"{folder}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def time_axis(data):
    return np.array([r["t"] for r in data])


def farm_power(data):
    n = n_turbines(data)
    return np.array([sum(r[f"power_T{i}"] for i in range(n)) for r in data])


def turbine_power(data, i):
    return np.array([r[f"power_T{i}"] for r in data])


def yaw(data, i):
    return np.array([r[f"yaw_T{i}"] for r in data])


def series(data, key, i):
    return np.array([r[f"{key}_T{i}"] for r in data])


def yaw_cmd(data, i):
    key = f"ctrl_yaw_cmd_T{i}"
    if key in data[0]:
        return np.array([r[key] for r in data])
    return np.zeros(len(data))


def wind_speed(data):
    return np.array([r["wind_speed_SI"] for r in data])


def wind_direction(data):
    return np.array([r["wind_direction_SI"] for r in data])


def ctrl_time(data):
    return np.array([r["ctrl_compute_time"] for r in data])


def add_wind(ax, data, t):
    ax_d = ax.twinx()
    ax.plot(t, wind_speed(data), color="tab:blue", lw=LW, label="wind speed")
    ax_d.plot(t, wind_direction(data), color="tab:red", lw=LW, label="wind direction")
    ax.set_ylabel("Wind speed (m/s)")
    ax_d.set_ylabel("Wind direction (deg)")
    ax.grid(True)
    ax_d.grid(False)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_d.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="best")


def main():
    data = {name: load_case(name) for name in CASES}
    t = time_axis(data["baseline"])

    powers = {name: farm_power(d) for name, d in data.items()}
    means = {name: p.mean() / 1e6 for name, p in powers.items()}
    energies = {name: np.trapezoid(p, t) / 3.6e9 for name, p in powers.items()}

    print("Power/energy summary:")
    for name in CASES:
        label = CASES[name][1]
        print(f"  {label:15s}: mean {means[name]:7.3f} MW, energy {energies[name]:7.2f} MWh")
    print("  energy gain vs no-control:")
    for name in CASES:
        if name == "baseline":
            continue
        gain = 100.0 * (energies[name] - energies["baseline"]) / energies["baseline"]
        print(f"    {CASES[name][1]:15s}: {gain:+7.2f} %")

    def add_turbine(ax, getter, i, *args):
        for name, (_, label, _) in CASES.items():
            ax.plot(t, getter(data[name], i, *args), color=TURB_COLOR[i], lw=LW,
                    ls=CASE_LS[label], label=f"{label} T{i}")
        return ax

    # 1. Farm power + wind condition
    fig1, (a1, a1w) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
    for name, (_, label, color) in CASES.items():
        a1.plot(t, powers[name] / 1e6, color=color, lw=LW, ls=CASE_LS[label], label=label)
    a1.set_ylabel("Farm power (MW)")
    a1.set_title("Farm power comparison")
    a1.legend()
    a1.grid(True)

    add_wind(a1w, data["baseline"], t)
    a1w.set_xlabel("Time (s)")
    fig1.tight_layout()
    plt.show()

    # 2. Individual turbine power + wind condition
    fig2, (a2, a2w) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
    for i in range(n_turbines(data["baseline"])):
        add_turbine(a2, turbine_power, i)
    a2.set_ylabel("Turbine power (W)")
    a2.set_title("Individual turbine power")
    a2.legend()
    a2.grid(True)

    add_wind(a2w, data["baseline"], t)
    a2w.set_xlabel("Time (s)")
    fig2.tight_layout()
    plt.show()

    # 3. Yaw command (per turbine) + wind condition
    n = n_turbines(data["baseline"])
    fig3, axes3 = plt.subplots(n + 1, 1, sharex=True, figsize=(9, 4 * (n + 1)))
    for i in range(n):
        add_turbine(axes3[i], yaw_cmd, i)
        axes3[i].axhline(0, color="k", lw=0.5)
        axes3[i].set_ylabel(f"Yaw command T{i} (deg)")
        axes3[i].set_title(f"Yaw command T{i}")
        axes3[i].legend()
        axes3[i].grid(True)

    add_wind(axes3[-1], data["baseline"], t)
    axes3[-1].set_xlabel("Time (s)")
    axes3[-1].set_title("Wind condition")

    fig3.tight_layout()
    plt.show()

    # 4. Control compute time
    fig4, ax = plt.subplots(figsize=(9, 4))
    for name, (_, label, color) in CASES.items():
        ax.plot(t, ctrl_time(data[name]), color=color, lw=LW, ls=CASE_LS[label], label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Per-step control compute time (s)")
    ax.set_title("Control compute time comparison")
    ax.legend()
    ax.grid(True)
    fig4.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
