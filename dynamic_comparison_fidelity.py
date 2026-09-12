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
config = "1x3-na"                 # farm configuration: "1x2", "1x3", "1x3-na", ...
windCond = "steadyWind"        # steadyWind or timeVaryingWind
estimate_method = "tpbe"       # tpbe (two-point bandit) or cde (central difference)

CASE_STYLE = {
    "greedy": ("greedy", "tab:gray"),
    "lf": ("low-fidelity", "tab:blue"),
    "hf": ("high-fidelity", "tab:red"),
}


def case_folder(case):
    """Build the dynamic_Data folder name from the naming rule:
    (fidelity)_(control method)-(gradient method)-(config)-(wind condition)."""
    if case == "greedy":
        return f"no-control-{config}-{windCond}"
    if case == "lf":
        return f"lf_warm-start-{estimate_method}-{config}-{windCond}"
    if case == "hf":
        return f"hf_{estimate_method}-{config}-{windCond}"
    raise ValueError(f"Unknown case: {case}")


CASES = {
    case: (case_folder(case), label, color)
    for case, (label, color) in CASE_STYLE.items()
}

CASE_LS = {"greedy": ":", "low-fidelity": "-", "high-fidelity": "--"}
TURB_COLOR = {i: f"C{i}" for i in range(10)}
LW = 1
N_TURB = 3

P_RATE = N_TURB * 5.0e6  # common farm rated power for realized-cost normalization (3 x 5 MW)


def n_turbines(data):
    return sum(1 for k in data[0] if k.startswith("power_T"))


def load_case(name):
    folder, _, _ = CASES[name]
    path = os.path.join(DATA_DIR, folder, f"{folder}.pkl")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No data file for case '{name}'.\n  Expected: {path}\n"
            f"  Check config={config!r}, windCond={windCond!r}, "
            f"estimate_method={estimate_method!r}."
        )
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


def yaw_cmd(data, i):
    key = f"ctrl_yaw_cmd_T{i}"
    if key in data[0]:
        return np.array([r[key] for r in data])
    return np.zeros(len(data))


def wind_speed(data):
    if "wind_speed_SI" in data[0]:
        return np.array([r["wind_speed_SI"] for r in data])
    return np.array([r["wind_speed_T0"] for r in data])


def wind_direction(data):
    if "wind_direction_SI" in data[0]:
        return np.array([r["wind_direction_SI"] for r in data])
    return np.array([r["wind_direction_T0"] for r in data])


def realized_cost(data):
    return -farm_power(data) / P_RATE


def oracle_cost(data):
    if "qs_cost" in data[0]:
        return np.array([r["qs_cost"].sum() for r in data])
    if "cost_base" in data[0]:
        return np.array([r["cost_base"].sum() for r in data])
    return None


def gradient(data, i):
    if "grad" not in data[0]:
        return np.zeros(len(data))
    return np.array([r["grad"][i] for r in data])


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
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower left")


def truncate_to_common(data):
    """Keep only the overlapping time window shared by all cases."""
    t_end = min(time_axis(d)[-1] for d in data.values())
    return {name: [r for r in d if r["t"] <= t_end] for name, d in data.items()}


def main():
    data = {name: load_case(name) for name in CASES}
    if windCond == "steadyWind":
        data = truncate_to_common(data)
        t_common = time_axis(data["greedy"])[-1]
        print(f"steadyWind: truncated all cases to a common {t_common:.0f} s window "
              f"({len(data['greedy'])} steps)")
    t = time_axis(data["greedy"])

    powers = {name: farm_power(d) for name, d in data.items()}
    means = {name: p.mean() / 1e6 for name, p in powers.items()}
    energies = {name: np.trapezoid(p, t) / 3.6e9 for name, p in powers.items()}

    print("Power/energy summary:")
    for name in CASES:
        label = CASES[name][1]
        print(f"  {label:20s}: mean {means[name]:7.3f} MW, energy {energies[name]:7.2f} MWh")
    print("  energy gain vs greedy (no control):")
    for name in CASES:
        if name == "greedy":
            continue
        gain = 100.0 * (energies[name] - energies["greedy"]) / energies["greedy"]
        print(f"    {CASES[name][1]:20s}: {gain:+7.2f} %")

    def add_turbine(ax, getter, i, *args):
        for name, (_, label, _) in CASES.items():
            ax.plot(t, getter(data[name], i, *args), color=TURB_COLOR[i], lw=LW,
                    ls=CASE_LS[label], label=label)
        return ax

    # 1. Farm power + wind condition
    fig1, (a1, a1w) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
    for name, (_, label, color) in CASES.items():
        a1.plot(t, powers[name] / 1e6, color=color, lw=LW, ls=CASE_LS[label], label=label)
    a1.set_ylabel("Farm power (MW)")
    a1.set_title("Farm power comparison")
    a1.legend(loc="lower left")
    a1.grid(True)

    add_wind(a1w, data["greedy"], t)
    a1w.set_xlabel("Time (s)")
    fig1.tight_layout()
    plt.show()

    # 2. Yaw command (per turbine)
    n = n_turbines(data["greedy"])
    fig3, axes3 = plt.subplots(n, 1, sharex=True, figsize=(9, 4 * n))
    for i in range(n):
        add_turbine(axes3[i], yaw_cmd, i)
        axes3[i].axhline(0, color="k", lw=0.5)
        axes3[i].set_ylabel(f"Yaw command T{i} (deg)")
        axes3[i].set_title(f"Yaw command T{i}")
        axes3[i].legend(loc="lower left")
        axes3[i].grid(True)

    axes3[-1].set_xlabel("Time (s)")

    fig3.tight_layout()
    plt.show()

    # 3. Control compute time
    fig4, ax = plt.subplots(figsize=(9, 4))
    for name, (_, label, color) in CASES.items():
        ax.plot(t, ctrl_time(data[name]), color=color, lw=LW, ls=CASE_LS[label], label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Per-step control compute time (s)")
    ax.set_title("Control compute time comparison")
    ax.legend(loc="lower left")
    ax.grid(True)
    fig4.tight_layout()
    plt.show()

    # 4. Cost comparison
    fig5, ax = plt.subplots(figsize=(9, 5))
    for name, (_, label, color) in CASES.items():
        ax.plot(t, realized_cost(data[name]), color=color, lw=LW, ls=CASE_LS[label],
                label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Cost $f = -P/P_{\mathrm{rate}}$")
    ax.set_title("Cost comparison")
    ax.legend(loc="lower left")
    ax.grid(True)
    fig5.tight_layout()
    plt.show()

    # 5. Estimated gradient per turbine (lf vs hf; greedy has no gradient)
    fig6, axes6 = plt.subplots(n, 1, sharex=True, figsize=(9, 4 * n))
    for i in range(n):
        for name, (_, label, color) in CASES.items():
            if name == "greedy":
                continue
            axes6[i].plot(t, gradient(data[name], i), color=color, lw=LW,
                          ls=CASE_LS[label], label=label)
        axes6[i].axhline(0, color="k", lw=0.5)
        axes6[i].set_ylabel(f"Gradient T{i} (1/deg)")
        axes6[i].set_title(f"Estimated gradient T{i}")
        axes6[i].legend(loc="lower left")
        axes6[i].grid(True)
    axes6[-1].set_xlabel("Time (s)")
    fig6.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
