import numpy as np
from pathlib import Path


def parse_wnd(wnd_path):
    '''Parse an InflowWind uniform-wind time-series (.wnd) file.

    Columns (whitespace-separated, lines starting with "!" are comments):
        Time  WindSpeed  WindDir  VertSpeed  HorizShear  VertShear  LinVShear  Gust
    Only the first three are used here.
    '''
    times, speeds, dirs = [], [], []
    with open(wnd_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("!"):
                continue
            parts = line.split()
            times.append(float(parts[0]))
            speeds.append(float(parts[1]))
            dirs.append(float(parts[2]))
    return (np.asarray(times, dtype=float),
            np.asarray(speeds, dtype=float),
            np.asarray(dirs, dtype=float))


def read_wind_profile(inflow, inflow_path):
    '''Read the wind definition from a parsed InflowWind file.

    Returns a wind profile object consumed by wind_at_time():
        ("steady",       speed, floris_dir, None)
        ("time_varying", None,  None,        (times, speeds, dirs))
    '''
    wind_type = int(inflow["WindType"])
    if wind_type == 1:                                   # steady wind
        speed     = float(inflow["HWindSpeed"])
        prop_dir  = float(inflow["PropagationDir"])
        floris_dir = (270.0 + prop_dir) % 360.0          # OpenFAST PropagationDir -> FLORIS "from" direction
        return ("steady", speed, floris_dir, None)
    elif wind_type == 2:                                 # uniform wind time series (.wnd)
        fname = str(inflow["Filename_Uni"]).replace('"', "").strip()
        wnd_path = Path(inflow_path).parent / fname
        return ("time_varying", None, None, parse_wnd(wnd_path))
    else:
        raise NotImplementedError(f"WindType={wind_type} is not supported")


def wind_at_time(profile, t):
    '''Evaluate wind speed [m/s] and FLORIS direction [deg] at time t [s].

    Time-varying wind is linearly interpolated between .wnd samples (and held
    constant outside the sampled range, matching np.interp).'''
    kind = profile[0]
    if kind == "steady":
        return profile[1], profile[2]
    times, speeds, dirs = profile[3]
    speed = float(np.interp(t, times, speeds))
    direction = (270.0 + float(np.interp(t, times, dirs))) % 360.0
    return speed, direction
