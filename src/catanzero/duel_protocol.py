"""Engine-free definitions shared by the simulator and MLflow coordinator."""
from math import sqrt

CONDITIONS = {"neither": (1, "F_static"), "f_only": (1, "F"),
              "v4_only": (2, "F_static"), "both": (2, "F")}


def wilson_interval(wins, trials, z=1.959963984540054):
    if trials == 0:
        return [0., 1.]
    p = wins / trials
    denominator = 1 + z*z/trials
    center = (p + z*z/(2*trials)) / denominator
    half = z * sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / denominator
    return [max(0., center-half), min(1., center+half)]
