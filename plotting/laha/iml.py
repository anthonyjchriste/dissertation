import numpy as np


def s_sen(s_samp: float, sr: float, t: float) -> float:
    return s_samp * sr * t


def s_iml(s_sens: np.ndarray) -> float:
    return s_sens.sum()


def mu_s_sen(samp: int,
             mu_s_samp: float,
             sigma_s_samp: float,
             mu_sr: float,
             sigma_sr: float,
             t: float) -> (float, float):
    _mu_s_sen = mu_s_samp * mu_sr * t

    delta_s_samp = sigma_s_samp / np.sqrt(samp)
    delta_sr = sigma_sr / np.sqrt(samp)

    s_samp_term = delta_s_samp / mu_s_samp
    sr_term = delta_sr / mu_sr
    delta_s_sen = np.abs(_mu_s_sen) * np.sqrt((s_samp_term * s_samp_term) + (sr_term * sr_term)) * np.abs(t)

    return _mu_s_sen, delta_s_sen


def mu_s_iml(samp: int,
             mu_s_samp: float,
             sigma_s_samp: float,
             mu_sr: float,
             sigma_sr: float,
             mu_b: float,
             sigma_b: float,
             t: float) -> (float, float):
    _mu_s_iml = mu_s_samp * mu_sr * mu_b * t

    delta_s_samp = sigma_s_samp / np.sqrt(samp)
    delta_sr = sigma_sr / np.sqrt(samp)
    delta_b = sigma_b / np.sqrt(samp)

    s_samp_term = delta_s_samp / mu_s_samp
    sr_term = delta_sr / mu_sr
    b_term = delta_b / mu_b
    delta_s_iml = np.abs(_mu_s_iml) * np.sqrt(
        (s_samp_term * s_samp_term) + (sr_term * sr_term) + (b_term * b_term)) * np.abs(t)

    return _mu_s_iml, delta_s_iml
