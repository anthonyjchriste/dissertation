import numpy as np

def sem(sigma: float, n: float) -> float:
    return sigma / np.sqrt(n)


def mu_s_sd(n: float,
            mu_s_samp: float,
            mu_sr: float,
            mu_t_sd: float,
            sigma_t_sd: float) -> (float, float):
    # Result
    _mu_s_sd = mu_s_samp * mu_sr * mu_t_sd

    # Errors
    delta_s_samp = 0.0
    delta_sr = 0.0
    delta_t_sd = sem(sigma_t_sd, n)

    samp_term = delta_s_samp / mu_s_samp
    sr_term = delta_sr / mu_sr
    t_sd_term = delta_t_sd / mu_t_sd
    delta_s_sd = np.abs(_mu_s_sd) * np.sqrt((samp_term * samp_term) + (sr_term * sr_term) + (t_sd_term * t_sd_term))

    return _mu_s_sd, delta_s_sd


def mu_s_d(n: float,
           mu_s_samp: float,
           mu_sr: float,
           mu_t_sd: float,
           sigma_t_sd: float,
           mu_sd: float,
           sigma_sd: float) -> (float, float):
    (_mu_s_sd, delta_s_sd) = mu_s_sd(n, mu_s_samp, mu_sr, mu_t_sd, sigma_t_sd)

    _mu_s_d = _mu_s_sd * mu_sd

    delta_sd = sem(sigma_sd, n)

    s_sd_term = delta_s_sd / _mu_s_sd
    sd_term = delta_sd / mu_sd
    delta_s_d = np.abs(_mu_s_d) * np.sqrt((s_sd_term * s_sd_term) + (sd_term * sd_term))

    return _mu_s_d, delta_s_d


def mu_s_dl(n: float,
            mu_s_samp: float,
            mu_sr: float,
            mu_t_sd: float,
            sigma_t_sd: float,
            mu_sd: float,
            sigma_sd: float,
            mu_dr: float,
            sigma_dr: float,
            t: float) -> (float, float):
    _mu_s_d, delta_s_d = mu_s_d(n, mu_s_samp, mu_sr, mu_t_sd, sigma_t_sd, mu_sd, sigma_sd)

    _mu_s_dl = _mu_s_d * mu_dr * t

    delta_dr = sem(sigma_dr, n)

    s_d_term = delta_s_d / _mu_s_d
    dr_term = delta_dr / mu_dr
    delta_s_dl = np.abs(_mu_s_dl) * np.sqrt((s_d_term * s_d_term) + (dr_term * dr_term)) * np.abs(t)

    return _mu_s_dl, delta_s_dl