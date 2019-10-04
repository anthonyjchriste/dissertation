import numpy as np

import laha.errors as errors


def mu_s_sd(samps: float,
            detections: float,
            mu_s_samp: float,
            sigma_s_samp: float,
            mu_sr: float,
            sigma_sr: float,
            mu_t_sd: float,
            sigma_t_sd: float) -> (float, float):
    # Result
    _mu_s_sd = mu_s_samp * mu_sr * mu_t_sd

    # Errors
    delta_s_samp = errors.sem(sigma_s_samp, samps)
    delta_sr = errors.sem(sigma_sr, samps)
    delta_t_sd = errors.sem(sigma_t_sd, detections)

    delta_s_sd = errors.propagate_multiplication(_mu_s_sd,
                                                 (mu_s_samp, delta_s_samp),
                                                 (mu_sr, delta_sr),
                                                 (mu_t_sd, delta_t_sd))

    return _mu_s_sd, delta_s_sd


def mu_s_d(samples: float,
           detections: float,
           mu_s_samp: float,
           mu_sr: float,
           mu_t_sd: float,
           sigma_t_sd: float,
           mu_sd: float,
           sigma_sd: float) -> (float, float):
    (_mu_s_sd, delta_s_sd) = mu_s_sd(samples, detections, mu_s_samp, mu_sr, mu_t_sd, sigma_t_sd)

    _mu_s_d = _mu_s_sd * mu_sd

    delta_sd = errors.sem(sigma_sd, detections)
    delta_s_d = errors.propagate_multiplication(_mu_s_d,
                                                (_mu_s_sd, delta_s_sd),
                                                (mu_sd, delta_sd))

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
