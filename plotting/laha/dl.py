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


def mu_s_dl(samples: float,
            detections: float,
            mu_s_samp: float,
            mu_sr: float,
            mu_t_sd: float,
            sigma_t_sd: float,
            mu_sd: float,
            sigma_sd: float,
            mu_dr: float,
            sigma_dr: float,
            t: float) -> (float, float):
    _mu_s_d, delta_s_d = mu_s_d(samples, detections, mu_s_samp, mu_sr, mu_t_sd, sigma_t_sd, mu_sd, sigma_sd)

    _mu_s_dl = _mu_s_d * mu_dr * t

    delta_dr = errors.sem(sigma_dr, detections)
    e = errors.propagate_multiplication(_mu_s_dl,
                                        (_mu_s_d, delta_s_d),
                                        (mu_dr, delta_dr))
    delta_s_dl = errors.propagate_constant(e, t)

    return _mu_s_dl, delta_s_dl
