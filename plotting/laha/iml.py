import numpy as np

import laha.errors as errors


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

    delta_s_samp = errors.sem(sigma_s_samp, samp)
    delta_sr = errors.sem(sigma_sr, samp)

    e = errors.propagate_multiplication(_mu_s_sen,
                                        (mu_s_samp, delta_s_samp),
                                        (mu_sr, delta_sr))
    delta_s_sen = errors.propagate_constant(e, t)

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

    delta_s_samp = errors.sem(sigma_s_samp, samp)
    delta_sr = errors.sem(sigma_sr, samp)
    delta_b = errors.sem(sigma_b, samp)

    e = errors.propagate_multiplication(_mu_s_iml,
                                        (mu_s_samp, delta_s_samp),
                                        (mu_sr, delta_sr),
                                        (mu_b, delta_b))
    delta_s_iml = errors.propagate_constant(e, t)

    return _mu_s_iml, delta_s_iml
