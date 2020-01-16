import typing

import matplotlib.pyplot as plt
import numpy as np

import laha.iml as iml
import laha.dl as dl

S_IN_DAY = 86_400
S_IN_YEAR = 31_540_000


def plot_iml_level_opq():
    plt.figure(figsize=(12, 5))
    sample_size_bytes = 2
    sample_rate_hz = 12_000
    x_values = np.arange(S_IN_YEAR, step=S_IN_DAY)
    y_values = x_values * sample_size_bytes * sample_rate_hz

    plt.plot(x_values, y_values)

    plt.title("IML Size (Lokahi) Sample Size=4, SR=12000, Len=1yr")
    plt.xlabel("Time (S)")
    plt.ylabel("Bytes")

    plt.savefig("../src/figures/plot_iml_level_opq.png")
    plt.show()


def plot_iml_level_lokahi():
    plt.figure(figsize=(12, 5))
    sample_size_bytes = 4
    sample_rates_hz = [80, 800, 8000]
    x_values = np.arange(S_IN_YEAR, step=S_IN_DAY)
    for sample_rate_hz in sample_rates_hz:
        y_values = x_values * sample_size_bytes * sample_rate_hz
        plt.plot(x_values, y_values, label="%d Hz" % sample_rate_hz)

    plt.title("IML Size (Lokahi) Sample Size=4, SR=[80,800,8000], Len=1yr")
    plt.xlabel("Time (S)")
    plt.ylabel("Bytes")

    plt.legend()
    plt.savefig("../src/figures/plot_iml_level_lokahi.png")
    plt.show()


def plot_iml_level_no_opt_var_sample_size(sample_sizes_bytes: typing.List[int],
                                          sample_rate_hz: int,
                                          window_length_s: int):
    x_values = np.arange(window_length_s, step=S_IN_DAY)

    plt.figure(figsize=(12, 5))

    for sample_size_bytes in sample_sizes_bytes:
        y_values = x_values * sample_size_bytes * sample_rate_hz
        plt.plot(x_values, y_values, label="%d bytes" % sample_size_bytes)

    plt.title("IML No Opt: Sample Size Bytes=%s, Sample Rate Hz=%d, Window Length S=%d" % (
        str(sample_sizes_bytes),
        sample_rate_hz,
        window_length_s
    ))
    plt.xlabel("Time (S)")
    plt.ylabel("Bytes")
    plt.legend()

    plt.savefig("../src/figures/plot_iml_level_no_opt_var_sample_size.png")
    plt.show()


def plot_iml_level_no_opt_var_sample_rate(sample_size_bytes: int,
                                          sample_rates_hz: typing.List[int],
                                          window_length_s: int):
    x_values = np.arange(window_length_s, step=S_IN_DAY)

    plt.figure(figsize=(12, 5))

    for sample_rate_hz in sample_rates_hz:
        y_values = x_values * sample_size_bytes * sample_rate_hz
        plt.plot(x_values, y_values, label="%d Hz" % sample_rate_hz)

    plt.title("IML No Opt: Sample Size Bytes=%d, Sample Rate Hz=%s, Window Length S=%d" % (
        sample_size_bytes,
        str(sample_rates_hz),
        window_length_s
    ))
    plt.xlabel("Time (S)")
    plt.ylabel("Bytes")
    plt.legend()

    plt.savefig("../src/figures/plot_iml_level_no_opt_var_sample_rate.png")
    plt.show()


def plot_iml_level_no_opt_var_num_sensors(sample_size_bytes: int,
                                          sample_rate_hz: int,
                                          window_length_s: int,
                                          num_boxes: typing.List[int]):
    x_values = np.arange(window_length_s, step=S_IN_DAY)

    plt.figure(figsize=(12, 5))

    for boxes in num_boxes:
        y_values = (x_values * sample_size_bytes * sample_rate_hz) * boxes
        plt.plot(x_values, y_values, label="%d Sensors" % boxes)

    plt.title("IML No Opt: Sample Size=%d, Hz=%d, Window Len S=%d, Sensors=%s" % (
        sample_size_bytes,
        sample_rate_hz,
        window_length_s,
        str(num_boxes)
    ))
    plt.xlabel("Time (S)")
    plt.ylabel("Bytes")
    plt.yscale("log")
    plt.legend()

    plt.savefig("../src/figures/plot_iml_level_no_opt_var_num_sensors.png")
    plt.show()


def plot_iml_level_no_opt_var_std(sample_size_bytes: int,
                                  mean_sample_rate_hz: int,
                                  window_length_s: int,
                                  std: float):
    x_values = np.arange(1, window_length_s + 1, step=S_IN_DAY)

    plt.figure(figsize=(12, 5))

    # First, lets plot the average
    y_values = x_values * sample_size_bytes * mean_sample_rate_hz
    # plt.plot(x_values, y_values, label="Mean Size Bytes")

    e_sr = std / np.sqrt(mean_sample_rate_hz * x_values)
    e = e_sr * np.abs(mean_sample_rate_hz * x_values)

    plt.errorbar(x_values, y_values, yerr=e)

    plt.show()


def plot_aml_level_opq_single(window_length_s: int):
    sub_levels = ["measurements", "trends"]

    sl_to_size = {
        "measurements": 144,
        "trends": 323
    }

    sl_to_rate = {
        "measurements": 1,
        "trends": 60
    }

    plt.figure(figsize=(12, 5))

    x_values = np.arange(1, window_length_s + 1, step=S_IN_DAY)

    total_y = np.zeros(len(x_values))
    for sl in sub_levels:
        size = sl_to_size[sl]
        rate = sl_to_rate[sl]
        y_values = x_values * size * 1.0 / rate
        total_y += y_values
        plt.plot(x_values, y_values, label="Sub-Level=%s, Size Bytes=%d, Rate Hz=1/%d" % (sl, size, rate))

    plt.plot(x_values, total_y, label="AML Total Bytes")

    plt.xlabel("Time (S)")
    plt.ylabel("Bytes")

    plt.title("AML Data Growth (OPQ): Window Len S=%d" % window_length_s)

    plt.legend()
    plt.savefig("../src/figures/plot_aml_level_opq_single.png")
    plt.show()


def plot_aml_level_lokahi_single(window_length_s: int):
    sub_levels = ["80Hz", "800Hz", "8000Hz"]
    sl_to_size = {
        "80Hz": 3117,
        "800Hz": 3117,
        "8000Hz": 3117
    }
    sl_to_rate = {
        "80Hz": 51.2,
        "800Hz": 40.96,
        "8000Hz": 32.768
    }

    plt.figure(figsize=(12, 5))

    x_values = np.arange(1, window_length_s + 1, step=S_IN_DAY)

    total_y = np.zeros(len(x_values))
    for sl in sub_levels:
        size = sl_to_size[sl]
        rate = sl_to_rate[sl]
        y_values = x_values * size * 1.0 / rate
        total_y += y_values
        plt.plot(x_values, y_values, label="Sub-Level=%s, Size Bytes=%d, Rate Hz=1/%f" % (sl, size, rate))

    # plt.plot(x_values, total_y, label="AML Total Bytes")

    plt.xlabel("Time (S)")
    plt.ylabel("Bytes")

    plt.title("AML Data Growth (Lokahi): Window Len S=%d" % window_length_s)

    plt.legend()
    plt.savefig("../src/figures/plot_aml_level_lokahi_single.png")
    plt.show()


def plot_dl_opq_no_err():
    plt.figure(figsize=(12, 5))
    # x_values = np.arange(S_IN_YEAR, step=S_IN_DAY)
    x_values = np.arange(S_IN_DAY * 2, step=S_IN_DAY)
    N = 93472.0
    mu_s_samp = 2.0
    sigma_s_samp = 0.0
    mu_sr = 12_000.0
    sigma_sr = 0.0
    mu_t_sd = 11.787460720323569
    sigma_t_sd = 15.040829579595933
    mu_dr = 0.293433583168666
    sigma_dr = 10.403490650228573
    mu_sd = 1.185407440686306
    sigma_sd = 1.0209460091478992

    # y_values = (mean_sample_size * mean_sample_rate * mean_event_len) * mean_event_rate * mean_boxes_recv * x_values
    y_values = []
    e_values = []
    for t in x_values:
        y, e = dl.mu_s_dl(N,
                          mu_s_samp,
                          mu_sr,
                          mu_t_sd,
                          sigma_t_sd,
                          mu_sd,
                          sigma_sd,
                          mu_dr,
                          sigma_dr,
                          t)
        y_values.append(y)
        e_values.append(e)

    y_values = np.array(y_values)
    e_values = np.array(e_values)

    plt.plot(x_values, y_values)

    # plt.plot(x_values, y_values + e_values)
    # plt.plot(x_values, y_values - e_values)

    plt.title("$\mu$ DL (OPQ)")
    plt.xlabel("Time (S)")
    plt.ylabel("Bytes")
    plt.savefig("../src/figures/plot_dl_opq_no_err.png")
    plt.show()


def plot_dl_opq_err():
    plt.figure(figsize=(12, 5))
    # x_values = np.arange(S_IN_YEAR, step=S_IN_DAY)
    x_values = np.arange(S_IN_DAY * 2, step=S_IN_DAY)
    N = 93472.0
    mu_s_samp = 2.0
    sigma_s_samp = 0.0
    mu_sr = 12_000.0
    sigma_sr = 0.0
    mu_t_sd = 11.787460720323569
    sigma_t_sd = 15.040829579595933
    mu_dr = 0.293433583168666
    sigma_dr = 10.403490650228573
    mu_sd = 1.185407440686306
    sigma_sd = 1.0209460091478992

    # y_values = (mean_sample_size * mean_sample_rate * mean_event_len) * mean_event_rate * mean_boxes_recv * x_values
    y_values = []
    e_values = []
    for t in x_values:
        y, e = dl.mu_s_dl(N,
                          mu_s_samp,
                          mu_sr,
                          mu_t_sd,
                          sigma_t_sd,
                          mu_sd,
                          sigma_sd,
                          mu_dr,
                          sigma_dr,
                          t)
        y_values.append(y)
        e_values.append(e)

    y_values = np.array(y_values)
    e_values = np.array(e_values)

    plt.plot(x_values, y_values, label="$\mu$ DL")

    plt.plot(x_values, y_values + e_values, label="$+\delta$")
    plt.plot(x_values, y_values - e_values, label="$-\delta$")

    plt.title("$\mu$ DL (OPQ) with Error Bounds")
    plt.xlabel("Time (S)")
    plt.ylabel("Bytes")
    plt.legend()
    plt.savefig("../src/figures/plot_dl_opq_err.png")
    plt.show()


def plot_iml_avg_opq():
    plt.figure(figsize=(12, 5))
    x_values = np.arange(S_IN_YEAR, step=S_IN_DAY)

    y_values = []
    e_values = []
    for t in x_values:
        samps = t * 12_000
        y, e = iml.mu_s_iml(samps,
                            2,
                            0.0,
                            12_000,
                            0.0,
                            8.9,
                            0.7,
                            t)
        y_values.append(y)
        e_values.append(e)
        # print(e)

    e_values = np.array(e_values)
    y_values = np.array(y_values)

    plt.plot(x_values, y_values)
    plt.plot(x_values, y_values + e_values)
    plt.plot(x_values, y_values - e_values)
    plt.show()


def plot_iml_average():
    plt.figure(figsize=(12, 5))
    s_samp = 2
    sr = 12_000
    mu_n_sen = 15.0
    sigma_n_sen = 0.7
    x_values = np.arange(S_IN_YEAR, step=S_IN_DAY)
    y_values = s_samp * sr * mu_n_sen * x_values / 1_000_000_000.0
    e_values = (sigma_n_sen / np.sqrt(x_values)) * np.abs(s_samp * sr * x_values)

    print("iml size one day", y_values[1])
    print("iml size one week", y_values[7])
    print("iml size one year", y_values[365])

    plt.plot(x_values, y_values)

    plt.title("IML Estimated Size Growth: SR=12000, Size Sample=2, T=1yr, ")
    plt.xlabel("Time (S)")
    plt.ylabel("GB")

    plt.legend()
    plt.savefig("../src/figures/plot_iml_opq_avg.png")
    plt.show()


def plot_dl_opq_avg():
    plt.figure(figsize=(12, 5))

    mu_dr = 3138.2362879905268
    sigma_dr = 185544.8

    x_values = np.arange(1, S_IN_YEAR, step=S_IN_DAY)
    y_values = mu_dr * x_values / 1_000_000.0
    e_values = (sigma_dr / np.sqrt(x_values)) * np.abs(x_values)

    print("dl size one day", y_values[1])
    print("dl size one week", y_values[7])
    print("dl size one year", y_values[365])

    plt.plot(x_values, y_values)

    plt.title("DL (OPQ) $\mu DR$=13064.9 T=1yr")
    plt.xlabel("Time (S)")
    plt.ylabel("GB")

    plt.legend()
    plt.savefig("../src/figures/plot_dl_opq_avg.png")
    plt.show()


def plot_il_opq_avg():
    plt.figure(figsize=(12, 5))

    mu_dr = 234.83720746762222
    sigma_dr = 6288.48

    x_values = np.arange(S_IN_YEAR, step=S_IN_DAY)
    y_values = mu_dr * x_values / 1_000_000_000.0
    e_values = (sigma_dr / np.sqrt(x_values)) * np.abs(x_values)

    print("il size one day", y_values[1])
    print("il size one week", y_values[7])
    print("il size one year", y_values[365])

    plt.plot(x_values, y_values)

    plt.title("IL (OPQ) $\mu IR$=234.8 T=1yr")
    plt.xlabel("Time (S)")
    plt.ylabel("GB")

    plt.legend()
    plt.savefig("../src/figures/plot_il_opq_avg.png")
    plt.show()


def gb(b: float) -> float:
    return b / 1024.0 / 1024.0 / 1024.0


def plot_laha_opq():
    plt.figure(figsize=(12, 5))
    x_values = np.arange(1, S_IN_YEAR * 3, step=S_IN_DAY)

    # IML
    mu_n_sens = 1.0
    sigma_n_sens = 0.7
    s_samp = 2
    sr = 12000

    iml_size = s_samp * sr * mu_n_sens * x_values / 1_000_000_000.0
    iml_error = (sigma_n_sens / np.sqrt(x_values)) * np.abs(s_samp * sr * x_values)
    plt.plot(x_values, iml_size, label="IML")

    # AML
    s_meas = 145.0
    r_meas = 1.0 / 1.0
    s_trend = 365.0
    r_trend = 1.0 / 60.0
    aml_size_meas = s_meas * r_meas * mu_n_sens * x_values / 1_000_000_000.0
    aml_size_trend = s_trend * r_trend * mu_n_sens * x_values / 1_000_000_000.0
    aml_size_total = aml_size_meas + aml_size_trend

    aml_size_meas_e = (sigma_n_sens / np.sqrt(x_values)) * np.abs(s_meas * r_meas * x_values)
    aml_size_trends_e = (sigma_n_sens / np.sqrt(x_values)) * np.abs(s_trend * r_trend * x_values)
    aml_size_e = np.sqrt(aml_size_meas_e ** 2 + aml_size_trends_e ** 2)

    plt.plot(x_values, aml_size_meas, label="AML (Measurements)")
    plt.plot(x_values, aml_size_trend, label="AML (Trends)")
    plt.plot(x_values, aml_size_total, label="AML")

    # DL
    mu_dr = 40.69
    sigma_dr = 3138.2362879905268

    dl_size = mu_dr * x_values / 1_000_000_000.0
    dl_size_e = (sigma_dr / np.sqrt(x_values)) * np.abs(x_values)
    plt.plot(x_values, dl_size, label="DL")

    # IL
    mu_il = 234.83720746762222
    sigma_il = 6288.484706127778

    il_size = mu_il * x_values / 1_000_000_000.0
    il_size_e = (sigma_il / np.sqrt(x_values)) * np.abs(x_values)
    plt.plot(x_values, il_size, label="IL")

    # PL
    mu_pl = 0.22
    pl_size = mu_pl * x_values / 1_000_000_000.0
    plt.plot(x_values, pl_size, label="PL")

    # Total
    laha_size = iml_size + aml_size_total + dl_size + il_size + pl_size
    laha_size_e = np.sqrt(iml_error ** 2 + aml_size_e ** 2 + dl_size_e ** 2 + il_size_e ** 2)
    plt.plot(x_values, laha_size, label="Laha")

    print("iml", iml_size[-1], iml_error[-1])
    print("aml meas", aml_size_meas[-1], aml_size_meas_e[-1])
    print("aml trend", aml_size_trend[-1], "%f" % aml_size_trends_e[-1])
    print("aml total", aml_size_total[-1], aml_size_e[-1])
    print("dl", dl_size[-1], dl_size_e[-1])
    print("il", il_size[-1], il_size_e[-1])
    print("pl", pl_size[-1])
    print("total", laha_size[-1], laha_size_e[-1])

    plt.yscale("log")
    plt.title("Laha (OPQ)")
    plt.xlabel("Time (s)")
    plt.ylabel("GB")

    plt.legend()
    plt.savefig("../src/figures/plot_laha_opq.png")
    plt.show()


def plot_laha_opq_no_iml():
    plt.figure(figsize=(12, 5))
    x_values = np.arange(1, S_IN_YEAR, step=S_IN_DAY)

    # IML
    mu_n_sens = 15.0
    sigma_n_sens = 0.7

    # AML
    s_meas = 145.0
    r_meas = 1.0 / 1.0
    s_trend = 365.0
    r_trend = 1.0 / 60.0
    aml_size_meas = s_meas * r_meas * mu_n_sens * x_values / 1_000_000_000.0
    aml_size_trend = s_trend * r_trend * mu_n_sens * x_values / 1_000_000_000.0
    aml_size_total = aml_size_meas + aml_size_trend

    aml_size_meas_e = (sigma_n_sens / np.sqrt(x_values)) * np.abs(s_meas * r_meas * x_values)
    aml_size_trends_e = (sigma_n_sens / np.sqrt(x_values)) * np.abs(s_trend * r_trend * x_values)
    aml_size_e = np.sqrt(aml_size_meas_e ** 2 + aml_size_trends_e ** 2)

    plt.plot(x_values, aml_size_meas, label="AML (Measurements)")
    plt.plot(x_values, aml_size_trend, label="AML (Trends)")
    plt.plot(x_values, aml_size_total, label="AML")

    # DL
    mu_dr = 3138.2362879905268
    sigma_dr = 185544.81743550362

    dl_size = mu_dr * x_values/ 1_000_000_000.0
    dl_size_e = (sigma_dr / np.sqrt(x_values)) * np.abs(x_values)
    plt.plot(x_values, dl_size, label="DL")

    # IL
    mu_il = 234.83720746762222
    sigma_il = 6288.484706127778

    il_size = mu_il * x_values/ 1_000_000_000.0
    il_size_e = (sigma_il / np.sqrt(x_values)) * np.abs(x_values)
    plt.plot(x_values, il_size, label="IL")

    # PL
    mu_pl = 0.22
    pl_size = mu_pl * x_values / 1_000_000_000.0

    # Total
    laha_size = aml_size_total + dl_size + il_size + pl_size
    laha_size_e = np.sqrt(aml_size_e ** 2 + dl_size_e ** 2 + il_size_e ** 2)
    plt.plot(x_values, laha_size, label="Laha")

    plt.title("Laha (OPQ) Sans IML")
    plt.xlabel("Time (S)")
    plt.ylabel("Bytes")

    plt.legend()
    plt.savefig("../src/figures/plot_laha_opq_no_iml.png")
    plt.show()


def plot_laha_opq_pie():
    plt.figure(figsize=(12, 5))
    x_values = np.arange(1, S_IN_YEAR * 3, step=S_IN_DAY)

    # IML
    mu_n_sens = 1.0
    sigma_n_sens = 15.0
    s_samp = 2
    sr = 12000

    iml_size = s_samp * sr * mu_n_sens * x_values

    # AML
    s_meas = 145.0
    r_meas = 1.0 / 1.0
    s_trend = 365.0
    r_trend = 1.0 / 60.0
    aml_size_meas = s_meas * r_meas * mu_n_sens * x_values
    aml_size_trend = s_trend * r_trend * mu_n_sens * x_values

    # DL
    mu_dr = 3138.2362879905268
    dl_size = mu_dr * x_values
    # IL
    mu_il = 234.83720746762222
    il_size = mu_il * x_values

    # PL
    mu_pl = 0.22
    pl_size = mu_pl * x_values

    plt.subplot(1, 2, 1)
    labels = ["IML", "AML", "DL", "IL", "PL"]
    values = [gb(iml_size[-1]), gb(aml_size_meas[-1] + aml_size_trend[-1]), gb(dl_size[-1]), gb(il_size[-1]), gb(pl_size[-1])]
    total = sum(values)
    plt.pie(values, labels=labels, autopct=lambda p: "%.2f gb" % (p * total / 100.0))
    plt.title("Laha (OPQ) w/ IML")

    plt.subplot(1, 2, 2)
    labels = ["AML (M)", "AML (T)", "DL", "IL", "PL"]
    values = [gb(aml_size_meas[-1]), gb(aml_size_trend[-1]), gb(dl_size[-1]), gb(il_size[-1]), gb(pl_size[-1])]
    total = sum(values)
    plt.pie(values, labels=labels, autopct=lambda p: "%.2f gb" % (p * total / 100.0))

    plt.title("Laha (OPQ) w/o IML")

    plt.savefig("../src/figures/plot_laha_opq_pie.png")
    plt.show()


def plot_laha_lokahi_pie():
    plt.figure(figsize=(12, 5))
    x_values = np.arange(1, S_IN_YEAR * 3, step=S_IN_DAY)

    # IML
    mu_n_sens = 1.0
    sigma_n_sens = 0.0
    s_samp = 4
    sr = 8000

    iml_size = s_samp * sr * mu_n_sens * x_values

    # AML
    # s_meas = 0
    # r_meas = 1.0 / 1.0
    s_trend = 2471.0
    r_trend = 1.0 / 32.768
    # aml_size_meas = s_meas * r_meas * mu_n_sens * x_values
    aml_size_trend = s_trend * r_trend * mu_n_sens * x_values

    # DL
    mu_dr = 402.81624834955454
    dl_size = mu_dr * x_values
    # IL
    mu_il = 37.11652361890925
    il_size = mu_il * x_values

    # PL
    pl_size = 0.01 * x_values

    plt.subplot(1, 2, 1)
    labels = ["IML", "AML", "DL", "IL", "PL"]
    values = [gb(iml_size[-1]), gb(aml_size_trend[-1]), gb(dl_size[-1]), gb(il_size[-1]), gb(pl_size[-1])]
    total = sum(values)
    plt.pie(values, labels=labels, autopct=lambda p: "%.2f gb" % (p * total / 100.0))
    plt.title("Laha (Lokahi) w/ IML")

    plt.subplot(1, 2, 2)
    labels = ["AML (T)", "DL", "IL", "PL"]
    values = [gb(aml_size_trend[-1]), gb(dl_size[-1]), gb(il_size[-1]), gb(pl_size[-1])]
    total = sum(values)
    plt.pie(values, labels=labels, autopct=lambda p: "%.2f gb" % (p * total / 100.0))

    plt.title("Laha (Lokahi) w/o IML")

    plt.savefig("../src/figures/plot_laha_lokahi_pie.png")
    plt.show()


def plot_laha_lokahi():
    plt.figure(figsize=(12, 5))
    x_values = np.arange(1, S_IN_YEAR * 3, step=S_IN_DAY)

    # IML
    mu_n_sens = 1.0
    sigma_n_sens = 0.0
    s_samp = 4
    sr = 8000

    iml_size = s_samp * sr * mu_n_sens * x_values
    iml_error = (sigma_n_sens / np.sqrt(x_values)) * np.abs(s_samp * sr * x_values)
    plt.errorbar(x_values, iml_size / 1_000_000_000.0, yerr=iml_error / 1_000_000_000.0, label="IML")

    # AML
    # s_meas = 145.0
    # r_meas = 1.0 / 1.0
    s_trend = 2471
    r_trend = 1.0 / 32.768
    # aml_size_meas = s_meas * r_meas * mu_n_sens * x_values
    aml_size_trend = s_trend * r_trend * mu_n_sens * x_values
    aml_size_total = aml_size_trend

    # aml_size_meas_e = (sigma_n_sens / np.sqrt(x_values)) * np.abs(s_meas * r_meas * x_values)
    aml_size_trends_e = (sigma_n_sens / np.sqrt(x_values)) * np.abs(s_trend * r_trend * x_values)
    aml_size_e = aml_size_trends_e

    # plt.errorbar(x_values, aml_size_meas, yerr=aml_size_meas_e, label="AML (Measurements)")
    plt.errorbar(x_values, aml_size_trend  / 1_000_000_000.0, yerr=aml_size_trends_e / 1_000_000_000.0, label="AML (Trends)")
    plt.errorbar(x_values, aml_size_total  / 1_000_000_000.0, yerr=aml_size_e / 1_000_000_000.0, label="AML")

    # DL
    mu_dr = 402.81624834955454
    sigma_dr = 0

    dl_size = mu_dr * x_values
    dl_size_e = (sigma_dr / np.sqrt(x_values)) * np.abs(x_values)
    plt.errorbar(x_values, dl_size  / 1_000_000_000.0, yerr=dl_size_e  / 1_000_000_000.0, label="DL")

    # IL
    mu_il = 37.11652361890925
    sigma_il = 0.0

    il_size = mu_il * x_values
    il_size_e = (sigma_il / np.sqrt(x_values)) * np.abs(x_values)
    plt.errorbar(x_values, il_size  / 1_000_000_000.0, yerr=il_size_e  / 1_000_000_000.0, label="IL")

    # PL
    mu_pl = 0.01
    pl_size = mu_pl * x_values
    plt.plot(x_values, pl_size / 1_000_000_000.0, label="PL")

    # Total
    laha_size = iml_size + aml_size_total + dl_size + il_size + pl_size
    laha_size_e = np.sqrt(iml_error ** 2 + aml_size_e ** 2 + dl_size_e ** 2 + il_size_e ** 2)
    plt.errorbar(x_values, laha_size / 1_000_000_000.0, yerr=laha_size_e / 1_000_000_000.0, label="Laha")

    print("iml", gb(iml_size[-1]), gb(iml_error[-1]))
    # print("aml meas", gb(aml_size_meas[-1]), gb(aml_size_meas_e[-1]))
    print("aml trend", gb(aml_size_trend[-1]), "%f" % gb(aml_size_trends_e[-1]))
    print("aml total", gb(aml_size_total[-1]), gb(aml_size_e[-1]))
    print("dl", gb(dl_size[-1]), gb(dl_size_e[-1]))
    print("il", gb(il_size[-1]), gb(il_size_e[-1]))
    print("pl", gb(pl_size[-1]))
    print("total", gb(laha_size[-1]), gb(laha_size_e[-1]))

    plt.title("Laha (Lokahi)")
    plt.xlabel("Time (S)")
    plt.ylabel("GB")
    plt.yscale("log")
    plt.legend()
    plt.savefig("../src/figures/plot_laha_lokahi.png")
    plt.show()


if __name__ == "__main__":
    # sample_sizes = [1, 2, 4, 8, 16]
    # sample_rates = [80, 800, 8_000, 12_000]
    # num_boxes = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
    # plot_iml_level_no_opt_var_sample_size(sample_sizes, 10, S_IN_YEAR)
    # plot_iml_level_no_opt_var_sample_rate(2, sample_rates, S_IN_YEAR)
    # plot_iml_level_no_opt_var_num_sensors(4, 12000, S_IN_YEAR, num_boxes)
    # plot_iml_level_no_opt_var_std(2, 80, S_IN_DAY * 5, 1000)
    # plot_aml_level_opq_single(S_IN_YEAR)
    # plot_aml_level_lokahi_single(S_IN_YEAR)
    # plot_iml_level_opq()
    # plot_iml_level_lokahi()
    # plot_dl_opq_no_err()
    # plot_dl_opq_err()
    # plot_iml_average()
    # plot_dl_opq_avg()
    # plot_il_opq_avg()
    # plot_laha_opq()
    # plot_laha_opq_no_iml()
    # plot_laha_opq_pie()
    plot_laha_lokahi_pie()
    # plot_laha_lokahi()
