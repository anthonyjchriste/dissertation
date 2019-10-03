import typing

import matplotlib.pyplot as plt
import numpy as np

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

    plt.plot(x_values, total_y, label="AML Total Bytes")

    plt.xlabel("Time (S)")
    plt.ylabel("Bytes")

    plt.title("AML Data Growth (Lokahi): Window Len S=%d" % window_length_s)

    plt.legend()
    plt.savefig("../src/figures/plot_aml_level_lokahi_single.png")
    plt.show()


def plot_dl_opq():
    plt.figure(figsize=(12, 5))
    x_values = np.arange(S_IN_YEAR, step=S_IN_DAY)
    N = 93472.0
    mean_sample_size = 2.0
    std_sample_size = 0.0
    mean_sample_rate = 12_000.0
    std_sample_rate = 0.0
    mean_event_len = 11.787460720323569
    std_event_len = 15.040829579595933
    mean_event_rate = 0.293433583168666
    std_event_rate = 0.293433583168666
    mean_boxes_recv = 1.185407440686306
    std_boxes_recv = 1.0209460091478992

    e_sample_size = 0.0
    e_sample_rate = 0.0
    e_event_len = std_event_len / np.sqrt(N)
    e_event_rate = std_event_rate / np.sqrt(N)
    e_boxes_recv = std_boxes_recv / np.sqrt(N)
    e_s_sd = np.abs(mean_boxes_recv)

    y_values = (mean_sample_size * mean_sample_rate * mean_event_len) * mean_event_rate * mean_boxes_recv * x_values

    plt.plot(x_values, y_values)
    plt.savefig("../src/figures/plot_dl_opq.png")
    plt.show()


def plot_dl_lokahi():
    pass


if __name__ == "__main__":
    sample_sizes = [1, 2, 4, 8, 16]
    sample_rates = [80, 800, 8_000, 12_000]
    num_boxes = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]
    # plot_iml_level_no_opt_var_sample_size(sample_sizes, 10, S_IN_YEAR)
    # plot_iml_level_no_opt_var_sample_rate(2, sample_rates, S_IN_YEAR)
    # plot_iml_level_no_opt_var_num_sensors(4, 12000, S_IN_YEAR, num_boxes)
    # plot_iml_level_no_opt_var_std(2, 80, S_IN_DAY * 5, 1000)
    # plot_aml_level_opq_single(S_IN_YEAR)
    # plot_aml_level_lokahi_single(S_IN_YEAR)
    # plot_iml_level_opq()
    # plot_iml_level_lokahi()
    plot_dl_opq()
