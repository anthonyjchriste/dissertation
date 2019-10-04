

import matplotlib.pyplot as plt
import numpy as np


def example_1():
    sample_size_bytes = 2
    sample_rate_hz = 12_000
    sensors_sending_data = np.array([10, 10, 10, 10, 10, 10, 10, 10, 8, 11])  # 80% 10, 10% 8, 10% 11
    mean_sensors_sending_data = sensors_sending_data.mean()
    sigma_sensors_sending_data = sensors_sending_data.std()

    x_values = np.arange(1, 31_540_000, step=86_400)  # seconds in year by seconds in day
    y_values = sample_size_bytes * sample_rate_hz * mean_sensors_sending_data * x_values

    e_values = []
    for i in range(len(x_values)):
        delta_sensors_sending_data = sigma_sensors_sending_data / np.sqrt(x_values[i])
        delta_mean_size = delta_sensors_sending_data * np.abs(sample_size_bytes * sample_rate_hz * x_values[i])
        e_values.append(delta_mean_size)
    e_values = np.array(e_values)

    plt.plot(x_values, y_values)
    plt.plot(x_values, y_values + e_values)
    plt.plot(x_values, y_values - e_values)

    plt.title("Example 1")
    plt.xlabel("Time (S)")
    plt.ylabel("Bytes")

    plt.show()


def example_2():
    sensors_sending_data = np.array([10, 10, 10, 10, 10, 10, 10, 10, 8, 11])  # 80% 10, 10% 8, 10% 11
    mean_sample_size_bytes = 2
    sigma_sample_size_bytes = 0.0
    mean_sample_rate_hz = 12_000
    sigma_sample_rate_hz = 0.0
    mean_sensors_sending_data = sensors_sending_data.mean()
    sigma_sensors_sending_data = sensors_sending_data.std()

    x_values = np.arange(1, 31_540_000, step=86_400)  # seconds in year by seconds in day
    y_values = mean_sample_size_bytes * mean_sample_rate_hz * mean_sensors_sending_data * x_values

    e_values = []
    for i in range(len(x_values)):
        num_samples = x_values[i] * mean_sample_rate_hz
        delta_sample_size_bytes = sigma_sample_size_bytes / np.sqrt(num_samples)
        delta_sample_rate_hz = sigma_sample_rate_hz / np.sqrt(num_samples)
        delta_sensors_sending_data = sigma_sensors_sending_data / np.sqrt(x_values[i])
        delta_size_bytes = np.abs(y_values[i]) * np.sqrt((delta_sample_size_bytes / mean_sample_size_bytes) ** 2 + (
                    delta_sample_rate_hz / mean_sample_rate_hz) ** 2 + (
                                                                     delta_sensors_sending_data /
                                                                     mean_sensors_sending_data) ** 2) * \
                           np.abs(x_values[i])
        e_values.append(delta_size_bytes)
    e_values = np.array(e_values)

    plt.plot(x_values, y_values)
    plt.plot(x_values, y_values + e_values)
    plt.plot(x_values, y_values - e_values)

    plt.title("Example 2")
    plt.xlabel("Time (S)")
    plt.ylabel("Bytes")

    plt.show()


if __name__ == "__main__":
    example_1()
    example_2()
