# Let's say we wanted to plot upper and lower bounds on a sensor network the following known parameters
# - sample_size_bytes
# - sample_rate_hz
# - sensing_length_seconds
#
# And the following estimated parameter
# - mean_sensors_sending_data
#
# mean_size_bytes = sample_size_bytes * sample_rate_hz * sensing_length_seconds * mean_sensors_sending_data
#
# So far, so good. Now we want to calculate the error bounds. We'll use standard error of the mean.
#
# delta_sensors_sending_data = sigma_sensors_sending_data / sqrt(sensing_length_seconds)
# delta_size_bytes = delta_sensors_sending_data * |mean_size_bytes|

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    sample_size_bytes = 2
    sample_rate_hz = 12_000
    sensors_sending_data = np.array([10, 10, 10, 10, 10, 10, 10, 10, 9, 11]) # 80% 10, 10% 9, 10% 8

    x_values = np.arange(1, 31_540_000, step=86_400) # seconds in year by seconds in day
    # x_values = np.arange(1, 86_400 * 2, step=86_400) # seconds in year by seconds in day
    y_values = sample_size_bytes * sample_rate_hz * sensors_sending_data.mean() * x_values

    e_values = []
    for i in range(len(x_values)):
        print("(%f / %d) * |%f| = %F" % (sensors_sending_data.std(), x_values[i], np.abs(y_values[i]), (sensors_sending_data.std() / x_values[i]) * np.abs(y_values[i])))
        # e_values.append((sensors_sending_data.std() / x_values[i]) * np.abs(y_values[i]))
        e_values.append((1000000.0 / x_values[i]) * np.abs(y_values[i]))
    e_values = np.array(e_values)

    plt.plot(x_values, y_values)
    plt.plot(x_values, y_values + e_values)
    plt.plot(x_values, y_values - e_values)
    plt.show()
