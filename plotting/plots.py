import matplotlib.pyplot as plt

def plot_iml_level_no_opt(sample_size: int,
                          sample_rate_hz: int,
                          window_length_s: int):

    x_values = list(range(window_length_s))
    y_values = [sample_size * sample_rate_hz * x for x in x_values]

    plt.figure(figsize=(12, 5))
    plt.plot(x_values, y_values)
    plt.title("IML No Opt: Sample Size Bytes=%d, Sample Rate Hz=%d, Window Length S=%d" % (
        sample_size,
        sample_rate_hz,
        window_length_s
    ))
    plt.xlabel("Time (S)")
    plt.ylabel("Bytes")
    plt.savefig("plot_iml_level_no_opt.png")
    plt.show()


if __name__ == "__main__":
    plot_iml_level_no_opt(5, 10, 10)
