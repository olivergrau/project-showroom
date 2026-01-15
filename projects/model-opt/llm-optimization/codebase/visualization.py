import numpy as np
import matplotlib.pyplot as plt


def plot_speculative_stats(spec_stats):
    """
    Visualize speculative decoding results returned by run_speculative_trials().
    Produces four plots:
      1) Latency (baseline vs k)
      2) Throughput (baseline vs k)
      3) ROUGE-1 (baseline vs k)
      4) ROUGE-L (baseline vs k)
    Each plot is a boxplot with means overlaid as points.
    """
    def _boxplot_with_means(ax, data_list, labels, ylabel, title):
        ax.boxplot(data_list, labels=labels, showfliers=True)
        means = [float(np.mean(d)) for d in data_list]
        x = np.arange(1, len(means) + 1)
        ax.plot(x, means, marker="o", linestyle="None")  # default color
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.7)

    # Prepare ordered data: baseline first, then k sorted
    k_values = list(spec_stats["assistant_k_values"])
    k_values_sorted = sorted(k_values)

    labels = ["baseline"] + [f"spec k={k}" for k in k_values_sorted]

    # --- Latency ---
    lat_data = [np.asarray(spec_stats["baseline_latency"], dtype=np.float64)]
    lat_data += [np.asarray(spec_stats["speculative"][k]["latency"], dtype=np.float64) for k in k_values_sorted]

    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    _boxplot_with_means(
        ax=ax,
        data_list=lat_data,
        labels=labels,
        ylabel="Latency (s)",
        title="Speculative Decoding: Latency per Trial (baseline vs k)"
    )
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()

    # --- Throughput ---
    tps_data = [np.asarray(spec_stats["baseline_throughput"], dtype=np.float64)]
    tps_data += [np.asarray(spec_stats["speculative"][k]["throughput"], dtype=np.float64) for k in k_values_sorted]

    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    _boxplot_with_means(
        ax=ax,
        data_list=tps_data,
        labels=labels,
        ylabel="Throughput (tokens/s)",
        title="Speculative Decoding: Throughput per Trial (baseline vs k)"
    )
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()

    # --- ROUGE-1 ---
    r1_data = [np.asarray(spec_stats["baseline_rouge1"], dtype=np.float64)]
    r1_data += [np.asarray(spec_stats["speculative"][k]["rouge1"], dtype=np.float64) for k in k_values_sorted]

    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    _boxplot_with_means(
        ax=ax,
        data_list=r1_data,
        labels=labels,
        ylabel="ROUGE-1",
        title="Speculative Decoding: ROUGE-1 per Trial (baseline vs k)"
    )
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()

    # --- ROUGE-L ---
    rL_data = [np.asarray(spec_stats["baseline_rougeL"], dtype=np.float64)]
    rL_data += [np.asarray(spec_stats["speculative"][k]["rougeL"], dtype=np.float64) for k in k_values_sorted]

    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    _boxplot_with_means(
        ax=ax,
        data_list=rL_data,
        labels=labels,
        ylabel="ROUGE-L",
        title="Speculative Decoding: ROUGE-L per Trial (baseline vs k)"
    )
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()


def plot_run_trials_stats(stats):
    """
    Visualize repeated paired trial results returned by run_trials().

    Produces two 2×2 plot blocks:
      1) Latency + Throughput (absolute + delta)
      2) ROUGE-1 + ROUGE-L (absolute + delta)
    """

    # ========= Block 1: Latency + Throughput =========
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Latency and Throughput Across Trials", fontsize=14)

    # --- Latency (boxplot) ---
    axes[0, 0].boxplot(
        [stats["baseline_latency"], stats["kv_latency"]],
        labels=["Baseline", "KV cache"],
    )
    axes[0, 0].set_title("Mean Latency per Trial")
    axes[0, 0].set_ylabel("Latency (s)")
    axes[0, 0].grid(True)

    # --- Latency delta ---
    latency_delta = stats["baseline_latency"] - stats["kv_latency"]
    axes[0, 1].plot(latency_delta, marker="o")
    axes[0, 1].axhline(0)
    axes[0, 1].set_title("Latency Improvement per Trial (Baseline − KV)")
    axes[0, 1].set_xlabel("Trial")
    axes[0, 1].set_ylabel("Latency Delta (s)")
    axes[0, 1].grid(True)

    # --- Throughput (boxplot) ---
    axes[1, 0].boxplot(
        [stats["baseline_throughput"], stats["kv_throughput"]],
        labels=["Baseline", "KV cache"],
    )
    axes[1, 0].set_title("Throughput per Trial")
    axes[1, 0].set_ylabel("Tokens / second")
    axes[1, 0].grid(True)

    # --- Throughput delta ---
    throughput_delta = stats["kv_throughput"] - stats["baseline_throughput"]
    axes[1, 1].plot(throughput_delta, marker="o")
    axes[1, 1].axhline(0)
    axes[1, 1].set_title("Throughput Improvement per Trial (KV − Baseline)")
    axes[1, 1].set_xlabel("Trial")
    axes[1, 1].set_ylabel("Throughput Delta (tokens/s)")
    axes[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # ========= Block 2: ROUGE =========
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("ROUGE Stability and Differences Across Trials", fontsize=14)

    # --- ROUGE-1 (boxplot) ---
    axes[0, 0].boxplot(
        [stats["baseline_rouge1"], stats["kv_rouge1"]],
        labels=["Baseline", "KV cache"],
    )
    axes[0, 0].set_title("ROUGE-1 Across Trials")
    axes[0, 0].set_ylabel("ROUGE-1")
    axes[0, 0].grid(True)

    # --- ROUGE-1 delta ---
    rouge1_delta = stats["kv_rouge1"] - stats["baseline_rouge1"]
    axes[0, 1].plot(rouge1_delta, marker="o")
    axes[0, 1].axhline(0)
    axes[0, 1].set_title("ROUGE-1 Difference per Trial (KV − Baseline)")
    axes[0, 1].set_xlabel("Trial")
    axes[0, 1].set_ylabel("ROUGE-1 Delta")
    axes[0, 1].grid(True)

    # --- ROUGE-L (boxplot) ---
    axes[1, 0].boxplot(
        [stats["baseline_rougeL"], stats["kv_rougeL"]],
        labels=["Baseline", "KV cache"],
    )
    axes[1, 0].set_title("ROUGE-L Across Trials")
    axes[1, 0].set_ylabel("ROUGE-L")
    axes[1, 0].grid(True)

    # --- ROUGE-L delta ---
    rougeL_delta = stats["kv_rougeL"] - stats["baseline_rougeL"]
    axes[1, 1].plot(rougeL_delta, marker="o")
    axes[1, 1].axhline(0)
    axes[1, 1].set_title("ROUGE-L Difference per Trial (KV − Baseline)")
    axes[1, 1].set_xlabel("Trial")
    axes[1, 1].set_ylabel("ROUGE-L Delta")
    axes[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
