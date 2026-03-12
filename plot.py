import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Load traces
# =========================

zipf = pd.read_csv("/Users/adhariya/src/shardingsystemresilience/results/timeseries_zipf_full_2000.csv")
uniform = pd.read_csv("/Users/adhariya/src/shardingsystemresilience/results/timeseries_uniform_full_2000.csv")

print("Zipf columns:", zipf.columns)
print("Uniform columns:", uniform.columns)

# correct time column
time_col = "t"
import os
os.makedirs("results", exist_ok=True)

# ======================
# plotting helper
# ======================

def plot_metric(metric, ylabel, filename):

    if metric not in zipf.columns:
        print(f"Skipping {metric} (not found)")
        return

    plt.figure(figsize=(8,5))

    plt.plot(zipf[time_col], zipf[metric], label="Zipf workload", linewidth=2)
    plt.plot(uniform[time_col], uniform[metric], label="Uniform workload", linewidth=2)

    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(f"Time vs {ylabel}")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"results/{filename}.png", dpi=300)
    plt.show()


# ======================
# plots
# ======================

plot_metric("imbalance", "Imbalance", "time_vs_imbalance")

plot_metric("fanout", "Fanout", "time_vs_fanout")

plot_metric("fanout_var", "Fanout Variance", "time_vs_fanout_variance")

plot_metric("fanout_req_mean", "Workload (requests per step)", "time_vs_workload")