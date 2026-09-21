"""Creates plots for comparing navigation methods."""

import matplotlib.pyplot as plt


def plot_comparison(summary, metric, title, ylabel, output_file):
    """Create a bar chart for one evaluation metric."""

    methods = list(summary.keys())
    values = [summary[method][metric] for method in methods]

    plt.figure(figsize=(8, 5))
    plt.bar(methods, values)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()

    plt.savefig(output_file)
    plt.close()