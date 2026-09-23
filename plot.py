import matplotlib.pyplot as plt


def plot_convergence(histories, output_path="convergence.png"):
    """histories: dict like {"GA": ga_history, "ACO": aco_history} —
    each value is a list of best-fitness-so-far, one entry per generation/iteration."""
    plt.figure(figsize=(8, 5))

    for label, history in histories.items():
        plt.plot(history, label=label)

    plt.xlabel("Generation / Iteration")
    plt.ylabel("Best fitness so far")
    plt.title("GA vs ACO Convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(output_path)
    plt.close()