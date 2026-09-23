import time

import yaml
from tabulate import tabulate
import tracemalloc


from src.models import Result
from src.problem import generate_problem, compute_makespan, compute_energy
from src.ga_solver import ga
from src.aco_solver import aco

import numpy as np
import matplotlib.pyplot as plt

IDLE_POWER_WATTS = 100
MAX_POWER_WATTS = 200

def pad(histories):
    L = max(len(h) for h in histories)
    return np.array([h + [h[-1]] * (L - len(h)) for h in histories])

def print_problem_details(problem_config, weights, aco_weights):
    print("  PROBLEM CONFIGURATION")

    rows = [
        ("Num VMs", problem_config["num_vms"]),
        ("Num Servers", problem_config["num_servers"]),
        ("Server CPU Capacity", problem_config["server_cpu_capacity"]),
        ("Server RAM Capacity", problem_config["server_ram_capacity"]),
        ("VM CPU Range", tuple(problem_config["vm_cpu_range"])),
        ("VM RAM Range", tuple(problem_config["vm_ram_range"])),
        ("VM Duration Range", tuple(problem_config["vm_duration_range"])),
        ("Seed", problem_config["seed"]),
    ]
    print(tabulate(rows, tablefmt="grid"))

    print("\n  Fitness Weights:")
    for k, v in weights.items():
        print(f"    {k}: {v}")

    print("\n  ACO Heuristic Weights:")
    for k, v in aco_weights.items():
        print(f"    {k}: {v}")
    print()



def print_results(problem, ga_result, aco_result, seed=None):
    rows = [
        (
            "ACO",
            f"{aco_result.peak_memory_kb:.2f}",
            f"{aco_result.steps_run:.2f}",
            f"{aco_result.runtime:.2f}",
            f"{aco_result.makespan:.2f}",
            f"{aco_result.energy:.2f}",
            f"{aco_result.active_servers}",
            f"{aco_result.best_fitness:.4f}",
        ),
        (
            "GA",
            f"{ga_result.peak_memory_kb:.2f}",
            f"{ga_result.steps_run:.2f}",
            f"{ga_result.runtime:.2f}",
            f"{ga_result.makespan:.2f}",
            f"{ga_result.energy:.2f}",
            f"{ga_result.active_servers}",
            f"{ga_result.best_fitness:.4f}",
        ),
    ]

    if seed is not None:
        print(f"\n--- Seed {seed} ---")

    print(tabulate(
        rows,
        headers=[
            "Algorithm",
            "Peak Memory (KB)",
            "Steps Run",
            "Runtime (s)",
            "Makespan (s)",
            "Energy (J)",
            "Active Servers",
            "Best Fitness",
        ],
        tablefmt="grid",
    ))


def run_solver(solver_fn, problem, idle_power_watts, max_power_watts, contention_factor, *args, **kwargs):
    tracemalloc.start()
    start = time.time()
    best, history, _ = solver_fn(problem, *args, **kwargs)
    runtime = time.time() - start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_kb = peak_bytes / 1024

    makespan = compute_makespan(problem, best.chromosome, contention_factor)
    energy = compute_energy(problem, best.chromosome, idle_power_watts, max_power_watts, contention_factor)
    steps = len(history)                     
    active_servers = len(set(best.chromosome)) 
    
    result = Result(
        best_chromosome=best.chromosome,
        best_fitness=best.fitness,
        fitness_history=history,
        runtime=runtime,
        makespan=makespan,
        energy=energy,
        peak_memory_kb=peak_kb,
        steps_run=steps,
        active_servers=active_servers
    )


    return result, history
    
def compute_average_results(results):
    n = len(results)

    best_fitness = 0
    runtime = 0
    makespan = 0
    energy = 0
    peak_memory_kb = 0
    steps_run = 0
    active_servers = 0

    for r in results:
        best_fitness += r.best_fitness
        runtime += r.runtime
        makespan += r.makespan
        energy += r.energy
        peak_memory_kb += r.peak_memory_kb
        steps_run += r.steps_run
        active_servers += r.active_servers

    return Result(
        best_chromosome=[],
        best_fitness=best_fitness / n,
        fitness_history=[],
        runtime=runtime / n,
        makespan=makespan / n,
        energy=energy / n,
        peak_memory_kb=peak_memory_kb / n,
        steps_run=steps_run / n,
        active_servers=active_servers / n,
    )

def has_overload(problem, chromosome):
    for server_id, server in enumerate(problem.servers):
        cpu = 0
        ram = 0

        for vm_id, assigned_server in enumerate(chromosome):
            if assigned_server == server_id:
                vm = problem.vms[vm_id]
                cpu += vm.cpu_usage
                ram += vm.ram_usage

        if cpu > server.cpu_capacity or ram > server.ram_capacity:
            return True

    return False

def main():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    problem_config = config["problem"]
    ga_config = config["ga"]
    aco_config = config["aco"]
    aco_weights = config["aco_weights"]
    weights = config["fitness_weights"]

    print_problem_details(problem_config, weights, aco_weights)

    ga_results = []
    aco_results = []
    ga_histories = []
    aco_histories = []

    for i in range(5):
        problem = generate_problem(
            num_vms=problem_config["num_vms"],
            num_servers=problem_config["num_servers"],
            server_cpu_capacity=problem_config["server_cpu_capacity"],
            server_ram_capacity=problem_config["server_ram_capacity"],
            vm_cpu_range=tuple(problem_config["vm_cpu_range"]),
            vm_ram_range=tuple(problem_config["vm_ram_range"]),
            vm_duration_range=tuple(problem_config["vm_duration_range"]),
            seed=problem_config["seed"][i],
        )

        seed = problem_config["seed"][i]

        ga_result, ga_history = run_solver(
            ga,
            problem,
            problem_config["idle_power_watts"],
            problem_config["max_power_watts"],
            problem_config["contention_factor"],
            weights,
            size=ga_config["population_size"],
            tournament_size=ga_config["tournament_size"],
            mutation_rate=ga_config["mutation_rate"],
            max_generations=ga_config["max_generations"],
            patience=ga_config["patience"],
        )
        ga_results.append(ga_result)
        ga_histories.append(ga_history)
        if has_overload(problem, ga_result.best_chromosome):
            print("ga overload")
        
        aco_result, aco_history = run_solver(
            aco,
            problem,
            problem_config["idle_power_watts"],
            problem_config["max_power_watts"],
            problem_config["contention_factor"],
            weights,
            aco_weights,
            colony_size=aco_config["colony_size"],
            max_iterations=aco_config["max_iterations"],
            evaporation_rate=aco_config["evaporation_rate"],
            patience=aco_config["patience"],
        )
        aco_results.append(aco_result)
        aco_histories.append(aco_history)
        if has_overload(problem, aco_result.best_chromosome):
            print("aco overload")
        
        print_results(
            problem,
            ga_result,
            aco_result,
            seed=seed
        )

            
    avg_ga = compute_average_results(ga_results);
    avg_aco = compute_average_results(aco_results)
    
    print_results(problem, avg_ga, avg_aco)
    
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, results, color in [("GA", ga_histories, "tab:blue"), ("ACO", aco_histories, "tab:orange")]:
        H = pad([r for r in results])
        for row in H:
            ax.plot(row, color=color, alpha=0.2)          # individual runs
        ax.plot(H.mean(axis=0), color=color, lw=2, label=f"{name} (mean of 5)")

    ax.set_xlabel("Generation / iteration")
    ax.set_ylabel("Best fitness (lower is better)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("pic.png", dpi=200)


if __name__ == "__main__":
    main()

