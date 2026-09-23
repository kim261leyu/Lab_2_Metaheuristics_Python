import random
from src.models import Individual
from src.problem import fitness

def init_pheromone(problem):
    return [[1.0] * len(problem.servers) for _ in range(len(problem.vms))]
    
def pick_server(problem, vm_cpu, vm_ram, cpu_remaining, ram_remaining, pheromone, alpha, beta):
    scores = list()
    
    for i in range(len(problem.servers)):
        fitness = fit(vm_cpu, vm_ram, cpu_remaining[i], ram_remaining[i]);
        score = pheromone[i]**alpha * fitness**beta;
        scores.append(score);
        
    choice = random.choices(range(len(problem.servers)), scores, k=1);
    
    return choice[0]
    
    
def fit(vm_cpu, vm_ram, server_cpu_remaining, server_ram_remaining):
    if server_cpu_remaining <= 0 or server_ram_remaining <= 0:
        return 0.0001
        
    cpu_ratio = vm_cpu / server_cpu_remaining;
    ram_ratio = vm_ram / server_ram_remaining;
    
    if cpu_ratio > 1 or ram_ratio > 1:
        return 0.0001
        
    return (cpu_ratio + ram_ratio) / 2

def build_ant_solution(problem, pheromones, aco_weights):
    cpu_remaining = [i.cpu_capacity for i in problem.servers];
    ram_remaining = [i.ram_capacity for i in problem.servers];
    
    chromosome = list();
    
    for i, vm in enumerate(problem.vms):
        server = pick_server(problem, vm.cpu_usage, vm.ram_usage, cpu_remaining, ram_remaining, pheromones[i], aco_weights["alpha"], aco_weights["beta"]);
        chromosome.append(server);
        
        cpu_remaining[server] -= vm.cpu_usage;
        ram_remaining[server] -= vm.ram_usage;
        
    return chromosome
    
def run_colony(problem, pheromones, aco_weights, weights, colony_size):
    ant_results = list();
    
    for i in range(colony_size):
        chromosome = build_ant_solution(problem, pheromones, aco_weights);
        score = fitness(problem, chromosome, weights);
        ant_result = Individual(chromosome, score);
        ant_results.append(ant_result);
        
    return ant_results;
    
def update_pheromone(pheromones, ants, evaporation_rate):
    for i in range(len(pheromones)):
        for j in range(len(pheromones[0])):
            pheromones[i][j] *= 1 - evaporation_rate;
            
    best_ant_result = min(ants, key=lambda ant: ant.fitness); # theres chromosome and fitness, so in chromosome indexes is vms and values is servers
    
    for vm, server in enumerate(best_ant_result.chromosome):
        pheromones[vm][server] += 1 / best_ant_result.fitness;
    
    
def aco(problem, weights, aco_weights, colony_size, max_iterations, evaporation_rate, patience=20):
    pheromones = init_pheromone(problem);

    best_ever = None;
    history = list()
    stagnant_generations = 0    
    best_iteration = 0

    for i in range(max_iterations):
        ants = run_colony(problem, pheromones, aco_weights, weights, colony_size)
        current_best = min(ants, key=lambda ant: ant.fitness);
        if best_ever is None or current_best.fitness < best_ever.fitness:
            best_ever = current_best;
            stagnant_generations = 0;
            best_iteration = i;
        else:
            stagnant_generations += 1;
            
        history.append(best_ever.fitness if best_ever else current_best.fitness);    
        
        if stagnant_generations >= patience:
            break
        update_pheromone(pheromones, ants, evaporation_rate);
        
    return best_ever, history, best_iteration;
    
        
        

