import random
from src.models import Individual
from src.problem import fitness



def random_chromosome(problem):
    chromosome = [0 for i in range(len(problem.vms))]
    server_range = len(problem.servers) - 1
    
    for i in range(len(problem.vms)):
        server = random.randint(0, server_range);
        chromosome[i] = server;
        
    return chromosome
    
def create_population(problem, size, weights: dict):
    population = list();
    for i in range(size):
        chromosome = random_chromosome(problem);
        score = fitness(problem, chromosome, weights);
        individual = Individual(chromosome, score);
        population.append(individual);
        
    return population

    
def tournament_selection(population, tournament_size):
    winner = population[random.randint(0, len(population) - 1)];
    winner_score = winner.fitness;
    
    for i in range(tournament_size - 1):
        participant = population[random.randint(0, len(population) - 1)];

        if participant.fitness < winner_score:
            winner = participant;
            winner_score = participant.fitness
                
    return winner;
    
    
def crossover(parent1, parent2):
    child_chromosome = list();
    for i in range(0, len(parent1.chromosome)):
        if i % 2 == 0:
            child_chromosome.append(parent1.chromosome[i]);
        else:
            child_chromosome.append(parent2.chromosome[i]);

    
    return child_chromosome;
    
    
def mutate(chromosome, mutation_rate, num_servers):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = random.randint(0, num_servers - 1)
    return chromosome
    
    
def build_new_population(problem, population, weights, tournament_size, mutation_rate):
    best = min(population, key=lambda ind: ind.fitness)
    new_population = [best] 
    while len(new_population) < len(population):
        parent1 = tournament_selection(population, tournament_size)
        parent2 = tournament_selection(population, tournament_size)

        child_chromosome = crossover(parent1, parent2)
        child_chromosome = mutate(child_chromosome, mutation_rate, len(problem.servers))

        child_score = fitness(problem, child_chromosome, weights)
        new_population.append(Individual(child_chromosome, child_score))

    return new_population
    
def ga(problem, weights, size, tournament_size, mutation_rate, max_generations, patience=20):
    population = create_population(problem, size, weights);
    best_ever = None;
    history = list()
    stagnant_generations = 0    
    best_generation = 0

    for generation in range(max_generations):
        new_population = build_new_population(problem, population, weights, tournament_size, mutation_rate);

        current_best = min(new_population, key=lambda ind: ind.fitness)
        history.append(best_ever.fitness if best_ever else current_best.fitness)

        if best_ever is None or current_best.fitness < best_ever.fitness:
            best_ever = current_best
            stagnant_generations = 0
            best_generation = generation
        else:
            stagnant_generations += 1

        if stagnant_generations >= patience:
            break

        population = new_population
        
    return best_ever, history, best_generation;
    
    
        
        
        