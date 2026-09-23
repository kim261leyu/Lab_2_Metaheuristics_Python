from dataclasses import dataclass, field

@dataclass
class VM:
    cpu_usage: int
    ram_usage: int
    duration: int 

@dataclass
class Server:
    cpu_capacity: int
    ram_capacity: int
    

@dataclass
class Result:
    best_chromosome: list[int]
    best_fitness: float
    fitness_history: list[float]
    runtime: float
    makespan: float
    energy: float
    peak_memory_kb: float
    steps_run: float
    active_servers: float

@dataclass
class ProblemCluster:
    vms: list[VM]
    servers: list[Server]
    

@dataclass
class Individual:
    chromosome: list[int]
    fitness: float
    
    
    
    