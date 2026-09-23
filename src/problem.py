import random
from math import sqrt
 
from .models import VM, Server, ProblemCluster
 
 
def generate_problem(
    num_vms,
    num_servers,
    server_cpu_capacity,
    server_ram_capacity,
    vm_cpu_range,
    vm_ram_range,
    vm_duration_range,
    seed=None,
):

    if seed is not None:
        random.seed(seed)
 
    servers = [
        Server(cpu_capacity=server_cpu_capacity, ram_capacity=server_ram_capacity)
        for _ in range(num_servers)
    ]
 
    vms = [
        VM(
            cpu_usage=random.randint(*vm_cpu_range),
            ram_usage=random.randint(*vm_ram_range),
            duration=random.randint(*vm_duration_range),
        )
        for _ in range(num_vms)
    ]
 
    return ProblemCluster(vms=vms, servers=servers)



def decode(problem, chromosome):
    cpu_used = [0] * len(problem.servers);
    ram_used = [0] * len(problem.servers);
    
    for vm_index, server_index in enumerate(chromosome):
            cpu_used[server_index] += problem.vms[vm_index].cpu_usage;
            ram_used[server_index] += problem.vms[vm_index].ram_usage;
            
    return cpu_used, ram_used
    
 
def fitness(problem, chromosome, weights):
    cpu_used, ram_used = decode(problem, chromosome);
    
    cpu_ratios = [0 for i in range(len(problem.servers))];
    ram_ratios = [0 for i in range(len(problem.servers))];
    
    used_servers = 0;
    penalty = 0;
    for i in range(len(problem.servers)):
        if cpu_used[i] > 0:
            cpu_ratios[i] = cpu_used[i] / problem.servers[i].cpu_capacity;
        if ram_used[i] > 0:
            ram_ratios[i] = ram_used[i] / problem.servers[i].ram_capacity;
        if ram_used[i] > 0 or cpu_used[i] > 0:
            used_servers += 1;
        cpu_violation = max(0, cpu_used[i] - problem.servers[i].cpu_capacity)
        ram_violation = max(0, ram_used[i] - problem.servers[i].ram_capacity)
        penalty += cpu_violation ** 2 + ram_violation ** 2;
        
    active_cpu_ratios = [cpu_ratios[i] for i in range(len(problem.servers)) if ram_used[i] > 0 or cpu_used[i] > 0]
    active_ram_ratios = [ram_ratios[i] for i in range(len(problem.servers)) if ram_used[i] > 0 or cpu_used[i] > 0]
    
    mean_cpu = sum(active_cpu_ratios) / len(active_cpu_ratios);
    mean_ram = sum(active_ram_ratios) / len(active_ram_ratios);
    
    dev_cpu = [(mean_cpu - active_cpu_ratios[i]) ** 2 for i in range(len(active_cpu_ratios))];
    dev_ram = [(mean_ram - active_ram_ratios[i]) ** 2 for i in range(len(active_ram_ratios))];
    
    std_dev_cpu = sqrt(sum(dev_cpu) / len(dev_cpu));
    std_dev_ram = sqrt(sum(dev_ram) / len(dev_ram));
    
    std_dev = (std_dev_cpu + std_dev_ram) / 2;
    
    active_servers = used_servers / len(problem.servers);
    
    score = std_dev * weights["balance"] + active_servers * weights["consolidation"] +  penalty * weights["penalty"];
    
    return score
    
def server_busy_time(problem, chromosome, contention_factor):
    busy_time_sum = [0] * len(problem.servers);
    busy_time_max = [0] * len(problem.servers);
    
    for vm_index, server_index in enumerate(chromosome):
        busy_time_max[server_index] = max(busy_time_max[server_index], problem.vms[vm_index].duration)
        busy_time_sum[server_index] += problem.vms[vm_index].duration
        
    busy_time = [0] * len(problem.servers)
    for i in range(len(problem.servers)):
        busy_time[i] = busy_time_max[i] + contention_factor * (busy_time_sum[i] - busy_time_max[i])
        
    return busy_time
    
def compute_makespan(problem, chromosome, contention_factor):
    busy_times = server_busy_time(problem, chromosome, contention_factor)
    return max(busy_times)
 
def compute_energy(problem, chromosome, idle_power_watts, max_power_watts, contention_factor):
    cpu_used, _ = decode(problem, chromosome)
    busy_times = server_busy_time(problem, chromosome, contention_factor)

    total_energy = 0
    for i in range(len(problem.servers)):
        if cpu_used[i] > 0:
            utilization = cpu_used[i] / problem.servers[i].cpu_capacity
            power = idle_power_watts + (max_power_watts - idle_power_watts) * utilization
            total_energy += power * busy_times[i]

    return total_energy
