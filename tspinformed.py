import functools
import math
import numpy as np
import random

import utils

POP_SIZE = 100 # population size
MAX_GEN = 500 # maximum number of generations
CX_PROB = 0.8 # crossover probability
MUT_PROB = 0.2 # mutation probability
MUT_MAX_LEN = 10 # maximum lenght of the swapped part
REPEATS = 10 # number of runs of algorithm (should be at least 10)
INPUT = 'inputs/tsp_std.in' # the input file
OUT_DIR = 'tsp' # output directory for logs
EXP_ID = 'Erx' # the ID of this experiment (used to create log names)

# reads the input set of values of objects
def read_locations(filename):
    locations = []
    with open(filename) as f:
        for l in f.readlines():
            tokens = l.split(' ')
            locations.append((float(tokens[0]), float(tokens[1])))
    return locations

@functools.lru_cache(maxsize=None) # this enables caching of the values
def distance(loc1, loc2):
    # based on https://stackoverflow.com/questions/15736995/how-can-i-quickly-estimate-the-distance-between-two-latitude-longitude-points
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(math.radians, [loc1[1], loc1[0], loc2[1], loc2[0]])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    # Radius of earth in kilometers is 6371
    km = 6371.01 * c
    return km

def fitness(ind, cities):

    num_cities = len(cities)
    assert len(ind) == num_cities
    assert sum(ind) == num_cities*(num_cities - 1)//2

    dist = 0
    for a, b in zip(ind, ind[1:]):
        dist += distance(cities[a], cities[b])

    dist += distance(cities[ind[-1]], cities[ind[0]])

    return utils.FitObjPair(fitness=-dist, 
                            objective=dist)
"normalized fitness added"
def normalized_fitness(ind, cities):
    max_dist = sum(distance(cities[a], cities[b]) for a, b in zip(ind, ind[1:]))
    max_dist += distance(cities[ind[-1]], cities[ind[0]])
    dist = sum(distance(cities[a], cities[b]) for a, b in zip(ind, ind[1:]))
    dist += distance(cities[ind[-1]], cities[ind[0]])
    fitness_value = 1 - (dist / max_dist)  
    return utils.FitObjPair(fitness=fitness_value, objective=dist)

def create_ind(ind_len):
    ind = list(range(ind_len))
    random.shuffle(ind)
    return ind

def create_pop(pop_size, create_individual):
    return [create_individual() for _ in range(pop_size)]

def tournament_selection(pop, fits, k):
    selected = []
    for _ in range(k):
        p1 = random.randrange(0, len(pop))
        p2 = random.randrange(0, len(pop))
        if fits[p1] > fits[p2]:
            selected.append(pop[p1][:])
        else:
            selected.append(pop[p2][:])

    return selected

def order_cross(p1, p2):
    point1 = random.randrange(1, len(p1))
    point2 = random.randrange(1, len(p1))
    start = min(point1, point2)
    end = max(point1, point2)
    o1mid = p2[start:end]
    o2mid = p1[start:end]

    restp1 = [c for c in p1[end:] + p1[:end] if c not in o1mid]
    restp2 = [c for c in p2[end:] + p2[:end] if c not in o2mid]

    o1 = restp1[-start:] + o1mid + restp1[:-start]
    o2 = restp2[-start:] + o2mid + restp2[:-start]

    return o1, o2
def pmx_crossover(p1, p2):
    start, end = sorted(random.sample(range(len(p1)), 2))
    o1, o2 = p1[:], p2[:]
    for i in range(start, end):
        o1[o1.index(p2[i])], o1[i] = o1[i], p2[i]
        o2[o2.index(p1[i])], o2[i] = o2[i], p1[i]
    return o1, o2

# Cycle Crossover (CX)
def cycle_crossover(p1, p2):
    size = len(p1)
    o1, o2 = [-1] * size, [-1] * size
    cycle = 0
    while -1 in o1:
        idx = o1.index(-1)
        start = idx
        while True:
            o1[idx] = p1[idx] if cycle % 2 == 0 else p2[idx]
            o2[idx] = p2[idx] if cycle % 2 == 0 else p1[idx]
            idx = p1.index(p2[idx])
            if idx == start:
                break
        cycle += 1
    return o1, o2
# Nearest Neighbor Initialization
def nearest_neighbor_initialization(ind_len, locations):
    start = random.choice(range(ind_len))  
    tour = [start]
    unvisited = set(range(ind_len)) - {start}
    while unvisited:
        nearest = min(unvisited, key=lambda x: distance(locations[tour[-1]], locations[x]))
        tour.append(nearest)
        unvisited.remove(nearest)
    return tour

# Greedy Mutation
def greedy_mutation(ind, locations):
    a, b = sorted(random.sample(range(len(ind)), 2))  
    subpath = ind[a:b] 
    best_path = min([subpath, subpath[::-1]], key=lambda path: sum(
        distance(locations[path[i]], locations[path[i + 1]]) for i in range(len(path) - 1)))
    ind[a:b] = best_path
    return ind

# Edge Recombination Crossover (ERX)
def erx_crossover(p1, p2):

    adjacency_list = {i: set() for i in range(len(p1))}
    for i in range(len(p1)):
        adjacency_list[p1[i]].update({p1[i - 1], p1[(i + 1) % len(p1)]})
        adjacency_list[p2[i]].update({p2[i - 1], p2[(i + 1) % len(p2)]})

    offspring = []
    current = random.choice(p1)  

    while len(offspring) < len(p1):
        offspring.append(current)
        for neighbor in adjacency_list:
            adjacency_list[neighbor].discard(current)

        next_candidates = adjacency_list[current]
        if next_candidates:
            current = min(next_candidates, key=lambda x: len(adjacency_list[x]))
        else:
            
            unvisited = [city for city in range(len(p1)) if city not in offspring]
            if not unvisited:
                break  # All cities are already in the offspring
            current = random.choice(unvisited)

    return offspring, offspring


def swap_mutate(p, max_len):
    source = random.randrange(1, len(p) - 1)
    dest = random.randrange(1, len(p))
    lenght = random.randrange(1, min(max_len, len(p) - source))

    o = p[:]
    move = p[source:source+lenght]
    o[source:source + lenght] = []
    if source < dest:
        dest = dest - lenght 
    
    o[dest:dest] = move
    
    return o
"inversion mutuation added "
def inversion_mutation(ind):
    start, end = sorted(random.sample(range(len(ind)), 2))
    ind[start:end] = reversed(ind[start:end])
    return ind

def mate(pop, operators):
    for o in operators:
        pop = o(pop)
    return pop

def crossover(pop, cross, cx_prob):
    off = []
    for p1, p2 in zip(pop[0::2], pop[1::2]):
        if random.random() < cx_prob:
            o1, o2 = cross(p1, p2)
        else:
            o1, o2 = p1[:], p2[:]
        off.append(o1)
        off.append(o2)
    return off

def mutation(pop, mutate, mut_prob):
    return [mutate(p) if random.random() < mut_prob else p[:] for p in pop]

# implements the evolutionary algorithm
# arguments:
#   pop_size  - the initial population
#   max_gen   - maximum number of generation
#   fitness   - fitness function (takes individual as argument and returns 
#               FitObjPair)
#   operators - list of genetic operators (functions with one arguments - 
#               population; returning a population)
#   mate_sel  - mating selection (funtion with three arguments - population, 
#               fitness values, number of individuals to select; returning the 
#               selected population)
#   map_fn    - function to use to map fitness evaluation over the whole 
#               population (default `map`)
#   log       - a utils.Log structure to log the evolution run
def evolutionary_algorithm(pop, max_gen, fitness, operators, mate_sel, *, map_fn=map, log=None):
    evals = 0
    for G in range(max_gen):
        fits_objs = list(map_fn(fitness, pop))
        evals += len(pop)
        if log:
            log.add_gen(fits_objs, evals)
        fits = [f.fitness for f in fits_objs]
        objs = [f.objective for f in fits_objs]

        mating_pool = mate_sel(pop, fits, POP_SIZE)
        offspring = mate(mating_pool, operators)

        pop = offspring[:-1] + [max(list(zip(fits, pop)), key = lambda x: x[0])[1]]

    return pop

if __name__ == '__main__':
    locations = read_locations(INPUT)

    cr_ind = functools.partial(create_ind, ind_len=len(locations))
    cr_ind_nn = functools.partial(nearest_neighbor_initialization, ind_len=len(locations), locations=locations)

    fit = functools.partial(fitness, cities=locations)
    norm_fit = functools.partial(normalized_fitness, cities=locations)

    xover_default = functools.partial(crossover, cross=order_cross, cx_prob=CX_PROB)
    xover_pmx = functools.partial(crossover, cross=pmx_crossover, cx_prob=CX_PROB)
    xover_cx = functools.partial(crossover, cross=cycle_crossover, cx_prob=CX_PROB)
    xover_erx = functools.partial(crossover, cross=erx_crossover, cx_prob=CX_PROB)

    mut_swap = functools.partial(mutation, mut_prob=MUT_PROB, mutate=functools.partial(swap_mutate, max_len=MUT_MAX_LEN))
    mut_inversion = functools.partial(mutation, mut_prob=MUT_PROB, mutate=inversion_mutation)
    mut_greedy = functools.partial(mutation, mut_prob=MUT_PROB, mutate=functools.partial(greedy_mutation, locations=locations))

    experiments = [
        {
            'id': 'default',
            'xover': xover_default,
            'mut': mut_swap,
            'fit': fit,
            'cr_ind': cr_ind,
        },
        {
            'id': 'pmx',
            'xover': xover_pmx,
            'mut': mut_swap,
            'fit': fit,
            'cr_ind': cr_ind,
        },
        {
            'id': 'cx',
            'xover': xover_cx,
            'mut': mut_swap,
            'fit': fit,
            'cr_ind': cr_ind,
        },
        {
            'id': 'inversion',
            'xover': xover_default,
            'mut': mut_inversion,
            'fit': fit,
            'cr_ind': cr_ind,
        },
        {
            'id': 'normalized',
            'xover': xover_default,
            'mut': mut_swap,
            'fit': norm_fit,
            'cr_ind': cr_ind,
        },
        {
            'id': 'erx',
            'xover': xover_erx,
            'mut': mut_greedy,  # Use greedy mutation for ERX
            'fit': fit,
            'cr_ind': cr_ind_nn,
        },
    ]

    import multiprocessing
    pool = multiprocessing.Pool(processes=4)  # Use only 4 processes


    import matplotlib.pyplot as plt

    results = {}
    for experiment in experiments:
        EXP_ID = experiment['id']
        fit_func = experiment['fit']
        xover_func = experiment['xover']
        mut_func = experiment['mut']
        create_func = experiment['cr_ind']

        best_inds = []
        for run in range(REPEATS):
            log = utils.Log(OUT_DIR, EXP_ID, run, write_immediately=True, print_frequency=5)
            pop = create_pop(POP_SIZE, create_func)
            pop = evolutionary_algorithm(pop, MAX_GEN, fit_func, [xover_func, mut_func], tournament_selection, map_fn=pool.map, log=log)
            bi = max(pop, key=fit_func)
            best_inds.append(bi)

            # Save the best individual and KML file
            best_template = '{individual}'
            with open('resources/kmltemplate.kml') as f:
                best_template = f.read()

            with open(f'{OUT_DIR}/{EXP_ID}_{run}.best', 'w') as f:
                f.write(str(bi))

            with open(f'{OUT_DIR}/{EXP_ID}_{run}.best.kml', 'w') as f:
                bi_kml = [f'{locations[i][1]},{locations[i][0]},5000' for i in bi]
                bi_kml.append(f'{locations[bi[0]][1]},{locations[bi[0]][0]},5000')
                f.write(best_template.format(individual='\n'.join(bi_kml)))

        results[EXP_ID] = [fit_func(bi).objective for bi in best_inds]

        
        for i, bi in enumerate(best_inds):
            print(f'Run {i} for {EXP_ID}: difference = {fit_func(bi).objective}')

        
        utils.summarize_experiment(OUT_DIR, EXP_ID)

    
    plt.figure(figsize=(12, 8))
    for exp_id, diffs in results.items():
        evals, lower, mean, upper = utils.get_plot_data(OUT_DIR, exp_id)
        utils.plot_experiment(evals, lower, mean, upper, legend_name=f'{exp_id.capitalize()} Settings')
    plt.legend()
    plt.title('Comparison of Default, PMX, CX, Inversion, Normalized, and ERX Settings')
    plt.show()




    # you can also plot mutiple experiments at the same time using 
    # utils.plot_experiments, e.g. if you have two experiments 'default' and 
    # 'tuned' both in the 'partition' directory, you can call
    # utils.plot_experiments('partition', ['default', 'tuned'], 
    #                        rename_dict={'default': 'Default setting'})
    # the rename_dict can be used to make reasonable entries in the legend - 
    # experiments that are not in the dict use their id (in this case, the 
    # legend entries would be 'Default settings' and 'tuned') 
