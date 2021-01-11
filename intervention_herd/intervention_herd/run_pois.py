import covasim as cv
import covasim.utils as cvu
import numpy as np
import sciris as sc
import pandas as pd
import os

############# Dates #######################

start_day    = '2020-03-01'
end_day      = '2021-10-01'

restrictions = '2020-03-01'
lifting      = '2021-01-15' #320 days

############# Model Setup #################
# Which network to use - Generate with 'make_rand_pop.py'
popfile = 'randppl.pop'

# Model parameters
pars = sc.objdict(
        pop_size     = 200e3,
        pop_infected = 100,  
     
        beta         = 0.005, # Calibrated such that R0=2.5 with poisson network
    
        start_day    = start_day,
        end_day      = end_day,
    
        rand_seed    = 271220,
    
        verbose      = 0,
        )


############ Interventions and Imports ################
## Make interventions - Quite simple in this form. A scaling of beta.
## int_level=1 - No interventions
## int_level=0 - Complete stop of transmission
def make_ints(int_level):
    interventions = [cv.change_beta(days    = [restrictions, lifting],
                                    changes = [int_level, 1])]
    
    return interventions


############## Simulation/calibration setup ############
## Statistic which need to be optimized
def stat_cum_inf(sim):
    cum_inf = sim.results['cum_infections'][-1]
        
    return cum_inf

## Initialize simulation with intervention
def make_sim(pars, int_level=1, load_pop=True, popfile=popfile):
    sim = cv.Sim(pars=pars,
                 popfile=popfile,
                 load_pop=load_pop)
    
    if int_level != 1: sim.pars['interventions'] = make_ints(int_level = int_level)
    
    sim.initialize()
    
    return sim

## Running simulation
def run_sim(pars, int_level, popfile=popfile, stat_fun=stat_cum_inf, return_stat=False, verbose=0.1):
    sim = make_sim(pars=pars, int_level=int_level, popfile=popfile)
    sim.run(verbose=verbose)
    
    stat = stat_fun(sim)
    
    if return_stat:
        return stat
    else:
        return sim
    

############## Analyses ###############
## Running for several seeds
N = 25
#basesim = make_sim(pars=pars, int_level=0.725)
#msim = cv.MultiSim(basesim)
#msim.run(n_runs=N, keep_people=True, verbose=.1, n_cpus=10)
#msim.median(quantiles=[0.025, 0.975])
#msim.save("msim_at_herd25_pois.msim")

msim = cv.load("msim_at_herd25_pois.msim")

## Check that there's still infectious individuals at lifting
for sim in msim.sims:
    print(sim.results['n_infectious'][320])

    
## Mean final size over different seeds
cum_inf = [0]*N
ind = 0
for sim in msim.sims:
    cum_inf[ind] = sim.results['cum_infectious'][-1]
    ind += 1

np.quantile(cum_inf, [0.025, 0.5, 0.975])
np.quantile(cum_inf, [0.025, 0.5, 0.975])/200e3
    
    
    
    
    
    
    
    


## Running free of interventions - mostly for plotting 
#simf = make_sim(pars=pars, int_level=1)
#msimf = cv.MultiSim(simf)
#msimf.run(n_runs=N, keep_people=True, verbose=.1, n_cpus=10)
#msimf.median(quantiles=[0.025, 0.975])
#msimf.save("msim_free_pois25.msim")

msimf = cv.load("msim_free_pois25.msim")

## Second wave - mostly for plotting
#simsw = make_sim(pars=pars, int_level=0.63)
#msimsw = cv.MultiSim(simsw)
#msimsw.run(n_runs=N, noise=0, keep_people=True, verbose=.1, n_cpus=10)
#msimsw.median(quantiles=[0.025, 0.975])
#msimsw.save("msim_sw_pois25.msim")

msimsw = cv.load("msim_sw_pois25.msim")
    


