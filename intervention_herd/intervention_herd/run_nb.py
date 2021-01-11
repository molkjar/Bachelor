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
lifting      = '2020-10-27' #250 days

############# Model Setup #################
# Which network to use - Generate with 'make_rand_pop.py'
popfile = 'randppl_disp.pop'

# Model parameters
pars = sc.objdict(
        pop_size     = 200e3,
        pop_infected = 100, 
     
        beta         = 0.004, # Calibrated such that R0=2.5 with dispersed network
    
        start_day    = start_day,
        end_day      = end_day,
    
        rand_seed    = 271220,
    
        verbose      = 0,
        )


############ Interventions ################
## Make interventions - Quite simple in this form. A scaling of beta.
## int_level=1 - No interventions
## int_level=0 - Complete stop of transmission
def make_ints(int_level):
    interventions = [cv.change_beta(days    = [restrictions, lifting],
                                    changes = [int_level, 1])]
    
    return interventions


############## Simulation/calibration setup ############
## Initialize simulation with intervention
def make_sim(pars, int_level=1, load_pop=True, popfile=popfile, betanoise=None):
    sim = cv.Sim(pars=pars,
                 popfile=popfile,
                 load_pop=load_pop)
    
    if betanoise is not None: sim.pars['beta'] += betanoise
    
    if int_level != 1: sim.pars['interventions'] = make_ints(int_level = int_level)
    
    sim.initialize()
    
    return sim

## Running simulation
def run_sim(pars, int_level, popfile=popfile, return_stat=False):
    sim = make_sim(pars=pars, int_level=int_level, popfile=popfile)
    sim.run()
    
    if return_stat:
        stat = sim.results['cum_infections'][-1]
        return stat
    else:
        return sim
    
    
    

############## Further analyses and plotting ###############
## Running for several seeds
N = 25
#basesim = make_sim(pars=pars, int_level=0.63)
#msim = cv.MultiSim(basesim)
#msim.run(n_runs=N, noise=0, keep_people=True, verbose=.1, n_cpus=5)
#msim.median(quantiles=[0.025, 0.975])
#msim.save("msim_at_herd25.msim")

msim = cv.load("msim_at_herd25.msim")


## Check that there's still infectious individuals at lifting
for sim in msim.sims:
    print(sim.results['n_infectious'][250])

    
## quantiles of final size over different seeds
cum_inf = [0]*N
ind = 0
for sim in msim.sims:
    cum_inf[ind] = sim.results['cum_infectious'][-1]
    ind += 1

np.quantile(cum_inf, [0.025, 0.5, 0.975])
np.quantile(cum_inf, [0.025, 0.5, 0.975])/200e3


## Run with noise on int_level
noise = 0.05
noiseval = noise*np.random.normal(size=N)

sims = [make_sim(pars=pars, int_level=0.63+i) for i in noiseval]
msim_intnoise = cv.MultiSim(sims)    
msim_intnoise.run(keep_people=True)
msim_intnoise.plot()




## Run with noise on beta
noise = 0.001
noiseval = noise*np.random.normal(size=N)

sims = [make_sim(pars=pars, int_level=0.69, betanoise=i) for i in noiseval]

msim_betanoise = cv.MultiSim(sims)
msim_betanoise.run(keep_people=True, verbose=.1)
msim_betanoise.plot()







## Running free of interventions
#simf = make_sim(pars=pars, int_level=1)
#msimf = cv.MultiSim(simf)
#msimf.run(n_runs=N, noise=0, keep_people=True, verbose=.1, n_cpus=5)
#msimf.median(quantiles=[0.025, 0.975])
#msimf.save("msim_free25.msim")
msimf = cv.load("msim_free25.msim")

## Running with to hars interventions --> Second wave
#simsw = make_sim(pars=pars, int_level=0.55)
#msimsw = cv.MultiSim(simsw)
#msimsw.run(n_runs=N, noise=0, keep_people=True, verbose=.1, n_cpus=5)
#msimsw.median(quantiles=[0.025, 0.975])
#msimsw.save("msim_sec_wave25.msim")
msimsw = cv.load("msim_sec_wave25.msim")
