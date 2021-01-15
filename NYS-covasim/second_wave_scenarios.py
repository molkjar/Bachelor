import covasim as cv
import covasim.utils as cvu
import optuna as op
import sciris as sc
import pandas as pd
import numpy as np
import os
from collections import defaultdict
import population

## Interesting part starts around line 200
## First part is setup, optimization workers and alike - Important to run before analysing but not that interesting.

############# Dates #######################

start_day    = '2020-03-01'
end_day      = '2022-03-01'

''' NYSonPause:
    General closure, incorporate school closures (happened few days before)
    Shelter-in-place order etc.
    Everything non-essential closed
    Lifting: Chosen quite arbitrary although somewhat at the right time'''

NYSonPause     = '2020-03-22'
schoolsClosure = '2020-03-16'

lifting        = '2020-07-20'
liftingSW      = '2021-08-01'

############# Model Setup #################
# Population file to load - Generate with 'make_ny_pop.py'
popfile = 'nyppl.pop'

# Layer specification file used in popgeneration
layers = pd.read_csv('layers.csv', index_col='layer')

# Data files to fit with
cumDeathsfile = 'EpiData/deathsNY20200106.csv' #Deaths 2021-01-06

# Model parameters
pars = sc.objdict(
        pop_size     = 200e3,
        pop_scale    = 100,
        rescale      = True,
        
        pop_infected = 10000,   # 0.05% of population infected at start of simulation - Should have a reference, but is stated in a NYT article somewhere.
    
        contacts     = layers['contacts'].to_dict(),
     
        beta         = 0.07576320418933516,
        beta_layer   = layers['beta_layer'].to_dict(),
    
        start_day    = start_day,
        end_day      = end_day,
    
        rand_seed    = 271220,
    
        verbose      = .1,
        )

# Intervention level fitted to first wave 
intv = {'H': 1.2765967578928226, 
        'W': 0.07393991037226055,
        'C': 0.07393991037226055}

############ Interventions ###############
''' Make interventions, as scaling of beta.
-- Level specific intervention effects
-- i.e. Households see increase in transmission with school/work closures

** intv = 0 - No transmission
** intv = 1 - Regular transmission (no intervention)
** intv > 1 - increase in transmission

As of now keep schools closed, maybe open them in fall, and close again at thanksgiving/december??
'''
    
def make_ints(lintv, intv=intv):
    
    interventions = [
        # School layer
        cv.change_beta(days    = [schoolsClosure, lifting, liftingSW],
                       changes = [0, lintv['S'], 1],
                       layers  = ['S'],
                       do_plot = True,
                      ),
        
        # Workplace layer
        cv.change_beta(days    = [NYSonPause, lifting, liftingSW],
                       changes = [intv['W'], lintv['W'], 1],
                       layers  = ['W'],
                       do_plot = False,
                      ),
        
        # Householsd layer
        cv.change_beta(days    = [NYSonPause, lifting, liftingSW],
                       changes = [intv['H'], lintv['H'], 1],
                       layers  = ['H'],
                       do_plot = True,
                      ),
        
        # Community layer
        cv.change_beta(days    = [NYSonPause, lifting, liftingSW],
                       changes = [intv['C'], lintv['C'], 1],
                       layers  = ['C1'],
                       do_plot = False,
                      ),
        cv.dynamic_pars(n_imports=dict(days=[0, 141, 142], vals=[0, 10, 0])),
    ]

    
    # Regenerate dynamic layers
    interventions.insert(0, population.UpdateNetworks())
    
    return interventions

############## Simulation/calibration setup ############
## Initialize simulation with intervention
def make_sim(pars, lintv={'S':1,'W':1,'H':1,'C':1}, load_pop=True, popfile=popfile, datafile=cumDeathsfile):
    sim = cv.Sim(pars=pars,
                 popfile=popfile,
                 load_pop=load_pop,
                 datafile=datafile)
    
    sim.pars['interventions'] = make_ints(lintv=lintv)
    
    sim.initialize()
    
    return sim

## Running simulation
def run_sim(pars, lintv={'S':1,'W':1,'H':1,'C':1}, popfile=popfile, return_stat=False, verbose=0.1):
    sim = make_sim(pars=pars, lintv=lintv, popfile=popfile)
    sim.run(verbose=verbose)
    
    if return_stat:
        stat = sim.results['cum_infections'][-1]
        return stat
    else:
        return sim


    

############## Calibration settings ###############
name      = 'lintv-SW-herd'

W_low     = 0.07 #0
W_high    = 1 #1

n_workers = 2 # Define how many workers to run in parallel
n_trials  = 50 # Define the number of trials, i.e. sim runs, per worker

db_name   = f'{name}.db'
storage   = f'sqlite:///{db_name}'



############### Calibration workings ##############
def run_trial(trial):
    ''' Define the objective for Optuna '''    
    lintv_W         = trial.suggest_uniform('lintv_W', W_low, W_high)
    lintv_H         = -0.3*lintv_W+1.3
    lintv           = {'S':lintv_W, 'W':lintv_W, 'H':lintv_H, 'C':lintv_W}
    
    cum_d = run_sim(pars, lintv=lintv, return_stat=True, verbose=0)
    return cum_d

def worker():
    ''' Run a single worker '''
    study = op.load_study(storage=storage, study_name=name)
    output = study.optimize(run_trial, n_trials=n_trials)
    return output

def run_workers():
    ''' Run multiple workers in parallel '''
    output = sc.parallelize(worker, n_workers)
    return output


def make_study():
    ''' Make a study, deleting one if it already exists '''
    if os.path.exists(db_name):
        os.remove(db_name)
        print(f'Removed existing calibration {db_name}')
    output = op.create_study(storage=storage, study_name=name)
    return output




########### Run the optimization ############
t0 = sc.tic()
make_study()
run_workers()
study = op.load_study(storage=storage, study_name=name)
best_pars = study.best_params
T = sc.toc(t0, output=True)
print(f'\n\nOutput: {best_pars}, time: {T:0.1f} s')

'''
Optimal intervention level estimate: lintv_W=0.355
!! lintv_H=-0.3*lintv_W+1.3 = 1.195
'''



########### Scenarios #############
## Code which is commented out (single #) are used to run the simulation which is loaded underneath

#basesim = make_sim(pars=pars, lintv={'W':0.355, 'C':0.355, 'S':0.355, 'H':-0.3*0.35+1.3})
#msim = cv.MultiSim(basesim)
#msim.run(n_runs=50, n_cpus=10)
#msim.median(quantiles=[0.025, 0.975])
#msim.plot()
#msim.save("second_wave_hd50.msim")

msim = cv.load("alreadyRun/second_wave_hd50.msim")

## Check that there's still infectious individuals left
for sim in msim.sims:
    print(sim.label)
    print(sim.results['new_infectious'][409])
    
    
## Final size --> Herd immunity threshold over different seeds - Quantiles
fin_size = [0]*50
ind = 0
for sim in msim.sims:
    fin_size[ind] = sim.results['cum_deaths'][-1]
    ind += 1

np.quantile(fin_size, [0.025, 0.5, 0.975])
np.quantile(fin_size, [0.025, 0.5, 0.975])/200e3
## [74.52449518, 75.2632428 , 75.86432382]

    
    
###### Running without interventions 
#basesimf = make_sim(pars=pars, lintv={'W':1, 'C':1, 'S':1, 'H':1})
#msimf = cv.MultiSim(basesimf)
#msimf.run(n_runs=50, n_cpus=10)
#msimf.median(quantiles=[0.025, 0.975])
#msimf.save("second_wave_free50.msim")

mismf = cv.load("alreadyRun/second_wave_free50.msim")


cum_inf = [0]*50
ind = 0
for sim in msimf.sims:
    cum_inf[ind] = sim.results['cum_infectious'][-1]
    ind += 1

np.quantile(cum_inf, [0.025, 0.5, 0.975])
np.quantile(cum_inf, [0.025, 0.5, 0.975])/200e3


####### With current estimated interventions
#I_W = 0.23084114685289414
#basesimCI = make_sim(pars=pars, lintv={'W':I_W, 'C':I_W, 'S':I_W, 'H':-0.3*I_W+1.3})
#msimCI = cv.MultiSim(basesimCI)
#msimCI.run(n_runs=50, n_cpus=10)
#msimCI.median(quantiles=[0.025, 0.975])
#msimCI.save("second_wave_fit_reopen50.msim")

msimCI = cv.load("alreadyRun/second_wave_fit_reopen50.msim")

cum_inf = [0]*50
ind = 0
for sim in msimCI.sims:
    cum_inf[ind] = sim.results['cum_deaths'][-1]
    ind += 1

np.quantile(cum_inf, [0.025, 0.5, 0.975])
np.quantile(cum_inf, [0.025, 0.5, 0.975])/200e3










###### Running with closed schools
basesimCS = make_sim(pars=pars, lintv={'W':0.44, 'C':0.44, 'S':0, 'H':-0.3*0.44+1.3})
msimCS = cv.MultiSim(basesimCS)
msimCS.run(n_runs=25, n_cpus=10)
msimCS.median(quantiles=[0.025, 0.975])



























