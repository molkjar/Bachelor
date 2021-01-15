import covasim as cv
import covasim.utils as cvu
import optuna as op
import sciris as sc
import pandas as pd
import numpy as np
import os
from collections import defaultdict
import population

'''
Given the transmission rate from first wave fitting --> Fit to partially observed second wave
'''


############# Dates #######################

start_day    = '2020-03-01'
end_day      = '2020-12-31'

''' NYSonPause:
    General closure, incorporate school closures (happened few days before)
    Shelter-in-place order etc.
    Everything non-essential closed
    Lifting: Chosen quite arbitrary although somewhat at the right time'''

NYSonPause     = '2020-03-22'
schoolsClosure = '2020-03-16'

lifting        = '2020-07-20'

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
        
        pop_infected = 10000,   # 0.05% of population infected at start of simulation
    
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

'''
    
def make_ints(lintv, intv=intv):
    
    interventions = [
        # School layer
        cv.change_beta(days    = [schoolsClosure, lifting],
                       changes = [0, lintv['S']],
                       layers  = ['S'],
                       do_plot = True,
                      ),
        
        # Workplace layer
        cv.change_beta(days    = [NYSonPause, lifting],
                       changes = [intv['W'], lintv['W']],
                       layers  = ['W'],
                       do_plot = False,
                      ),
        
        # Householsd layer
        cv.change_beta(days    = [NYSonPause, lifting],
                       changes = [intv['H'], lintv['H']],
                       layers  = ['H'],
                       do_plot = True,
                      ),
        
        # Community layer
        cv.change_beta(days    = [NYSonPause, lifting],
                       changes = [intv['C'], lintv['C']],
                       layers  = ['C1'],
                       do_plot = False,
                      ),
        cv.dynamic_pars(n_imports=dict(days=[0, 141, 142], vals=[0, 10, 0]), do_plot=False), # a small import to ensure the disease are present
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
def run_sim(pars, lintv={'S':1,'W':1,'H':1,'C':1}, popfile=popfile, return_mse=False, verbose=0.1):
    sim = make_sim(pars=pars, lintv=lintv, popfile=popfile)
    sim.run(verbose=verbose)
    
    if return_mse:
        fit = sim.compute_fit(skestimator='mean_squared_error') #MSE
        return fit.mismatch
    else:
        return sim


    

############## Calibration settings ###############
name      = 'SW_fit'

W_low     = 0 #0
W_high    = 0.3 #1

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
    
    cum_d = run_sim(pars, lintv=lintv, return_mse=True, verbose=0)
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



'''Best fit:
0.23084114685289414
'''
## Rerunning
I_W = 0.23084114685289414
basesim = make_sim(pars=pars, lintv={'W':I_W, 'C':I_W, 'S':I_W, 'H':-0.3*I_W+1.3})

msim = cv.MultiSim(basesim)
msim.run(n_runs=50, n_cpus=10)
msim.median(quantiles=[0.025, 0.975])
msim.save("second_wave_fit50.msim")
