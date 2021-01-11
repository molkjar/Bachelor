import covasim as cv
import covasim.utils as cvu
import optuna as op
import numpy as np
import sciris as sc
import pandas as pd
import os

############# Dates #######################

start_day    = '2020-03-01'
end_day      = '2021-10-01'

restrictions = '2020-03-01'
#lifting      = '2020-09-17' #200 days
#lifting      = '2020-10-01' #7 months/214 days
lifting      = '2020-10-27' #250 days

############# Model Setup #################
# Which network to use - Generate with 'make_rand_pop.py'
popfile = 'randppl_disp.pop'

# Model parameters
pars = sc.objdict(
        pop_size     = 200e3,
        pop_infected = 100,   # 0.05% of population infected at start of simulation - Should have a reference, but is stated in a NYT article somewhere.
     
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
    
################# Calibration Parameters ####################
name      = 'int_level_calib_disp'

int_low   = 0.55 #Interval to search
int_high  = 0.65

n_workers = 2 # Define how many workers to run in parallel
n_trials  = 100 # Define the number of trials, i.e. sim runs, per worker

db_name   = f'{name}.db'
storage   = f'sqlite:///{db_name}'
    
    
################# Calibration Workings ###################### 
def run_trial(trial):
    ''' Define the objective for Optuna '''
    int_level    = trial.suggest_uniform('int_level', int_low, int_high)
    cum_inf = run_sim(pars, 
                      int_level=int_level, 
                      popfile=popfile, 
                      stat_fun=stat_cum_inf, 
                      return_stat=True)
    
    return cum_inf

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











############# Run the optimization ##############
## Run everything above before optimizing
t0 = sc.tic()
make_study()
run_workers()
study = op.load_study(storage=storage, study_name=name)
best_pars = study.best_params
T = sc.toc(t0, output=True)
print(f'\n\nOutput: {best_pars}, time: {T:0.1f} s')


'''
Output: {'int_level': ~0.63}
'''






