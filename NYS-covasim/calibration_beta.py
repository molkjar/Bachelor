import covasim as cv
import covasim.utils as cvu
import optuna as op
import sciris as sc
import pandas as pd
import numpy as np
import make_ny_pop
import population
import os
from collections import defaultdict

'''
Script for calibrating beta given contact network to R0 estimate from (NY R0= 6.4 (4.3-9.0) 95% confidence interval)
https://doi.org/10.1101/2020.05.17.20104653; Ives et al
'''

############# Dates #######################
start_day    = '2020-03-01'
end_day      = '2020-03-31'


############ Model setup ##################
# Population file to load - Generate with 'make_ny_pop.py'
popfile = 'nyppl.pop'

# Layer specification file used in popgeneration
layers = pd.read_csv('layers.csv', index_col='layer')

# Model parameters
pars = sc.objdict(
        pop_size     = 200e3,
        pop_scale    = 100,
        rescale      = True,
        
        pop_infected = 10000,   # 0.05% of population infected at start of simulation
    
        contacts     = layers['contacts'].to_dict(),
     
        beta         = 0.0758, # Parameter to be calibrated
        beta_layer   = layers['beta_layer'].to_dict(),
    
        start_day    = start_day,
        end_day      = end_day,
    
        rand_seed    = 271220,
    
        verbose      = .1,
        )

############# Interventions #################
# Intervention function only used to re-sample community contacts
def make_ints():
    interventions = [population.UpdateNetworks()]
    
    return interventions

############# Simulation setup ##############
## Inititalize sim
def make_sim(pars, load_pop=True, popfile=popfile):
    sim = cv.Sim(pars=pars,
                 popfile=popfile,
                 load_pop=load_pop)
    
    sim.pars['interventions'] = make_ints() #regeneate community network
    
    sim.initialize()
    
    return sim

## Running simulation
def run_sim(pars, R0=4.3, label=None, return_mse=False):
    sim = make_sim(pars=pars)
    sim.run()
    
    if return_mse:
        mse = np.abs(sim.compute_r_eff()-R0)
        return mse
    else:
        return sim
    
############## Calibration settings ###############
name      = 'beta_low-calibration'

beta_low  = 0
beta_high = 0.2

n_workers = 2 # Define how many workers to run in parallel
n_trials  = 50 # Define the number of trials, i.e. sim runs, per worker

db_name   = f'{name}.db'
storage   = f'sqlite:///{db_name}'



############### Calibration workings ##############
def run_trial(trial):
    ''' Define the objective for Optuna '''
    pars["beta"] = trial.suggest_uniform('beta', beta_low, beta_high)
    mse = run_sim(pars, R0=9.0, return_mse=True)
    return mse


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
With Ives, Bozzuto:
beta_low  = 0.044788361258410944 (R0=4.3, lower bound in 95%CI)
beta_high = 0.0986475314806892   (R0=9.0, upper bound in 95%CI)
'''



########## Sensititvity to seed ############
pars['beta'] = 0.044788361258410944
msim = cv.MultiSim(make_sim(pars=pars))
msim.run(n_runs=1, keep_people=True)
msim.median()
msim.plot(to_plot=['r_eff'])


