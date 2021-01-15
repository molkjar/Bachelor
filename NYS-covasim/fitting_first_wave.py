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
Script for fitting simulation model to first wave epidmeic data. Parameter space are restricted by R0 and the feasible intervention effect intervals.
Analyses start at line ~200
'''

############# Dates #######################

start_day      = '2020-03-01'
end_day        = '2020-07-20'

''' NYSonPause:
    General closure, incorporate school closures (happened few days before)
    Shelter-in-place order etc.
    Everything non-essential closed
    Lifting: Chosen quite arbitrary although somewhat at the right time'''

NYSonPause     = '2020-03-22'
schoolsClosure = '2020-03-16'

lifting        = '2020-07-20' #141 days

############# Model Setup #################
# Population file to load - Generate with 'make_ny_pop.py'
popfile = 'nyppl.pop'

# Layer specification file used in popgeneration
layers = pd.read_csv('layers.csv', index_col='layer')

# Data files to fit with
cumDeathsfile = 'EpiData/deathsNYFW20200106.csv' #Deaths 2021-01-06

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


############ Interventions ###############
''' Make interventions, as scaling of beta.
-- Level specific intervention effects
-- i.e. Households see increase in transmission with school/work closures

** intervention level - For fitting
** intv = 0 - No transmission
** intv = 1 - Regular transmission (no intervention)
** intv > 1 - increase in transmission

'''
    
def make_ints(intv):

    
    interventions = [
        # School layer
        cv.change_beta(days    = [schoolsClosure],
                       changes = [0],
                       layers  = ['S'],
                       do_plot = False,
                      ),
        
        # Workplace layer
        cv.change_beta(days    = [NYSonPause, lifting],
                       changes = [intv['W'], 1],
                       layers  = ['W'],
                       do_plot = False,
                      ),
        
        # Householsd layer
        cv.change_beta(days    = [NYSonPause, lifting],
                       changes = [intv['H'], 1],
                       layers  = ['H'],
                       do_plot = False,
                      ),
        
        # Community layer
        cv.change_beta(days    = [NYSonPause, lifting],
                       changes = [intv['C'], 1],
                       layers  = ['C1'],
                       do_plot = True,
                      ),
    ]
    
    # Regenerate dynamic layers
    interventions.insert(0, population.UpdateNetworks())
    
    return interventions

############## Simulation/calibration setup ############
## Initialize simulation with intervention
def make_sim(pars, intv={'W':1,'H':1,'C':1}, load_pop=True, popfile=popfile, datafile=cumDeathsfile, beta=pars['beta']):
    pars['beta']=beta
    sim = cv.Sim(pars=pars,
                 popfile=popfile,
                 load_pop=load_pop,
                 datafile=datafile)
    
    sim.pars['interventions'] = make_ints(intv = intv)
    
    sim.initialize()
    
    return sim

## Running simulation
def run_sim(pars, intv={'W':1,'H':1,'C':1}, popfile=popfile, return_mse=False, verbose=0.1):
    sim = make_sim(pars=pars, intv=intv, popfile=popfile)
    sim.run(verbose=verbose)
    
    if return_mse:
        fit = sim.compute_fit(skestimator='mean_squared_error') #MSE
        return fit.mismatch
    else:
        return sim

    

############## Calibration settings ###############
name      = 'intv-fitting'

beta_low  = 0.0447
beta_high = 0.0986

W_low     = 0
W_high    = 1

H_low     = 0.8
H_high    = 2

#C_low     = 0
#C_high    = 0.5

n_workers = 2 # Define how many workers to run in parallel
n_trials  = 100 # Define the number of trials, i.e. sim runs, per worker

db_name   = f'{name}.db'
storage   = f'sqlite:///{db_name}'



############### Calibration workings ##############
def run_trial(trial):
    ''' Define the objective for Optuna '''
    pars['beta']   = trial.suggest_uniform('beta', beta_low, beta_high)
    
    intv_W         = trial.suggest_uniform('intv_W', W_low, W_high)
    intv_H         = trial.suggest_uniform('intv_H', H_low, H_high)
    #intv_C         = trial.suggest_uniform('intv_C', C_low, C_high)
    intv           = {'W':intv_W, 'H':intv_H, 'C':intv_W}
    
    mse = run_sim(pars, intv=intv, return_mse=True, verbose=0)
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




############## Analysing fitted model ##################
pars['beta'] = 0.07576320418933516
basesim = make_sim(pars=pars, intv={'W': 0.07393991037226055, 
                                    'H': 1.2765967578928226, 
                                    'C': 0.07393991037226055})
# Rerunning
msim = cv.MultiSim(basesim)
msim.run(n_runs=50, n_cpus=10)





#### Differece in beta, same alpha
beta = [0.035, 0.055, 0.07576320418933516, 0.095, 0.115]
simsbeta = [make_sim(pars=pars, beta=i, intv={'W': 0.07393991037226055, 
                                              'H': 1.2765967578928226, 
                                              'C': 0.07393991037226055})
           for i in beta]


msimbeta = cv.MultiSim(simsbeta)
msimbeta.run(reseed=False)

for i in range(5):
    msimbeta.sims[i].label = r'$\beta=$'+str(round(beta[i],2))
msimbeta.save('sens_beta_fw.msim')






#### Different alpha_CW, same beta
alpha_W = [0.03,0.05,0.07393991037226055,0.09,0.11]
simsalphap = [make_sim(pars=pars, intv={'W': a, 'H': 1.2765967578928226, 'C': a}) for a in alpha_W]

msimalphap = cv.MultiSim(simsalphap)
msimalphap.run(reseed=False)

for i in range(5):
    msimalphap.sims[i].label = r'$\alpha_C=$'+str(round(alpha_W[i],2))
msimalphap.save("sens_alpha_c.msim")


