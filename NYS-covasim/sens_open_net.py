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
end_day      = '2021-10-01'

''' NYSonPause:
    General closure, incorporate school closures (happened few days before)
    Shelter-in-place order etc.
    Everything non-essential closed
    Lifting: Chosen quite arbitrary although somewhat at the right time'''

NYSonPause     = '2020-03-22'
schoolsClosure = '2020-03-16'

lifting        = '2020-07-20'
liftingSW      = '2021-04-01'

############# Model Setup #################
# Population file to load - Generate with 'make_ny_pop.py'
popfiles = ['nyppl.pop', 'ppl/nyppl2867.pop', 'ppl/nyppl327849.pop', 'ppl/nyppl34.pop', 'ppl/nyppl46.pop', 'ppl/nyppl6234.pop', 'ppl/nyppl7.pop', 'ppl/nyppl87.pop', 'ppl/nyppl8765.pop', 'ppl/nyppl964.pop']

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
        cv.change_beta(days    = [schoolsClosure, lifting,liftingSW],
                       changes = [0, lintv['S'],1],
                       layers  = ['S'],
                       do_plot = True,
                      ),
        
        # Workplace layer
        cv.change_beta(days    = [NYSonPause, lifting,liftingSW],
                       changes = [intv['W'], lintv['W'],1],
                       layers  = ['W'],
                       do_plot = False,
                      ),
        
        # Householsd layer
        cv.change_beta(days    = [NYSonPause, lifting,liftingSW],
                       changes = [intv['H'], lintv['H'],1],
                       layers  = ['H'],
                       do_plot = True,
                      ),
        
        # Community layer
        cv.change_beta(days    = [NYSonPause, lifting,liftingSW],
                       changes = [intv['C'], lintv['C'],1],
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
def make_sim(pars, popfileind, lintv={'S':1,'W':1,'H':1,'C':1}, load_pop=True, datafile=cumDeathsfile):
    sim = cv.Sim(pars=pars,
                 popfile=popfiles[popfileind],
                 load_pop=load_pop,
                 datafile=datafile)
    
    sim.pars['interventions'] = make_ints(lintv=lintv)
    
    sim.initialize()
    
    return sim

###### Run sim for several generated networks
## No interventions
simsf = [make_sim(pars=pars, popfileind=i) for i in range(10)]
msimf = cv.MultiSim(simsf)
msimf.run(n_cpus=5)
msimf.median(quantiles=[0.025, 0.975])
msimf.save("sens_epiopen_tonet10.msim")


## to optimal interventions
## No interventions
sims = [make_sim(pars=pars, popfileind=i, lintv={'W':0.355, 'C':0.355, 'S':0.355, 'H':-0.3*0.355+1.3}) for i in range(10)]
msim = cv.MultiSim(sims)
msim.run(n_cpus=5)
msim.median(quantiles=[0.025, 0.975])
msim.save("sens_epiopt_tonet10.msim")


## With current intervention
I_W = 0.23084114685289414
simsCI = [make_sim(pars=pars, popfileind=i, lintv={'W':I_W, 'C':I_W, 'S':I_W, 'H':-0.3*I_W+1.3}) for i in range(10)]
msimCI = cv.MultiSim(simsCI)
msimCI.run(n_cpus=5)
msimCI.median(quantiles=[0.025, 0.975])
msimCI.save("sens_epiCI_tonet10.msim")
