import covasim as cv
import covasim.utils as cvu
import sciris as sc
import pandas as pd
import numpy as np
import make_ny_pop
import population
import os
from collections import defaultdict

############# Dates #######################
start_day    = '2020-03-01'
end_day      = '2020-03-31'


############ Model setup ##################
# Population file to load - Generate with 'make_ny_pop.py'
popfiles = ['nyppl.pop', 'ppl/nyppl2867.pop', 'ppl/nyppl327849.pop', 'ppl/nyppl34.pop', 'ppl/nyppl46.pop', 'ppl/nyppl6234.pop', 'ppl/nyppl7.pop', 'ppl/nyppl87.pop', 'ppl/nyppl8765.pop', 'ppl/nyppl964.pop']

# Layer specification file used in popgeneration
layers = pd.read_csv('layers.csv', index_col='layer')

# Model parameters
pars = sc.objdict(
        pop_size     = 200e3,
        pop_scale    = 100,
        rescale      = True,
        
        pop_infected = 10000,   # 0.05% of population infected at start of simulation - Should have a reference, but is stated in a NYT article somewhere.
    
        contacts     = layers['contacts'].to_dict(),
     
        beta         = 0.07576320418933516, # Parameter to be calibrated such that R0=4.5
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
def make_sim(pars, popfileind, load_pop=True):
    sim = cv.Sim(pars=pars,
                 popfile=popfiles[popfileind],
                 load_pop=load_pop)
    
    sim.pars['interventions'] = make_ints() #regen community network
    
    sim.initialize()
    
    return sim


#### Run sim for several networks
sims = [make_sim(pars=pars, popfileind=i) for i in range(10)]
msim = cv.MultiSim(sims)
msim.run(reseed=False)
msim.save('sens_beta_net10.msim')


####
