######### Find appropriate beta for given network such that R_0=2.5 #################
import covasim as cv
import covasim.utils as cvu
import optuna as op
import numpy as np
import sciris as sc
import pandas as pd
import os

pars = sc.objdict(
        pop_size     = 200e3,
        pop_infected = 100,
    
        beta         = 0.055,
    
        start_day    = '2020-03-01',
        end_day      = '2021-03-01',
    
        verbose      = 0,
        )


def make_sim(pars, beta=0.055, load_pop=True, popfile='randppl.pop'):
    sim = cv.Sim(pars=pars,
                 popfile=popfile,
                 load_pop=load_pop)
    
    sim.pars['beta'] = beta
    
    sim.initialize()
    
    return sim


def run_sim(pars, beta, R_target=2.5, label=None, return_mse=False):
    sim = make_sim(pars=pars, beta=beta)
    sim.run()
    R0 = sim.compute_r_eff()[1]
    
    if return_sim:
        return sim
    else:
        return np.abs(R0-R_target)
    
    
def run_trial(trial):
    ''' Define the objective for Optuna '''
    beta = trial.suggest_uniform('', 0, 0.05)
    R0 = run_sim(pars, beta)
    return R0

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


if __name__ == '__main__':

    # Settings
    n_workers = 2 # Define how many workers to run in parallel
    n_trials  = 10 # Define the number of trials, i.e. sim runs, per worker
    name      = 'beta-calibration'
    db_name   = f'{name}.db'
    storage   = f'sqlite:///{db_name}'

    # Run the optimization
    t0 = sc.tic()
    make_study()
    run_workers()
    study = op.load_study(storage=storage, study_name=name)
    best_pars = study.best_params
    T = sc.toc(t0, output=True)
    print(f'\n\nOutput: {best_pars}, time: {T:0.1f} s')
