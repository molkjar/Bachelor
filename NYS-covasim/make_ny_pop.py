import covasim as cv
import pandas as pd
import sciris as sc
import population

'''
Actually creating a population network with the methods from population.py.
Requires household mixing matrix, reference age distribution (householder), household size distribution and further layer parameters
'''

def make_people(seed, pop_size):
    mixing_H       = pd.read_csv('data/contMat.csv', index_col='Age group')
    reference_ages = pd.read_csv('data/ageref.csv', index_col='age', squeeze=True)
    households     = pd.read_csv('data/households.csv', index_col='size', squeeze=True)
    layers         = pd.read_csv('layers.csv', index_col='layer')

    cv.set_seed(seed)

    # Create people and household layer
    people = population.generate_people(int(pop_size), mixing_H, reference_ages, households)

    # Other layers
    population.add_school_contacts(people, mean_contacts=layers.loc['S', 'contacts'])
    population.add_work_contacts(people, mean_contacts=layers.loc['W', 'contacts'])
    population.add_other_contacts(people, layers)

    return people



t0 = sc.tic()
people = make_people(seed=1, pop_size=200e3)
sc.saveobj('nyppl.pop', people)
T = sc.toc(t0, output=False)
