import covasim as cv
import pandas as pd
import sciris as sc
import numpy as np

import population_random as pr


if __name__ == '__main__':
    #Without dispersion
    cv.set_seed(1)
    people = pr.generate_people(n_people=200e3, n_contacts=20, dispersion=None)
    sc.saveobj('randppl.pop', people)

    # With dispersion
    cv.set_seed(1)
    peopleDisp = pr.generate_people(n_people=200e3, n_contacts=20, dispersion=1.5)
    sc.saveobj('randppl_disp.pop', peopleDisp)
    


