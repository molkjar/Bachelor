import covasim as cv
import covasim.defaults as cvd
import covasim.utils as cvu
import numba as nb
import numpy as np
import pandas as pd

def generate_people(n_people: int, n_contacts: int, dispersion=None) -> cv.People:
    people = cv.People(pars={'pop_size': n_people})
    
    people.contacts['a'] = RandomLayer(people.indices(), n_contacts, dispersion)
    people.age = np.zeros(int(n_people))+40
    return people
    

class RandomLayer(cv.Layer):
    """
    Generate and dynamically update random layer
    """

    def __init__(self, inds, mean_contacts, dispersion=None, dynamic=False):
        super().__init__()
        self.inds = inds
        self.mean_contacts = mean_contacts
        self.dispersion = dispersion
        self.dynamic = dynamic
        self.update(force=True)

    @staticmethod
    @nb.njit
    def _get_contacts(inds, number_of_contacts):
        """
        Configuration model network generation

        Args:
            inds: person indices
            number_of_contacts: number of contacts for each ind
        """
        total_number_of_half_edges = np.sum(number_of_contacts)

        count = 0
        source = np.zeros((total_number_of_half_edges,), dtype=cvd.default_int)
        for i, person_id in enumerate(inds):
            n_contacts = number_of_contacts[i]
            source[count:count + n_contacts] = person_id
            count += n_contacts
        target = np.random.permutation(source)

        return source, target

    def update(self, force: bool = False) -> None:
        #Dynimically update network contacts

        if not self.dynamic and not force:
            return

        n_people = len(self.inds)

        # sample from pois or nb
        if pd.isna(self.dispersion):
            number_of_contacts = cvu.n_poisson(rate=self.mean_contacts, n=n_people)
        else:
            number_of_contacts = cvu.n_neg_binomial(rate=self.mean_contacts-1, dispersion=self.dispersion, n=n_people) + 1

        self['p1'], self['p2'] = self._get_contacts(self.inds, number_of_contacts)
        self['beta'] = np.ones(len(self['p1']), dtype=cvd.default_float)
        self.validate()