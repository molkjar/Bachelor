import covasim as cv
import covasim.defaults as cvd
import covasim.utils as cvu
import numba as nb
import numpy as np
import pandas as pd
from collections import defaultdict

'''
Methods for generating population network.

Most important methods:
    generate_people --> make_households
    add_work/school_contacts --> create_clusters
    add_other_contacts --> RandomLayer
    

Many thanks to Romesh Abeysuriya for guidance.
'''


def generate_people(n_people, mixing, reference_ages, households) -> cv.People:
    
    '''
    From demographic data (cencus) households are generated, in this way we generate people and assign 
    them to a household in the same action. Base for generating the multi-layered network - NOT for the 
    simple random network.
    
    Requires:   Household mixing matrix (See https://github.com/mobs-lab/mixing-patterns)
                Householder age distribution (Cencus data)
                Household size distribution (Cencus data)
                Number of individuals to generate.
    
    Creates a cv.People object.
    '''

    # Number of households to generate
    total_people      = sum(households.index * households.values)            #Total number of people found from household size dist
    household_percent = households / total_people                            #Normalizing distribution
    n_households      = (n_people * household_percent).round().astype(int)   #Actual number of households of different sizes with our pop size
    
    # The n_households are rounding, thus a bit unprecise, we adjust such that remaining individuals are in 1-person households
    n_households[1]   += n_people - sum(n_households * n_households.index) 

    # Draw householder from householder age distribution
    household_heads = np.random.choice(reference_ages.index, size=sum(n_households), p=reference_ages.values / sum(reference_ages))

    # Create individuals with age assigned to a household, based on the formerly created householders and household mixing matrices
    h_clusters, ages = make_households(n_households, n_people, household_heads, mixing)
    
    # Parse into a cv.People object
    contacts        = cv.Contacts()
    contacts['H']   = clusters_to_layer(h_clusters)
    people          = cv.People(pars={'pop_size': n_people}, age=ages)
    people.contacts = contacts

    return people




def add_school_contacts(people: cv.People, mean_contacts):
    '''
    Add school contact layer, from mean classroom size to a cv.People instance.
    Actual classroom size is drawn from poisson distribution.
    Everyone under 18 is assigned to a classroom cluster.
    '''

    classrooms = []

    # Create classrooms of children of same age, assign a teacher from the adult (>21) population
    for age in range(0, 18):
        children_thisage = cvu.true(people.age == age)
        classrooms.extend(create_clusters(children_thisage, mean_contacts)) #draw the clusters

        teachers = np.random.choice(cvu.true(people.age > 21), len(classrooms), replace=False) # add teachers
        for i in range(len(classrooms)):
            classrooms[i].append(teachers[i])

    # Add to cv.People 
    people.contacts['S'] = clusters_to_layer(classrooms)


    

def add_work_contacts(people: cv.People, mean_contacts):
    '''
    Add work contact layer, from mean number of coworkers and already generated people, to a cv.People instance.
    Actual size of workplace cluster drawn from poisson distribution.
    Everyone in the age interval [18, 65] are assigned to a workplace cluster.
    '''
    
    work_inds = cvu.true((people.age > 18) & (people.age <= 65))
    work_cl = create_clusters(work_inds, mean_contacts) #Draw clusters
    
    # Add to cv.People instance
    people.contacts['W'] = clusters_to_layer(work_cl)


    
    
def add_other_contacts(people: cv.People, layers):
    """
    Add random layers according to the csv layer file - For our cause only a single community layer
    
    """

    for layer_name, layer in layers.iterrows():

        if layer['cluster_type'] in {'home', 'school', 'work'}:
            # ignore non random layers
            continue
        
        people.contacts[layer_name] = RandomLayer(people.indices(), layer['contacts'], layer['dispersion'], dynamic=(not pd.isna(layer['dynamic']))) #if dynamic, resample layer every day


## HELPERS
## Households
def make_households(n_households, pop_size, household_heads, mixing_matrix):
    """
    Requires:   Household mixing matrix (See https://github.com/mobs-lab/mixing-patterns)
                Number of household of different sizes to generate
                Number of individuals
                List of household heads

    Creates
        h_clusters: household id lists
        ages: ages for all individuals
    """
    
    mixing_matrix = mixing_matrix.div(mixing_matrix.sum(axis=1), axis=0)
    samplers = [AliasSample(mixing_matrix.iloc[i, :].values) for i in range(mixing_matrix.shape[0])]  # Precompute samplers for each age

    age_lb = [int(x.split(' to ')[0]) for x in mixing_matrix.index]
    age_ub = [int(x.split(' to ')[1]) for x in mixing_matrix.index]

    h_clusters = []
    uids = np.arange(0, pop_size)
    ages = np.zeros(pop_size, dtype=int)
    h_added = 0
    p_added = 0

    for h_size, h_num in n_households.iteritems():
        for household in range(h_num):
            head = household_heads[h_added]
            # sample households
            household_ages = sample_household_cluster(samplers,
                                                       age_lb,
                                                       age_ub,
                                                       head,
                                                       h_size)
            # add ages to age list
            ub = p_added + h_size
            ages[p_added:ub] = household_ages
            
            # get the UID from each household cluster
            h_ids = uids[p_added:ub]
            h_clusters.append(h_ids)
            
            h_added += 1
            p_added += h_size
    return h_clusters, ages


def sample_household_cluster(sampler, bin_lower, bin_upper, reference_age, n):
    """
    gives list of ages in a household based on mixing matrix and householder
    """

    ages = [reference_age]  # The household head

    if n > 1:
        idx = np.digitize(reference_age, bin_lower) - 1  # First, find the index of the age of the head of household
        sampled_bins = sampler[idx].draw_n(n - 1)

        for bin in sampled_bins:
            ages.append(int(round(np.random.uniform(bin_lower[bin] - 0.5, bin_upper[bin] + 0.5))))

    return np.array(ages)









## Community
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

    @staticmethod #some stuff for running more efficiently
    @nb.njit
    def _get_contacts(inds, number_of_contacts): 
        """
        Configuration model network generation - Actually create half edges and "connecting them"

        Args:
            inds: person indices
            number_of_contacts: number of contacts for each ind drawn in update()
        """
        total_number_of_half_edges = np.sum(number_of_contacts) #*2/2

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

        # sample the number of edges from a given distribution
        if pd.isna(self.dispersion):
            number_of_contacts = cvu.n_poisson(rate=self.mean_contacts, n=n_people)
        else:
            number_of_contacts = cvu.n_neg_binomial(rate=self.mean_contacts, dispersion=self.dispersion, n=n_people) + 1

        self['p1'], self['p2'] = self._get_contacts(self.inds, number_of_contacts)
        self['beta'] = np.ones(len(self['p1']), dtype=cvd.default_float)
        self.validate()


class UpdateNetworks(cv.Intervention):
    """
    For resampling the network as an intevention
    """

    def apply(self, sim):
        for layer in sim.people.contacts.values():
            if isinstance(layer, RandomLayer):
                layer.update()

                
                
                
                
                
                
                
                
#For school and workplace
def create_clusters(people_to_cluster, mean_cluster_size):
    """
    Assign people to clusters from list of inds and a mean cluster size
    """
    clusters = []
    n_people = len(people_to_cluster)
    n_remaining = n_people

    while n_remaining > 0:
        this_cluster = cvu.poisson(mean_cluster_size)  # draw the cluster size
        if this_cluster > n_remaining:
            this_cluster = n_remaining
        clusters.append(people_to_cluster[(n_people - n_remaining) + np.arange(this_cluster)].tolist())
        n_remaining -= this_cluster

    return clusters






# General helpers
def clusters_to_layer(clusters: list):
    """
    Convert a list of fully connected bidirectional clusters to a Covasim layer
    """

    p1 = []
    p2 = []
    layer_members = set()

    for cluster in clusters:
        for i in cluster:
            layer_members.add(i)
            for j in cluster:
                if j != i:
                    p1.append(i)
                    p2.append(j)

    layer = cv.Layer()
    layer['p1'] = np.array(p1, dtype=cvd.default_int)
    layer['p2'] = np.array(p2, dtype=cvd.default_int)
    layer['beta'] = np.ones(len(p1), dtype=cvd.default_float)
    layer.validate()

    return layer






## INTERNAL HELPERS FOR HOUSEHOLD GENERATION

## Fast choice implementation
# From https://gist.github.com/jph00/30cfed589a8008325eae8f36e2c5b087
# by Jeremy Howard https://twitter.com/jeremyphoward/status/955136770806444032
@nb.njit
def sample(n, q, J, r1, r2):
    res = np.zeros(n, dtype=np.int32)
    lj = len(J)
    for i in range(n):
        kk = int(np.floor(r1[i] * lj))
        if r2[i] < q[kk]:
            res[i] = kk
        else:
            res[i] = J[kk]
    return res


class AliasSample():
    def __init__(self, probs):
        self.K = K = len(probs)
        self.q = q = np.zeros(K)
        self.J = J = np.zeros(K, dtype=np.int)

        smaller, larger = [], []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small, large = smaller.pop(), larger.pop()
            J[small] = large
            q[large] = q[large] - (1.0 - q[small])
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

    def draw_one(self):
        K, q, J = self.K, self.q, self.J
        kk = int(np.floor(np.random.rand() * len(J)))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]

    def draw_n(self, n):
        r1, r2 = np.random.rand(n), np.random.rand(n)
        return sample(n, self.q, self.J, r1, r2)






