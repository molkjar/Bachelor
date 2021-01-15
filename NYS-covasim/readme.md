# Running epidemics, calibrating and fitting to New York State data

### Misc Files and folders
* alreadyRun: cv.msim files with concluded simulations
* data: Census/Mistry data for network generation
* EpiData: Timeseries data for NYS covid deaths
* ppl: Several regenerated contact network, cv.People object

* nc.txt: a sample from the degree distribution
* plotting.py: Plotting the report figures
* second_wave_scenario.py: Simulations for the different mitigation strategies
* sens_beta_net.py: sensitivity of beta on regenerating network
* sens_open_net.py: sensitivity of sim outcome on regenerating network

#### Calibration and fitting
* calibration_beta.py: Calibrate transmission rate to NYS R0 est. from Ives et al
* fitting_first_wave.py: Fitting beta (in given CI interval) and intervention levels to first wave death data
* second_wave_fit.py: Fitting, given first wave fit, the second wave intervention level

#### Network
* layers.csv: Network layer parameters, used in several scripts, thus saves in a seperate file
* population.py: The inner workings of generating networks
* make_ny_pop.py: Easy way to call from population.py
* nyppl.pop: A generated network



