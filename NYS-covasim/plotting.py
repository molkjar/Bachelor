import covasim as cv
import sciris as sc
cv.options.set(dpi=200)
'''Plotting report figures from msim objects'''

## First wave fitting
fw = cv.load('alreadyRun/FinalFW50.msim')

fw.plot(to_plot=['new_deaths'], colors = ['steelblue'], fig_args={'figsize':(10,4)}, do_save=True)
fw.plot(to_plot=['cum_deaths'], colors = ['steelblue'], fig_args={'figsize':(10,4)}, do_save=True)

plots = sc.odict({
                '': [
                    'n_infectious',
                    'cum_infectious'
                ]
        })
fw.plot(to_plot=plots, colors=['steelblue', '#d7b194'], fig_args={'figsize':(10,4)})



## Second wave - Optimized interventions
sw_opt = cv.load('second_wave_hd50.msim')

sw_opt.plot(to_plot=plots, colors=['steelblue', '#d7b194'], fig_args={'figsize':(10,4)}, do_save=True)

sw_opt.plot(to_plot=['cum_deaths'], colors=['#d7b194'], fig_args={'figsize':(10,4)}, do_save=True)




## Second wave - Current track projected deaths vs observed
sw_CI = cv.load('alreadyRun/second_wave_fit50.msim')

sw_CI.plot(to_plot=['new_deaths'], colors=['steelblue'], fig_args={'figsize':(10,4)}, do_save=True)





## Second Wave current track concluded - Reopen
sw_CI_reopen = cv.load('alreadyRun/second_wave_fit_reopen50.msim')

sw_CI_reopen.plot(to_plot=plots,  colors=['steelblue', '#d7b194'], fig_args={'figsize':(10,4)}, do_save=True)





###### For supplementary
## Different networks R_t
net_beta = cv.load('sens_beta_net10.msim')
net_beta.median(quantiles=[0.025, 0.975])

net_beta.plot(to_plot=['r_eff'], colors=['seagreen'], fig_args={'figsize':(10,4)}, do_save=True)

## Different networks Hd?
net_hd = cv.load('sens_epiopt_tonet10.msim')

net_hd.plot(to_plot=plots, colors=['steelblue', '#d7b194'], do_save=True)


## Different networks Current ints?
net_CI = cv.load('sens_epiCI_tonet10.msim')

net_CI.plot(to_plot=['new_deaths'], colors=['steelblue'], do_save=True)






## R_eff from given beta
beta_r = cv.load('beta_r_eff.msim')
beta_r.plot(to_plot=['r_eff'], colors=['seagreen'], fig_args={'figsize':(10,4)}, do_save=True)


## FW Sens on alpha_C
msimalphap = cv.load('alreadyRun/sens_alpha_c.msim')
msimalphap.plot(to_plot=['cum_deaths'], do_save=True)
