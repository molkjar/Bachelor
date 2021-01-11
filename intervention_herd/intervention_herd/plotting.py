import covasim as cv
import sciris as sc
cv.options.set(dpi=200)

# Negative bbinomial plotting
nb_hd = cv.load("msim_at_herd25.msim")
nb_f = cv.load("msim_free25.msim")
nb_sw = cv.load("msim_sec_wave25.msim")

nb_hd.median(quantiles=[0.025, 0.975])
nb_f.median(quantiles=[0.025, 0.975])
nb_sw.median(quantiles=[0.025, 0.975])

nb_hd.label = r'$\alpha=0.63$'
nb_f.label = r'$\alpha=1$'
nb_sw.label = r'$\alpha=0.55$'

nb_tg = cv.MultiSim([nb_f, nb_hd, nb_sw])

nb_tg.plot(to_plot=plots, fig_args={'figsize': (10, 5)}, do_save=True)
nb_tg.plot(to_plot=['r_eff'], do_save=True)

plots = sc.odict({
                'Total counts': [
                    'cum_infectious',
                    'n_infectious',
                ]
        })


# Erdös-Renyi
pois_hd = cv.load("msim_at_herd25_pois.msim")
pois_f = cv.load("msim_free_pois25.msim")
pois_sw = cv.load("msim_sw_pois25.msim")

pois_hd.median(quantiles=[0.025, 0.975])
pois_f.median(quantiles=[0.025, 0.975])
pois_sw.median(quantiles=[0.025, 0.975])

pois_hd.label = r'$\alpha=0.725$'
pois_f.label = r'$\alpha=1$'
pois_sw.label = r'$\alpha=0.63$'

pois_tg = cv.MultiSim([pois_f, pois_hd, pois_sw])

pois_tg.plot(to_plot=plots, fig_args={'figsize': (10, 5)}, do_save=True)
pois_tg.plot(to_plot=['r_eff'], do_save=True)


