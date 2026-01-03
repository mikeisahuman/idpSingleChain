##  Mike Phillips, April 2025
##  Script for comparing xij and dij/Rij results
##  e.g. from calculation and simulation, or simulation under two conditions, etc.

"""
    Be sure to check all settings below!  (yes there are several)

    And set up desired directories / files further below!

    Command line arguments (optional):
      1. sequence name
      2. ij minimum separation
      3. normalize? (boolean)
"""


import xij_plotter_module as xpm
import numpy as np
#from os import path
import sys

args = sys.argv


##  GENERAL SETTINGS

seqname = 12980     # sequence name / index

#NORM = True     # normalize each set of results by their maximum xij or Rij value?
NORM = False

#rescaleRIJ = False      # rescale Rij to get Rij* ?  i.e. Rij/(homopolymer fit)
rescaleRIJ = True

#IJmin = 1       # cut out any diagonals with |I-J| < IJmin
IJmin = 10

SYM_CBAR = False    # (attempt) symmetric colorbars in ij maps?   [does not apply to Rij or diffs; only affects xij and Rij*]
#SYM_CBAR = True

#RIJrsc_cbarlim = (0.4, 1.6)     # for custom colorbar limits, just in Rij* map
RIJrsc_cbarlim = None

RND_CBAR = 50       # factor for rounding colorbar ticks (if power of 10, rounds min/max to that decimal place)
#RND_CBAR = 30

#NUM_CBAR = 5        # number of ticks above and below midpoint, on colorbars for maps
NUM_CBAR = 3

MAP_ORIGIN = 'upper'    # specify origin of maps, 'upper' or 'lower'
#MAP_ORIGIN = 'lower'

TITLE = True    # include title in plots ?  [applies to all plots]
TITLE = 'nameonly'      # use only seq. name as title?
#TITLE = False

#FSIZE = (9,7)   # figure size / shape, as (width, height)
#FSIZE = (12,9)
FSIZE = (12,11)

#SAVE = './test/plots/'  # save plots?  provide directory, or set to 'False'
#SAVE = '../out/schuler linkers maps/paper/alexa - M1 2par retick/'
#SAVE = '../out/schuler linkers maps/paper/cycf - M1 2par retick/'
#SAVE = '../out/schuler linkers maps/revised paper/alexa - aug2025 M1 2par/'
#SAVE = f'../out/schuler linkers maps/alexa/aug2025 M1/Rij_rescale_2par_imp/'
#SAVE = f'../out/schuler linkers maps/cycf/aug2025 M1/Rij_rescale_2par_imp/'
#SAVE = f'../out/schuler linkers maps/cycf/aug2025 M1/Rij_rho_imp/'
#SAVE = f'../out/schuler linkers maps/alexa/aug2025 M1/Rij_with-cycf_rescale_2par_imp/'
SAVE = f'../out/schuler linkers maps/alexa/aug2025 M1/Rij_with-cycf_rho_imp/'
#SAVE = False

#SHOW = True     # show plots?  will show when saving, too   [includes difference plots, if second set is provided]
SHOW = False

#RSEP_FIT = False    # fit R separation trend (vs |i-j|) ?  (also shows in plot)  choices: 'fit' or prefac value (e.g. 0.55)
RSEP_FIT = 'fit'
#RSEP_FIT = 0.55

#SAVE_RFIT = None    # csv file path, to save RSEP_FIT results  (appends to file if it exists)
SAVE_RFIT = SAVE

#XSEP_FIT = False    # fit x separation trend (vs |i-j|) ?  (also shows in plot)  choices: 'fit' or prefac value (e.g. 1)
XSEP_FIT = 'fit'
#XSEP_FIT = 1.

SAVE_XFIT = None    # csv file path, to save XSEP_FIT results  (appends to file if it exists)
#SAVE_XFIT = SAVE

RrscSEP_FIT = False # fit resclaed R separation trend ?
#RrscSEP_FIT = 'fit'
#RrscSEP_FIT = 1.

SAVE_RrscFIT = None # csv file path, to save rescaled RrscSEP_FIT
#SAVE_RrscFIT = SAVE


# settings for second set of results and differences comparisons, e.g. changes upon salt increase
# Note: function 'SAVE_2_f' is used for saving plots of second case (far below) in separate directory

RdiffSEP_FIT = False    # fit R difference separation ?
#RdiffSEP_FIT = 'fit'
#RdiffSEP_FIT = 0.01

SAVE_RdiffFIT = None    # csv file path, for RdiffSEP_FIT
#SAVE_RdiffFIT = SAVE

XdiffSEP_FIT = False    # fit x difference separation ?
#XdiffSEP_FIT = 'fit'
#XdiffSEP_FIT = 0.005

SAVE_XdiffFIT = None    # csv file path, for XdiffSEP_FIT
#SAVE_XdiffFIT = SAVE

RrscdiffSEP_FIT = False    # fit rescaled R difference separation ?
#RrscdiffSEP_FIT = 'fit'
#RrscdiffSEP_FIT = 0.005

SAVE_RrscdiffFIT = None    # csv file path, for RdiffSEP_FIT
#SAVE_RrscdiffFIT = SAVE

SAVE_2_f = lambda sdir: (sdir + '2') if (type(sdir) is str) else None       # format function to make second set directories

#SAVE_2 = False  # save plots of second set (used in differences) ?  e.g. higher salt, different model
SAVE_2 = SAVE_2_f(SAVE)
#SAVE_2 = SAVE + '2'

#SHOW_2 = False  # show plots of second set (used in differences) ?
SHOW_2 = True


##  COMMAND LINE ARGUMENTS

sel = seqname
if len(args) > 1:
    sel = args[1]

# special check for wenwei / mittal simulated IDPs
try:
    sel = int(sel)
    wsel = (sel - 5130) if (sel > 5129) else sel    # shifted labels for IDPs with charge (do NOT shift if uncharged)
    sname = str(sel)
except ValueError:
    sname = sel

if len(args) > 2:
    IJmin = int(args[2])

if len(args) > 3:
    NORM = eval(args[3].capitalize())


##  DIRECTORIES, FILES, LABELS

## EKV examples
#res1_file = f'./sample_ij_arrays/calvados_dij/{sel:}_clones_avg_dij.npy'
#res1_lbl = 'calvados'
#
#res2_file = None
#res2_lbl = None
#
#res1_file_2 = None
#res1_lbl_2 = None
#
#res2_file_2 = None
#res2_lbl_2 = None


## wenwei / mittal random construction IDPs
#res1_file = f'./sample_results/theory_xij/{sel:}_xij_T300_cs100.npy'
#res1_lbl = 'calc.'
#
##res2_file = None
#res2_file = f'./sample_results/wenwei_dij/dijrms_idp_{wsel:}.npy'
#
##res2_lbl = None
#res2_lbl = 'sim.'
#
#res1_file_2 = None
#res1_lbl_2 = None
#
#res2_file_2 = None
#res2_lbl_2 = None


## tesei / lindorff-larsen IDRome
#res1_file = f'./sample_results/theory_xij/{sel:}_xij_T310_cs150.npy'
#res1_lbl = 'calc.'
#
##res2_file = None
#res2_file = f'./sample_results/calvados_dij/{sel:}_dij_cs150.npy'
#
##res2_lbl = None
#res2_lbl = 'sim.'
#
##res1_file_2 = None
#res1_file_2 = f'./sample_results/theory_xij/{sel:}_xij_T310_cs250.npy'
#
##res1_lbl_2 = None
#res1_lbl_2 = 'calc. (250mM)'
#
##res2_file_2 = None
#res2_file_2 = f'./sample_results/calvados_dij/{sel:}_dij_cs250.npy'
#
##res2_lbl_2 = None
#res2_lbl_2 = 'sim. (250mM)'


## fcp1
#res1_file = f'./sample_results/theory_xij/{sel:}_xij_chargereg_PML_T300_cs160.npy'
#res1_lbl = 'charge reg.'
#
##res2_file = None
#res2_file = f'./sample_results/theory_xij/{sel:}_xij_fullion_PML_T300_cs160.npy'
#
##res2_lbl = None
#res2_lbl = 'full ion.'
#
##res1_file_2 = None
#res1_file_2 = f'./sample_results/theory_xij/{sel:}_xij_chargereg_PML_T300_cs660.npy'
#
##res1_lbl_2 = None
#res1_lbl_2 = 'charge reg. (hs)'
#
##res2_file_2 = None
#res2_file_2 = f'./sample_results/theory_xij/{sel:}_xij_fullion_PML_T300_cs660.npy'
#
##res2_lbl_2 = None
#res2_lbl_2 = 'full ion. (hs)'

## Linker sequences
##res1_file = f'../out/schuler linkers maps/alexa/M1/cion/xij/{sel:}_xij_T293_cs150.npy'
##res1_file = f'../out/schuler linkers maps/cycf/M1/cion/xij/{sel:}_xij_T293_cs150.npy'
res1_file = f'../out/schuler linkers maps/alexa/aug2025 M1/cion/xij/{sel:}_xij_T293_cs150.npy'
#res1_file = f'../out/schuler linkers maps/cycf/aug2025 M1/cion/xij/{sel:}_xij_T293_cs150.npy'
#res1_file = f'../out/schuler linkers maps/cycf/aug2025 M1/cion_rho/xij/{sel:}_xij_T293_cs150.npy'
#res1_lbl = 'M1'
res1_lbl = 'Alexa M1'

#res2_file = f'../out/schuler linkers maps/alexa/M0/fullion/xij/{sel:}_xij_T293_cs150.npy'
#res2_file = f'../out/schuler linkers maps/cycf/M0/fullion/xij/{sel:}_xij_T293_cs150.npy'
#res2_lbl = 'M0'
#res2_file = f'../out/schuler linkers maps/cycf/aug2025 M1/cion/xij/{sel:}_xij_T293_cs150.npy'
res2_file = f'../out/schuler linkers maps/cycf/aug2025 M1/cion_rho/xij/{sel:}_xij_T293_cs150.npy'
res2_lbl = 'Cy/CF M1'

res1_file_2 = None
res1_lbl_2 = None
res2_file_2 = None
res2_lbl_2 = None

## EKV examples
#res1_file = f'../data/EKV_dij_calvados-cut8nm/dijs/{sel:}_clones_avg_dij.npy'
#res1_lbl = 'sim.'
#
#res2_file = None
#res2_lbl = None
#
#res1_file_2 = f'../out/mittal ekv ML 01/round 2 - min_scalar method/xij/{sel:}_xij_T300_cs100.npy'
#res1_lbl_2 = 'calc.'
#
#res2_file_2 = None
#res2_lbl_2 = None



####    *    ####    *    ####    *    ####    *    ####
####    EVERYTHING BELOW HERE SHOULD BE AUTOMATIC   ####
####    *    ####    *    ####    *    ####    *    ####


##  LOAD AND PROCESS FROM FILES: res1 (necessary) and its difference partner (optional), similar for res2 for comparison
res1_xij, res1_dij, ijdiff, aijdiff = xpm.get_xij_dij(res1_file, \
        triangle='l', minsep=IJmin)     # main result 1 (necessary); ijdiffs established
if res1_file_2:
    res1_xij_2, res1_dij_2, ijdiff, aijdiff = xpm.get_xij_dij(res1_file_2, \
        triangle='l', minsep=IJmin)     # second result 1, for difference (if desired)
else:
    res1_xij_2 = np.zeros(res1_xij.shape)
    res1_dij_2 = res1_xij_2.copy()
res1_diff_xij = res1_xij_2 - res1_xij
res1_diff_dij = res1_dij_2 - res1_dij

if res2_file:
    res2_xij, res2_dij, ijdiff, aijdiff = xpm.get_xij_dij(res2_file, \
        triangle='u', minsep=IJmin)         # main result 2, as comparison against main result 1
    if res2_file_2:
        res2_xij_2, res2_dij_2, ijdiff, aijdiff = xpm.get_xij_dij(res2_file_2, \
            triangle='u', minsep=IJmin)     # second result 2, for difference (if desired)
    else:
        res2_xij_2 = np.zeros(res1_xij.shape)
        res2_dij_2 = res2_xij_2.copy()
else:
    res2_xij = np.zeros(res1_xij.shape)
    res2_dij = res2_xij.copy()
    res2_xij_2 = np.zeros(res1_xij.shape)
    res2_dij_2 = res2_xij_2.copy()
res2_diff_xij = res2_xij_2 - res2_xij
res2_diff_dij = res2_dij_2 - res2_dij


##  MAKE PLOTS

# > xij plots (cmap 'bwr')
xpr, xrmsd = xpm.corr_plot(res1_xij, res2_xij, 'xij', ijdiff, arrlabels=(res1_lbl, res2_lbl), valnorm=NORM, minsep=IJmin, \
    savedir=SAVE, show=SHOW, fsize=FSIZE, title=TITLE, name=sname, lw=0, ms=5, mew=0.2, mec='k', marker='8', color="teal")

xpm.map_plot(res1_xij, res2_xij, 'xij', arrlabels=(res1_lbl, res2_lbl), valnorm=NORM, minsep=IJmin, savedir=SAVE, show=SHOW, \
    fsize=FSIZE, title=TITLE, name=sname, sym_cbar=SYM_CBAR, rnd_cbar=RND_CBAR, num_cbar=NUM_CBAR, origin=MAP_ORIGIN, cmap='bwr')

xa1, xv1, xa2, xv2, xfunc = xpm.sep_plot(res1_xij, res2_xij, 'xij', arrlabels=(res1_lbl, res2_lbl), valnorm=NORM, minsep=IJmin, \
        savedir=SAVE, show=SHOW, fit_prefac=XSEP_FIT, fit_outdir=SAVE_XFIT, fsize=FSIZE, title=TITLE, name=sname, \
        opts1={'ls':'-', 'color':'tab:blue'}, opts2={'ls':'-', 'color':'tab:red'}, \
        fitopts1={'ls':'--', 'color':'deepskyblue'}, fitopts2={'ls':'--', 'color':'magenta'})


# > Rij (dij) plots (cmap 'hot')
rpr, rrmsd = xpm.corr_plot(res1_dij, res2_dij, 'Rij', ijdiff, arrlabels=(res1_lbl, res2_lbl), valnorm=NORM, minsep=IJmin, \
    savedir=SAVE, show=SHOW, fsize=FSIZE, title=TITLE, name=sname, lw=0, ms=5, mew=0.2, mec='k', marker='o', color="tomato")

xpm.map_plot(res1_dij, res2_dij, 'Rij', arrlabels=(res1_lbl, res2_lbl), valnorm=NORM, minsep=IJmin, savedir=SAVE, show=SHOW, \
    fsize=FSIZE, title=TITLE, name=sname, sym_cbar=SYM_CBAR, rnd_cbar=RND_CBAR, num_cbar=NUM_CBAR, origin=MAP_ORIGIN, cmap='hot')

ra1, rv1, ra2, rv2, rfunc = xpm.sep_plot(res1_dij, res2_dij, 'Rij', arrlabels=(res1_lbl, res2_lbl), valnorm=NORM, minsep=IJmin, \
        savedir=SAVE, show=SHOW, fit_prefac=RSEP_FIT, fit_outdir=SAVE_RFIT, fsize=FSIZE, title=TITLE, name=sname, \
        opts1={'ls':'-', 'color':'tab:green'}, opts2={'ls':'-', 'color':'tab:orange'}, \
        fitopts1={'ls':'--', 'color':'lime'}, fitopts2={'ls':'--', 'color':'goldenrod'})


# > rescaled (cmap 'seismic')
if rescaleRIJ and RSEP_FIT:
    ## from Rij
    res1_drsc_r = xpm.scaled_Rij_from_dij(res1_dij, aijdiff, rv1, A=ra1, fcn=rfunc)
    res2_drsc_r = xpm.scaled_Rij_from_dij(res2_dij, aijdiff, rv2, A=ra2, fcn=rfunc) if rv2 else res2_dij

    rscpr, rscrmsd = xpm.corr_plot(res1_drsc_r, res2_drsc_r, 'Rij*', ijdiff, \
        arrlabels=(res1_lbl, res2_lbl), valnorm=NORM, minsep=IJmin, \
        savedir=SAVE, show=SHOW, fsize=FSIZE, title=TITLE, name=sname, \
        lw=0, ms=5, mew=0.2, mec='k', marker='p', color="darkviolet")

    xpm.map_plot(res1_drsc_r, res2_drsc_r, 'Rij*', arrlabels=(res1_lbl, res2_lbl), valnorm=NORM, minsep=IJmin, \
        savedir=SAVE, show=SHOW, fsize=FSIZE, title=TITLE, name=sname, \
        sym_cbar=SYM_CBAR, lim_cbar=RIJrsc_cbarlim, rnd_cbar=RND_CBAR, num_cbar=NUM_CBAR, origin=MAP_ORIGIN, cmap='seismic')

    rsca1r, rscv1r, rsca2r, rscv2r, rscfuncr = xpm.sep_plot(res1_drsc_r, res2_drsc_r, 'Rij*', \
            arrlabels=(res1_lbl, res2_lbl), valnorm=NORM, minsep=IJmin, \
            savedir=SAVE, show=SHOW, fit_prefac=RrscSEP_FIT, fit_outdir=SAVE_RrscFIT, fsize=FSIZE, title=TITLE, name=sname, \
            opts1={'ls':'-', 'color':'tab:purple'}, opts2={'ls':'-', 'color':'tab:olive'}, \
            fitopts1={'ls':'--', 'color':'mediumorchid'}, fitopts2={'ls':'--', 'color':'darkkhaki'})


# > second set, and diffs (cmap 'bwr')
if res1_file_2:
    SAVE_XFIT_2 = SAVE_2_f(SAVE_XFIT)
    ## xij second set (i.e. 'res1_xij_2' , 'res2_xij_2')
    xpr_2, xrmsd_2 = xpm.corr_plot(res1_xij_2, res2_xij_2, 'xij', ijdiff, \
        arrlabels=(res1_lbl_2, res2_lbl_2), valnorm=NORM, minsep=IJmin, \
        savedir=SAVE_2, show=SHOW_2, fsize=FSIZE, title=TITLE, name=sname, \
        lw=0, ms=5, mew=0.2, mec='k', marker='8', color="teal")

    xpm.map_plot(res1_xij_2, res2_xij_2, 'xij', arrlabels=(res1_lbl_2, res2_lbl_2), valnorm=NORM, minsep=IJmin, \
        savedir=SAVE_2, show=SHOW_2, fsize=FSIZE, title=TITLE, name=sname, \
        sym_cbar=SYM_CBAR, rnd_cbar=RND_CBAR, num_cbar=NUM_CBAR, origin=MAP_ORIGIN, cmap='bwr')

    xa1_2, xv1_2, xa2_2, xv2_2, xfunc_2 = xpm.sep_plot(res1_xij_2, res2_xij_2, 'xij', arrlabels=(res1_lbl_2, res2_lbl_2), \
            valnorm=NORM, minsep=IJmin, savedir=SAVE_2, show=SHOW_2, fit_prefac=XSEP_FIT, fit_outdir=SAVE_XFIT_2, \
            fsize=FSIZE, title=TITLE, name=sname, \
            opts1={'ls':'-', 'color':'tab:blue'}, opts2={'ls':'-', 'color':'tab:red'}, \
            fitopts1={'ls':'--', 'color':'deepskyblue'}, fitopts2={'ls':'--', 'color':'magenta'})

    SAVE_RFIT_2 = SAVE_2_f(SAVE_RFIT)
    ## Rij (dij) second set (i.e. 'res1_dij_2' , 'res2_dij_2')
    rpr_2, rrmsd_2 = xpm.corr_plot(res1_dij_2, res2_dij_2, 'Rij', ijdiff, \
        arrlabels=(res1_lbl_2, res2_lbl_2), valnorm=NORM, minsep=IJmin, \
        savedir=SAVE_2, show=SHOW_2, fsize=FSIZE, title=TITLE, name=sname, \
        lw=0, ms=5, mew=0.2, mec='k', marker='o', color="tomato")

    xpm.map_plot(res1_dij_2, res2_dij_2, 'Rij', arrlabels=(res1_lbl_2, res2_lbl_2), valnorm=NORM, minsep=IJmin, \
        savedir=SAVE_2, show=SHOW_2, fsize=FSIZE, title=TITLE, name=sname, \
        sym_cbar=SYM_CBAR, rnd_cbar=RND_CBAR, num_cbar=NUM_CBAR, origin=MAP_ORIGIN, cmap='hot')

    ra1_2, rv1_2, ra2_2, rv2_2, rfunc_2 = xpm.sep_plot(res1_dij_2, res2_dij_2, 'Rij', arrlabels=(res1_lbl_2, res2_lbl_2), \
            valnorm=NORM, minsep=IJmin, savedir=SAVE_2, show=SHOW_2, fit_prefac=RSEP_FIT, fit_outdir=SAVE_RFIT_2, \
            fsize=FSIZE, title=TITLE, name=sname, \
            opts1={'ls':'-', 'color':'tab:green'}, opts2={'ls':'-', 'color':'tab:orange'}, \
            fitopts1={'ls':'--', 'color':'lime'}, fitopts2={'ls':'--', 'color':'goldenrod'})

    if rescaleRIJ and RSEP_FIT:
        SAVE_RrscFIT_2 = SAVE_2_f(SAVE_RrscFIT)
        ## from Rij
        res1_drsc_r_2 = xpm.scaled_Rij_from_dij(res1_dij_2, aijdiff, rv1_2, A=ra1_2, fcn=rfunc_2)
        res2_drsc_r_2 = xpm.scaled_Rij_from_dij(res2_dij_2, aijdiff, rv2_2, A=ra2_2, fcn=rfunc_2) if rv2_2 else res2_dij_2

        rscpr_2, rscrmsd_2 = xpm.corr_plot(res1_drsc_r_2, res2_drsc_r_2, 'Rij*', ijdiff, \
            arrlabels=(res1_lbl_2, res2_lbl_2), valnorm=NORM, minsep=IJmin, \
            savedir=SAVE_2, show=SHOW_2, fsize=FSIZE, title=TITLE, name=sname, \
            lw=0, ms=5, mew=0.2, mec='k', marker='p', color="darkviolet")

        xpm.map_plot(res1_drsc_r_2, res2_drsc_r_2, 'Rij*', arrlabels=(res1_lbl_2, res2_lbl_2), valnorm=NORM, minsep=IJmin, \
            savedir=SAVE_2, show=SHOW_2, fsize=FSIZE, title=TITLE, name=sname, sym_cbar=SYM_CBAR, rnd_cbar=RND_CBAR, \
            lim_cbar=RIJrsc_cbarlim, num_cbar=NUM_CBAR, origin=MAP_ORIGIN, cmap='seismic')

        rsca1r_2, rscv1r_2, rsca2r_2, rscv2r_2, rscfuncr_2 = xpm.sep_plot(res1_drsc_r_2, res2_drsc_r_2, 'Rij*', \
                arrlabels=(res1_lbl_2, res2_lbl_2), valnorm=NORM, minsep=IJmin, \
                savedir=SAVE_2, show=SHOW_2, fit_prefac=RrscSEP_FIT, fit_outdir=SAVE_RrscFIT_2, \
                fsize=FSIZE, title=TITLE, name=sname, \
                opts1={'ls':'-', 'color':'tab:purple'}, opts2={'ls':'-', 'color':'tab:olive'}, \
                fitopts1={'ls':'--', 'color':'mediumorchid'}, fitopts2={'ls':'--', 'color':'darkkhaki'})

    res2_diff_lbl = res2_lbl if res2_file_2 else None
    ## xij diff
    xdpr, xdrmsd = xpm.corr_plot(res1_diff_xij, res2_diff_xij, 'diffxij', ijdiff, \
        arrlabels=(res1_lbl, res2_diff_lbl), valnorm=NORM, minsep=IJmin, \
        savedir=SAVE, show=SHOW, fsize=FSIZE, title=TITLE, name=sname, \
        lw=0, ms=5, mew=0.2, mec='k', marker='8', color="olivedrab")

    xpm.map_plot(res1_diff_xij, res2_diff_xij, 'diffxij', arrlabels=(res1_lbl, res2_diff_lbl), valnorm=NORM, minsep=IJmin, \
        savedir=SAVE, show=SHOW, fsize=FSIZE, title=TITLE, name=sname, \
        sym_cbar=SYM_CBAR, rnd_cbar=RND_CBAR, num_cbar=NUM_CBAR, origin=MAP_ORIGIN, cmap='bwr')

    xda1, xdv1, xda2, xdv2, xfunc = xpm.sep_plot(res1_diff_xij, res2_diff_xij, 'diffxij', arrlabels=(res1_lbl, res2_diff_lbl), \
            valnorm=NORM, minsep=IJmin, \
            savedir=SAVE, show=SHOW, fit_prefac=XdiffSEP_FIT, fit_outdir=SAVE_XdiffFIT, fsize=FSIZE, title=TITLE, name=sname, \
            opts1={'ls':'-', 'color':'tab:brown'}, opts2={'ls':'-', 'color':'tab:pink'}, \
            fitopts1={'ls':'--', 'color':'peru'}, fitopts2={'ls':'--', 'color':'hotpink'})

    ## Rij (dij) diff
    rdpr, rdrmsd = xpm.corr_plot(res1_diff_dij, res2_diff_dij, 'diffRij', ijdiff, \
        arrlabels=(res1_lbl, res2_diff_lbl), valnorm=NORM, minsep=IJmin, \
        savedir=SAVE, show=SHOW, fsize=FSIZE, title=TITLE, name=sname, \
        lw=0, ms=5, mew=0.2, mec='k', marker='o', color="mediumvioletred")

    xpm.map_plot(res1_diff_dij, res2_diff_dij, 'diffRij', arrlabels=(res1_lbl, res2_diff_lbl), valnorm=NORM, minsep=IJmin, \
        savedir=SAVE, show=SHOW, fsize=FSIZE, title=TITLE, name=sname, \
        sym_cbar=SYM_CBAR, rnd_cbar=RND_CBAR, num_cbar=NUM_CBAR, origin=MAP_ORIGIN, cmap='bwr')

    rda1, rdv1, rda2, rdv2, rfunc = xpm.sep_plot(res1_diff_dij, res2_diff_dij, 'diffRij', arrlabels=(res1_lbl, res2_diff_lbl), \
        valnorm=NORM, minsep=IJmin, \
            savedir=SAVE, show=SHOW, fit_prefac=RdiffSEP_FIT, fit_outdir=SAVE_RdiffFIT, fsize=FSIZE, title=TITLE, name=sname, \
            opts1={'ls':'-', 'color':'tab:gray'}, opts2={'ls':'-', 'color':'tab:cyan'}, \
            fitopts1={'ls':'--', 'color':'darkgray'}, fitopts2={'ls':'--', 'color':'darkturquoise'})

    if rescaleRIJ and RdiffSEP_FIT:
        ## rescaled Rij diff
        res1_drsc_r_diff = res1_drsc_r_2 - res1_drsc_r
        res2_drsc_r_diff = res2_drsc_r_2 - res2_drsc_r
        rscdpr, rscdrmsd = xpm.corr_plot(res1_drsc_r_diff, res2_drsc_r_diff, 'diffRij*', ijdiff, \
            arrlabels=(res1_lbl, res2_lbl), valnorm=NORM, minsep=IJmin, savedir=SAVE, show=SHOW, fsize=FSIZE, title=TITLE, \
            name=sname, lw=0, ms=5, mew=0.2, mec='k', marker='p', color="rebeccapurple")

        xpm.map_plot(res1_drsc_r_diff, res2_drsc_r_diff, 'diffRij*', \
            arrlabels=(res1_lbl, res2_lbl), valnorm=NORM, minsep=IJmin, \
            savedir=SAVE, show=SHOW, fsize=FSIZE, title=TITLE, name=sname, \
            sym_cbar=SYM_CBAR, rnd_cbar=RND_CBAR, num_cbar=NUM_CBAR, origin=MAP_ORIGIN, cmap='seismic')

        rsca1r_d, rscv1r_d, rsca2r_d, rscv2r_d, rscfuncr_d = xpm.sep_plot(res1_drsc_r_diff, res2_drsc_r_diff, 'diffRij*', \
                arrlabels=(res1_lbl, res2_lbl), valnorm=NORM, minsep=IJmin, \
                savedir=SAVE, show=SHOW, fit_prefac=RrscdiffSEP_FIT, fit_outdir=SAVE_RrscdiffFIT, \
                fsize=FSIZE, title=TITLE, name=sname, \
                opts1={'ls':'-', 'color':'blueviolet'}, opts2={'ls':'-', 'color':'mediumseagreen'}, \
                fitopts1={'ls':'--', 'color':'mediumpurple'}, fitopts2={'ls':'--', 'color':'mediumaquamarine'})

