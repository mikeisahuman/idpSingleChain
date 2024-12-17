##  Mike Phillips, 8/2/2023
##  File for evaluating 'x' using predicted 'w2' from Parrot / CNN ML architecture.
##   > i.e. using a subset of Lindorff-Larson 30k set, for testing
##  * Re-vamped 6/11/2024 *
##     > now purely results comparison code  (no model or calculation, aside from basic statistics)
##  Updates:
##      * 'compare_res' compiles results as dictionary with reference information for comparison
##      * 'plot_Rdiff' uses comparison dictionary from 'compare_res' to show differences vs. choice of variable (N, SCD, w2)
##      * 'plot_corr' uses comparison dictionary to show correlation of a given quantity (Ree, Rg, x, w2)
##      * 'plot_charge' uses comparison dict. to show sequence distribution across charge content, net charge, absolute net charge, or N
##      * 'plot_2sets' takes two comparison dictionaries to compare against each other; e.g. for changes in prediction as salt changes
##      * 'draw_from_dist' randomly draws sequences from a given distribution, e.g. lengths N
##      * 'find_seqs' tabulates sequences having the desired properties (N, minimal charge content, minimal abs. net charge, max/min Ree)
##      * 'get_rrmsd' calculates Pearson 'r' and RMS difference between true and predicted values of Ree, x, w2
##      * 'all_plots' makes and saves all plots of interest (correlations of difference vs. var, and pred vs. true)


import numpy as np
import myPlotOptions as mpo
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd

from os import path

rand = np.random.rand

import sys

argv = sys.argv


# PARAMETERS  (used in specifying results file)

HPlist = ('A', 'I', 'L', 'M', 'F', 'P', 'W', 'Y', 'V')      # list of hydrophobic aminos (for tracking HP content)

#phval = 4          # value of pH
#phex = ['H','C','Y']   # exclusions in translating charge from pH
phval = None
phex = []
phex_str = ('-no' + ''.join(phex)) if (len(phex)>0) else ''

#csmm = 50          # salt concentration in mM
csmm = 150        # estimate based on CALVADOS

w3 = 0.2    # standard value of w3
if len(argv) > 1:
    w3 = float(argv[1])
w3_str = str(w3).replace('.','')


# FILE SETTINGS

# IDRome set and results
ref_file = "../IDRome_sequences/IDRome28k.csv"

results_dir = "../out/"
results_file = path.join(results_dir, f"IDRome_out.csv")

N_head = "N"
SCD_head = "SCD"

name_head = "names"
seq_head = "seqs"
w2_head = "pred_w2"

ref_name_head = "seq_name"
ref_seq_head = "fasta"
ref_w2_head = 'w2'

x_head = "x"
R_head = "Ree[nm]"
ref_x_head = "x"
ref_R_head = "Ree/nm"           # reference Ree heading from input file (from Lindorff-Larsen)
ref_Rg_head = "Rg/nm"

res_ref_R_head = "ref_" + R_head        # reference Ree heading from results file
#res_ref_R_head = ref_R_head


##########


# establish dictionary of results and reference values, with differences; extra sequence info optional  (charge, hydrophobicity, aminos)
def compare_res(file=results_file, compfile=ref_file, charge=True, hp=True, seqs=False, xw2=True):
    print(f"\n * Extracting results x, Ree, w2 from  '{file:}' ...")
    print(f"\n * Pulling reference x, Rg, w2, N, SCD from  '{compfile:}' ...\n")
    comp_dict = {'names':[], 'Rdiff':[], 'res_R':[], 'ref_R':[], 'xdiff':[], 'res_x':[], 'ref_x':[], \
        'Rgdiff':[], 'res_Rg':[], 'ref_Rg':[], 'W2diff':[], 'W2pred':[], 'W2ref':[], 'N':[], 'SCD':[]}
    rf = pd.read_csv(file)
    rf = rf.sort_values(name_head)
    cf = pd.read_csv(compfile)
    cnonecheck = ( np.isnan(cf[ref_w2_head]) )
    if any(cnonecheck):
        print(f"\n! Reference file count of 'None' values of w2:\t{sum(cnonecheck)}\n{cf[cnonecheck]}")
        cf = cf[~cnonecheck]
    cextracheck = np.asarray( [ s in rf[name_head].values for s in cf[ref_name_head].values ] )
    cextra_count = (~cextracheck).sum()
    if cextra_count > 0:
        print(f"\n >>  Note: reference file has more sequences than results file  (count = {cextra_count:})  <<")
    cf = cf[cextracheck]
    cf = cf.sort_values(ref_name_head)
    rnonecheck = ( np.isnan(rf[w2_head]) )
    if any(rnonecheck):
        print(f"\n!! Results file has 'None' values of w2:\t{sum(rnonecheck)}\n{rf[rnonecheck]}")
        rf = rf[~rnonecheck]
        cf = cf[~rnonecheck]
    comp_dict.update({'names':list(rf[name_head]), 'res_R':np.asarray(rf[R_head]), 'SCD':np.asarray(cf[SCD_head])})
    if res_ref_R_head in rf:
        comp_dict.update({'ref_R':np.asarray(rf[res_ref_R_head])})
    else:
        comp_dict.update({'ref_R':np.asarray(cf[ref_R_head])})
    comp_dict.update({'Rdiff':(comp_dict['res_R'] - comp_dict['ref_R'])})
    if ref_Rg_head:
        comp_dict.update({'res_Rg':(comp_dict['res_R']/np.sqrt(6)), 'ref_Rg':cf[ref_Rg_head]})
        comp_dict.update({'Rgdiff':(comp_dict['res_Rg'] - comp_dict['ref_Rg'])})
    if xw2:
        comp_dict.update({'res_x':np.asarray(rf[x_head]), 'W2pred':np.asarray(rf[w2_head]), \
            'ref_x':np.asarray(cf[ref_x_head]), 'W2ref':np.asarray(cf[ref_w2_head])})
        comp_dict.update({'xdiff':(comp_dict['res_x']-comp_dict['ref_x']), 'W2diff':(comp_dict['W2pred']-comp_dict['W2ref'])})
    if N_head:
        comp_dict.update({'N':np.asarray(cf[N_head])})
    if charge:
        Np = np.asarray( [sum([sq.count(a)*SI.aminos[a] if (SI.aminos[a] > 0) else 0 for a in SI.aminos]) for sq in rf[seq_head]] )
        Nm = np.asarray( [sum([sq.count(a)*(-SI.aminos[a]) if (SI.aminos[a] < 0) else 0 for a in SI.aminos]) for sq in rf[seq_head]] )
        comp_dict.update({'Np':Np, 'Nm':Nm})
    if hp:
        Nhp = np.asarray( [sum([sq.count(a) for a in HPlist]) for sq in rf[seq_head]] )
        comp_dict.update({'Nhp':Nhp})
    if seqs:
        comp_dict.update({'seqs':rf[seq_head]})
    return comp_dict


# check difference in Ree (or x, Rg)  against some variable  (N, scd, w2ref, w2pred, w2diff)
def plot_Rdiff(resd, var='N', yvar='R', omitNone=True, SAVE=False):
    if yvar.lower() == 'r':
        Rdiff = np.asarray(resd['Rdiff'])
        YLBL = "Ree difference (pred - LL)  [nm]"
    elif yvar.lower() == 'x':
        Rdiff = np.asarray(resd['xdiff'])
        YLBL = "x difference (pred - LL)"
    elif yvar.lower() == 'rg':
        Rdiff = np.asarray(resd['Rgdiff'])
        YLBL = "Rg difference (pred - LL)  [nm]"
    else:
        print(f"\nERROR: improper y-variable choice '{yvar:}'.\n")
        return
    w2ref = np.asarray(resd['W2ref'])

    MKSIZE = 6
    LNSTY = "--"

    fig, ax = plt.subplots(figsize=(9.3,7))

    if var.lower() == "n":
        comp_arr = np.asarray(resd['N'])
        XLBL = "length N"
        MKSTY = "^"
        MKCOLOR = "tab:purple"
    elif var.lower() == "scd":
        comp_arr = np.asarray(resd['SCD'])
        XLBL = "SCD"
        MKSTY = "s"
        MKCOLOR = "tab:green"
        MKSIZE = 5
    elif var.lower() == "w2diff":
        comp_arr = np.asarray(resd['W2diff'])
        XLBL = "w2 difference  (predicted - original)"
        MKSTY = "v"
        MKCOLOR = "tab:red"
        ax.plot([0,0],[Rdiff.min(),Rdiff.max()], LNSTY, color="gray", lw=1.5, zorder=4)
        ax.plot([comp_arr[comp_arr!=None].min(),comp_arr[comp_arr!=None].max()],[0,0], LNSTY, color="gray", lw=1.5, zorder=4)
    elif var.lower() == "w2pred":
        comp_arr = np.asarray(resd['W2pred'])
        XLBL = "predicted w2"
        MKSTY = "p"
        MKCOLOR = "darkmagenta"
    elif var.lower() == "w2ref":
        comp_arr = np.asarray(resd['W2ref'])
        XLBL = "true w2"
        MKSTY = "p"
        MKCOLOR = "blueviolet"
    else:
        print(f"\nERROR: improper variable choice '{var:}'.\n")
        return

    # check for any 'None' entries in 'w2ref', omit if present
    checkNone = (w2ref == None)
    countNone = checkNone.sum()
    if (countNone > 0) and ((var.lower() in ('w2ref', 'w2diff')) or omitNone):
        print(f"\nOmitting 'None' entries of w2ref for '{var:}', total {countNone:}\n")
        comp_arr = comp_arr[~checkNone]
        Rdiff = Rdiff[~checkNone]

    ax.plot(comp_arr, Rdiff, MKSTY, color=MKCOLOR, ms=MKSIZE)

    pr = pearsonr(Rdiff, comp_arr)
    r = pr.statistic

    ax.set_title(f"Pearson  r = {r:.5f}")
    ax.set_xlabel(XLBL)
    ax.set_ylabel(YLBL)
    fig.tight_layout()
    if SAVE:
        fig.savefig(SAVE)
    else:
        plt.show()
    plt.close()
    return r


# check correlation between predicted and reference (Ree, x, Rg, w2)
def plot_corr(resd, var='R', omitNone=True, SAVE=False):
    w2ref = np.asarray(resd['W2ref'])

    if var.lower() == 'r':
        resv = np.asarray(resd['res_R'])
        refv = np.asarray(resd['ref_R'])
        MKSTY = "o"
        MKCOLOR = "tab:blue"
        XLBL = "simulated  Ree [nm]"
        YLBL = "predicted  Ree [nm]"
    elif var.lower() == 'x':
        resv = np.asarray(resd['res_x'])
        refv = np.asarray(resd['ref_x'])
        MKSTY = "h"
        MKCOLOR = "darkorange"
        XLBL = "true  x"
        YLBL = "predicted  x"
    elif var.lower() == 'rg':
        resv = np.asarray(resd['res_Rg'])
        refv = np.asarray(resd['ref_Rg'])
        MKSTY = "D"
        MKCOLOR = "mediumvioletred"
        XLBL = "simulated  Rg [nm]"
        YLBL = "predicted  Rg  (Ree/$\sqrt{6}$)  [nm]"
    elif var.lower() == 'w2':
        resv = np.asarray(resd['W2pred'])
        refv = np.asarray(resd['W2ref'])
        MKSTY = "8"
        MKCOLOR = "teal"
        XLBL = "true  w2"
        YLBL = "predicted  w2"
    else:
        print(f"\nERROR: improper variable choice '{var:}'.\n")
        return

    # check for any 'None' entries, omit if present
    checkNone = (w2ref == None)
    countNone = checkNone.sum()
    if (countNone > 0) and ((var.lower() in ('w2ref', 'w2diff')) or omitNone):
        print(f"\nOmitting 'None' entries of {var:}, total {countNone:}\n")
        refv = refv[~checkNone]
        resv = resv[~checkNone]

    minv = min( resv.min(), refv.min() )
    maxv = max( resv.max(), refv.max() )
    pr = pearsonr(refv, resv)
    r = pr.statistic

    MKSIZE = 5
    LNSTY = "k--"

    fig, ax = plt.subplots(figsize=(9.3,7))
    ax.plot(refv, resv, MKSTY, color=MKCOLOR, ms=MKSIZE)
    ax.plot([minv,maxv],[minv,maxv], LNSTY, lw=1.5)
    ax.set_title(f"Pearson  r = {r:.5f}")
    ax.set_xlabel(XLBL)
    ax.set_ylabel(YLBL)
    fig.tight_layout()
    if SAVE:
        fig.savefig(SAVE)
    else:
        plt.show()
    plt.close()
    return r


# check sequence property (charge content, net charge, absolute net, or length N) - when results dictionary has that info included
def plot_charge(resd, charge='content', bins=50, density=True):
    Narr = np.asarray(resd['N'])
    if charge.lower() !='n':
        NParr, NMarr = np.asarray(resd['Np']), np.asarray(resd['Nm'])
        FParr, FMarr = NParr/Narr, NMarr/Narr

    if charge.lower() == "content":
        var = FParr + FMarr
        xlbl = r"charge content,  $f_+ + f_-$"
        color = "C0"
    elif charge.lower() == "net":
        var = FParr - FMarr
        xlbl = r"net charge,  $f_+ - f_-$"
        color = "C1"
    elif charge.lower() == "abs":
        var = np.abs(FParr - FMarr)
        xlbl = r"abs. net charge,  $|f_+ - f_-|$"
        color = "C2"
    elif charge.lower() == "n":
        var = Narr
        xlbl = r"length,  $N$"
        color = "C3"
    else:
        print(f"Invalid charge choice : '{charge:}'")
        return

    ylbl = "probability (density)" if density else "count"

    c,b = np.histogram(var, bins=bins, density=density)
    mean, std = var.mean(), var.std()

    fig, ax = plt.subplots(figsize=(9,7))
    ax.stairs(c,b, color=color, lw=2)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.plot([mean,mean], [0,100], 'k--', lw=1.2)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    fig.tight_layout()
    plt.show()
    plt.close()
    return


# check distributions of two sets of results, optionally as difference ('raw' or 'perc')
def plot_2sets(resd1, resd2, lb1='150', lb2='50', key='res_R', diff=False, coilglob=False, bins=50, perc_thr=20, density=True, SAVE=None):
    srt1 = np.argsort(resd1['names'])
    srt2 = np.argsort(resd2['names'])
    l1 = np.asarray(resd1[key])[srt1]
    l2 = np.asarray(resd2[key])[srt2]
    outd1 = {k:np.asarray(resd1[k])[srt1] for k in resd1}
    outd2 = {k:np.asarray(resd2[k])[srt2] for k in resd2}

    ylbl = "probability (density)" if density else "count"
    lbv = r'R_{ee}' if ('R' in key) else 'var'
    lbv = 'x' if ('x' in key) else lbv

    if (lbv == 'x') and coilglob:
        cgcheck = ((l1 > 1.) & (l2 < 1.)) | ((l1 < 1.) & (l2 > 1.))
        print(f"\nCount of sequences crossing x=1:\n\tNcg = {cgcheck.sum()}\n\tNtot = {len(l1)}  ;  fraction = {cgcheck.sum()/len(l1)}\n")
        l1 = l1[cgcheck]
        l2 = l2[cgcheck]
        outd1 = {k:outd1[k][cgcheck] for k in outd1}
        outd2 = {k:outd2[k][cgcheck] for k in outd2}

    if diff == 'raw':
        var = l1-l2
        var2 = None
        xlbl = (r"$%s^{%s} - %s^{%s}$") % (lbv, lb1, lbv, lb2)
        if 'R' in key:
            xlbl += "  [nm]"
        color = "tab:red"
        leg1, leg2 = None, None
    elif diff == 'perc':
        var = (l1-l2)*100/l1
        var2 = None
        xlbl = (r"percent change,  $(%s^{%s} - %s^{%s})/%s^{%s}$") % (lbv, lb1, lbv, lb2, lbv, lb1)
        color = "tab:purple"
        leg1, leg2 = None, None
#        perc_thr = 75
        perc_check = np.abs(var) >= perc_thr
        print(f"\nCount of sequences changing at least {perc_thr}%:\n\tNpc = {perc_check.sum()}\n\tNtot = {len(l1)}  ;  fraction = {perc_check.sum()/len(l1)}\n")
    else:
        var = l1
        var2 = l2
        xlbl = (r"${lbv:}$").format(lbv=lbv)
        if 'R' in key:
            xlbl += "  [nm]"
        color = "C0"
        color2 = "C1"
        try:
            int(lb1), int(lb2)
            leg1 = (r"$c_s = {lb1:}$ mM").format(lb1=lb1)
            leg2 = (r"$c_s = {lb2:}$ mM").format(lb2=lb2)
        except ValueError:
            leg1 = (r"{lb1:}").format(lb1=lb1)
            leg2 = (r"{lb2:}").format(lb2=lb2)

    c,b = np.histogram(var, bins=bins, density=density)
    mean, std = var.mean(), var.std()

    fig, ax = plt.subplots(figsize=(9,7))
    ax.stairs(c,b, color=color, lw=2, label=leg1)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    if type(var2) != type(None):
        c2,b2 = np.histogram(var2, bins=bins, density=density)
        mean2, std2 = var2.mean(), var2.std()
        ax.stairs(c2,b2, color=color2, lw=2, label=leg2)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.plot([mean2,mean2], [0,500], '--', lw=1, color='gray')
        ax.legend()
    ax.plot([mean,mean], [0,500], '--', lw=1, color='black')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    fig.tight_layout()
    if SAVE:
        fig.savefig(SAVE)
    else:
        plt.show()
    plt.close()
    return outd1, outd2


# draw some number of sequences using the distribution (of selected variable) from given results dictionary
def draw_from_dist(resd, num=10, var='N'):
    Varr = np.asarray(resd[var])
    Vrng = range(Varr.min(), Varr.max())
    Vprobs = [(Varr==v).sum()/len(Varr) for v in Vrng]
    seld = {k:[] for k in resd}
    for n in range(num):
        is_sel = False
        tot = 0
        r = rand()
        for (p,v) in zip(Vprobs, Vrng):
            tot += p
            if tot > r:
                inds = np.where(Varr==v)[0]
                for i in inds:
                    if resd['names'][i] not in seld['names']:
                        for k in seld:
                            seld[k].append(resd[k][i])
                        is_sel = True
                        break
                if is_sel:
                    break
    return seld


# find sequences from results dictionary: given length, high charge content, near neutrality, min/max Ree (Rg)
def find_seqs(resd, N=50, content=0.20, absnet=0.05, hpmax=0.50, xrank=False):
    Narr, NParr, NMarr = np.asarray(resd['N']), np.asarray(resd['Np']), np.asarray(resd['Nm'])
    NHParr = np.asarray(resd['Nhp'])
    FParr, FMarr, FHParr = NParr/Narr, NMarr/Narr, NHParr/Narr
    charges = FParr + FMarr
    netcharge = np.abs(FParr - FMarr)

    if N:
        Ncheck = (Narr == N)        # indices to consider only sequences of given length N
    else:
        Ncheck = True               # default to include all lengths, if not specified
    Ccheck = (charges >= content)       # indices with charge content above threshold
    Qcheck = (netcharge <= absnet)      # indices with net charge below threshold
    Hcheck = (FHParr < hpmax)           # indices with hydrophobics below threshold
    allcheck = (Ncheck & Ccheck & Qcheck & Hcheck)

    if np.any(allcheck):
        Slist = np.asarray(resd['names'])
        SCDlist = np.asarray(resd['SCD'])
        xlist = np.asarray(resd['ref_x'])
        Rlist = np.asarray(resd['ref_R'])
        Rsel = Rlist[allcheck]
        if xrank:
            # giving most expanded sequences, based on x; first 'xrank
            args = np.argsort(xlist[allcheck])
            Ssort, SCDsort, xsort, Rsort = Slist[allcheck][args], SCDlist[allcheck][args], xlist[allcheck][args], Rlist[allcheck][args]
            Csort = FParr[allcheck][args] - FMarr[allcheck][args]
            HPsort = FHParr[allcheck][args]
            return {'seqsort':Ssort[:xrank], 'SCDsort':SCDsort[:xrank], 'Rsort':Rsort[:xrank], 'xsort':xsort[:xrank], \
                    'Csort':Csort[:xrank], 'HPsort':HPsort[:xrank]}
        else:
            Rmin, Rmax = Rsel.min(), Rsel.max()
            xmin, xmax = xlist[allcheck][Rsel==Rmin], xlist[allcheck][Rsel==Rmax]
            seqmin, seqmax = Slist[allcheck][Rsel==Rmin], Slist[allcheck][Rsel==Rmax]
            SCDmin, SCDmax = SCDlist[allcheck][Rsel==Rmin], SCDlist[allcheck][Rsel==Rmax]
            FPmin, FPmax = FParr[allcheck][Rsel==Rmin], FParr[allcheck][Rsel==Rmax]
            FMmin, FMmax = FMarr[allcheck][Rsel==Rmin], FMarr[allcheck][Rsel==Rmax]
            Cmin = ( FPmin[0]+FMmin[0] )
            Cmax = ( FPmax[0]+FMmax[0] )
            HPmin, HPmax = FHParr[allcheck][Rsel==Rmin], FHParr[allcheck][Rsel==Rmax]
            return {'seqmin':seqmin[0], 'xmin':xmin[0], 'Rmin':Rmin, 'SCDmin':SCDmin[0], 'Cmin':Cmin, 'HPmin':HPmin[0], \
                    'seqmax':seqmax[0], 'xmax':xmax[0], 'Rmax':Rmax, 'SCDmax':SCDmax[0], 'Cmax':Cmax, 'HPmax':HPmax[0]}
    else:
        print(f"\nNo sequences of length N={N:} have charge content above {content:} and below net charge {absnet:} and below HP {hpmax:}")
        return


# metrics : pearson r & rms difference
def get_rrmsd(resd, xw2=True):
    difkeys = (f'Rdiff', f'xdiff', 'W2diff')
    Rkeys = (f'res_R', f'ref_R')
    xkeys = (f'res_x', f'ref_x')
    wkeys = ('W2pred', 'W2ref')
    Rr = pearsonr(resd[Rkeys[0]],resd[Rkeys[1]]).statistic
    if xw2:
        xr = pearsonr(resd[xkeys[0]],resd[xkeys[1]]).statistic
        wr = pearsonr(resd[wkeys[0]],resd[wkeys[1]]).statistic
        Rrms, xrms, wrms = [np.sqrt(np.square(resd[x]).mean()) for x in difkeys]
    else:
        xr, xrms = None, None
        wr, wrms = None, None
        Rrms = np.sqrt(np.square(resd[difkeys[0]]).mean())
    return {'r_R':Rr, 'r_x':xr, 'rms_R':Rrms, 'rms_x':xrms, 'r_w':wr, 'rms_w':wrms}


# make lots of plots and save !
def all_plots(resd=None, middir='', res_dir=results_dir, xw2=True):
    if res_dir:
        if not resd:
            resd = compare_res()
        if xw2:
            vs = ('R', 'x', 'w2')
            xs = ('N', 'SCD', 'w2ref', 'w2pred', 'w2diff')
        else:
            vs = ('R',)
            xs = ('N', 'SCD')
        for v in vs:
            if v != 'w2':
                for x in xs:
                    sav = path.join(res_dir, middir, f"{v}-diff_vs_{x}.png")
                    plot_Rdiff(resd, x, v, SAVE=sav)
            csav = path.join(res_dir, middir, f"corr_{v}.png")
            plot_corr(resd, v, SAVE=csav)
    return

