##  Mike Phillips, April 2025
##  Set of tools for comparing xij and dij/Rij results
##  e.g. from calculation and simulation, or simulation under two conditions, etc.

"""
Toolkit includes:

[ Plot Style Paramters ]
    * 'rcParams' for control over various plot display details


[ Loading & Processing ]
    * 'load_check' : load xij or dij from file
        > ensures dij is in units of [nm], and proper loading in special cases, based on list of flags
    * 'dij_from_xij' / 'xij_from_dij' : convert between xij and dij
    * 'get_xij_dij' : quickly get both xij and dij
        > places in the same specified sector (upper or lower triangle; i.e. above or below matrix diagonal)
        > keeps only the values away from the main diagonal by the given amount (minsep)
        > also returns index difference arrays (direct and absolute) for use elsewhere
    * 'scaled_Rij_from_dij' : construct rescaled distance array from dij
        > requires homopolymer fit parameters (nu, a) for rescaling


[ Utilities ]
    * 'check_norm' : adjust arrays with independent normalization, establish related labels
    * 'var_label' : make nice labels from simple variable shorthands
    * 'flat_corr' : flatten and compare given arrays with Pearson's r and RMSD
    * 'cbar_props' : combine arrays and establish colorbar properties (normalization, ticks)
    * 'sep_means' : get mean at each separation |i-j| from given array
    * 'sep_fit' : fit homopolymer scaling to trend of separation means


** Note: all plotters support automatic saving to specified directory via 'SAVE' keyword,
    and all allow for standard 'plot' (or 'imshow') keywords - see each plotter for details.

[ Plotters ]
    * 'corr_plot' : plot correlation between two arrays
        > returns Pearson's r and RMSD
    * 'map_plot' : plot map of one or two arrays in the same image (on either side of diagonal)
    * 'sep_plot' : plot separation means of one or two arrays, with optional fitting to homopolymer scaling
        > plot keywords more complicated: separate options for two arrays, and two fits
        > returns fit parameters and function, for use with 'scaled_Rij_from_xij' / 'scaled_Rij_from_dij'
"""


import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
from os import path, makedirs, listdir

import myPlotOptions as mpo     # ** either import this, or uncomment rc settings below **

###  PLOT STYLE PARAMETERS
#
##FINAL = True       # use extra big sizes, for publication?
#FINAL = False
#
## style
#plt.style.use("seaborn-notebook")
#
## fonts
#if FINAL:
#    SMALL_SIZE = 14
#    MEDIUM_SIZE = 20
#    LARGE_SIZE = 22
#    LW = 2.5
#    LW_AX = 2.0
#else:
#    SMALL_SIZE = 10
#    MEDIUM_SIZE = 13
#    LARGE_SIZE = 16
#    LW = 2.0
#    LW_AX = 1.5
#
#plt.rc('font', size=MEDIUM_SIZE)        # controls default text sizes
#plt.rc('axes', titlesize=LARGE_SIZE)    # fontsize of the axes title
#plt.rc('axes', labelsize=LARGE_SIZE)    # fontsize of the x and y labels
##plt.rc('axes', labelpad=10)              # padding between axes and corresponding labels
#plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
#plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
#plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
#plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title
#plt.rc('legend', edgecolor="inherit")   # ensure legend inherits color by default
#plt.rc('xtick', bottom=True, top=True, direction="in")  # ensure ticks are on all sides
#plt.rc('ytick', left=True, right=True, direction="in")  #
#plt.rc('axes', linewidth=LW_AX)         # adjust AXES linewidth
#plt.rc('lines', linewidth=LW)           # adjust all plotted linewidths


##  LOADING AND PROCESSING TOOLS

angstrom_flags = ('LL_dij', 'EKV_dij', 'calvados', 'MD')        # flags in filepath that indicate angstrom units of dij
xij_flag = 'xij'        # flag in filepath indicating array as xij
wenwei_flag = 'wenwei'  # flag in filepath indicating wenwei simulation (of Zheng / Mittal set)

lsquare = 3.8*8/100     # squared length scale for converting between xij and dij  [nm^2]


# loading tool for xij / dij arrays from specified npy files
def load_check(arrfile, x_flag=xij_flag, a_flags=angstrom_flags, w_flag=wenwei_flag):
    """
    Load a numpy array of xij or dij/Rij. Check for special case(s) and conversion from Angstrom to nanometer as necessary.
    arrfile : filepath (string) to numpy array
    x_flag  : string that indicates filepath points to xij array (no unit)
    a_flags : list of strings that each indicate filepath points to dij/Rij array in Angstrom, to convert to nm
    w_flag  : string that indicates Wenwei (Zheng/Mittal) simulation dij; special case for retrieval of pertinent array
    output -> numpy array of shape (N,N) either unitless (if xij), or in nanometers (if dij/Rij)
    """
    # either xij or dij; if dij, ensure units are [nm]
    arr = np.load(arrfile)
    angstrom_checks = [(f in arrfile) for f in a_flags]
    if w_flag in arrfile:
        arr = arr[1]        # special case: index '1' refers to T=300K; simulation output formatting
    elif (x_flag not in arrfile) and any(angstrom_checks):
        arr = arr / 10      # conversion A -> nm for cases specified by 'angstrom_flags' in filepath
    return arr


# conversions between xij and dij
def dij_from_xij(xij, aijdiff, l2=lsquare):
    """
    Calculate dij/Rij from given xij array. Also needs array of absolute differences, |i-j|.
    xij     : array of xij values of shape (N,N)
    aijdiff : array of absolute differences in residue indices, same shape as xij
    l2      : squared length scale (float) l*b
    output -> dij array in units from l2 (nanometers by default)
    """
    dij = np.sqrt(l2 * aijdiff * xij)
    return dij

def xij_from_dij(dij, aijdiff, l2=lsquare):
    """
    Calculate xij from given dij/Rij array. Also needs array of absolute differences, |i-j|.
    dij     : array of dij (aka Rij) values of shape (N,N)
    aijdiff : array of absolute differences in residue indices, same shape as dij
    l2      : squared length scale (float) l*b; should have units matching dij (nanometers by default)
    output -> xij array (unitless)
    """
    xij = np.zeros(dij.shape)
    xij[aijdiff>0] = np.square(dij[aijdiff>0])/(l2*aijdiff[aijdiff>0])
    return xij

# get both xij and dij for a given file ; place in specified triangle (upper or lower)
def get_xij_dij(arrfile, triangle='lower', minsep=0, x_flag=xij_flag, a_flags=angstrom_flags, w_flag=wenwei_flag):
    """
    Load and format numpy array from one file to get corresponding xij, dij, differences (i-j), and absolute differences |i-j|.
    arrfile  : filepath (string) to numpy array
    triangle : string indicating placement of the array, either 'lower' (row > col) or 'upper' (row < col)
    minsep   : minimum separation of residues (int) to include in the final array (other values are set to zero)
    x_flag   : string that indicates filepath points to xij array (no unit)
    a_flags  : list of strings that each indicate filepath points to dij/Rij array in Angstrom, to convert to nm
    w_flag   : string that indicates Wenwei (Zheng/Mittal) simulation dij; special case for retrieval of pertinent array
    output  -> tuple of xij, dij [nm], ij differences, absolute ij differences
    """
    arr = load_check(arrfile, x_flag, a_flags, w_flag)
    # get direct (i-j) and absolute difference |i-j|
    rng = np.arange(len(arr))
    J,I = np.meshgrid(rng,rng)
    diff = I-J
    absdiff = np.abs(diff)
    # array placement - label ('triangle') convention is consistent with setting MAP_ORIGIN='upper'
    # also: ensure other triangle is zero
    if triangle.lower()[0] == 'l':
        if np.isclose(arr[-1,0], 0.):
            arr = arr.transpose()
        assert (arr[-1,0] > 0)
        if not np.isclose(arr[0,-1], 0.):
            arr[diff<0] = 0.
        assert np.isclose(arr[0,-1], 0.)
    else:
        if np.isclose(arr[0,-1], 0.):
            arr = arr.transpose()
        assert (arr[0,-1] > 0)
        if not np.isclose(arr[-1,0], 0.):
            arr[diff>0] = 0.
        assert np.isclose(arr[-1,0], 0.)
    # convert as necessary
    if xij_flag in arrfile:
        xij = arr
        dij = dij_from_xij(arr, absdiff)
    else:
        xij = xij_from_dij(arr, absdiff)
        dij = arr
    # omit values that are too small of separation
    sepnull = (absdiff < minsep)
    xij[sepnull] = 0
    dij[sepnull] = 0
    return (xij, dij, diff, absdiff)


# rescaled Rij based on homopolymer fit, from dij
def scaled_Rij_from_dij(dij, aijdiff, nu, A=np.sqrt(lsquare), fcn=(lambda s,v,a: a*np.power(s,v))):
    """
    Calculate rescaled Rij (i.e. Rij*) from dij (aka Rij) and its homopolymer fit, as in Rij* = dij / (A * |i-j|^nu).
    dij     : array of dij values
    aijdiff : array of absolute residue separations, |i-j|  (same shape as dij)
    nu      : exponent (float) from fit
    A       : prefactor (float) from fit (unit matching dij)
    fcn     : function that defines fit, taking arguments (ijseparation, exponentnu, prefactorA); default is A*(s^nu)
    output -> array of rescaled Rij* values (unitless), same shape and format as dij
    """
    # Note: optional fitted prefactor (A) in nanometer
    r_rsc = np.zeros(dij.shape)
    r_rsc[aijdiff>0] = dij[aijdiff>0] / fcn(aijdiff[aijdiff>0], nu, A)
    return r_rsc



##  PLOT TOOLS

# > establish normalization of arrays, and related labels
def check_norm(arr1, arr2, norm, var, dijunit='nm'):
    """
    Normalize given pair of arrays, if desired, and assign a unit based on variable setting.
    arr1 : first array, as xij or dij or Rij* (rescaled Rij)
    arr2 : second array, same type/unit as the first
    norm : boolean setting for normalizing arrays to their maximum values (separately)
    var  : string variable choice, containing one of 'xij' or 'dij' or 'Rij*'
    output -> tuple of (raw or normalized) arr1, arr2, normalization label (for filenames in saving), axis unit string
    """
    if norm:
        arr1 = arr1/(np.abs(arr1)).max()
        arr2 = arr2/(np.abs(arr2)).max() if arr2.any() else arr2
        nlabel = "norm_"    # file label indicating normalization
        ax_unit = "   [norm.]"      # unit of quantitiy plotted
    else:
        nlabel = ""
        ax_unit = "   [unitless]" if (('*' in var) or ('x' in var)) else f'   [{dijunit}]'
    return (arr1, arr2, nlabel, ax_unit)

# > establish variable / axis labels
def var_label(var):
    """
    Make labels for axes and filenames based on variable choice.
    var  : string variable choice, containing one of 'xij' or 'dij' or 'Rij*', optionally prefixed with 'D' or 'diff'
    output -> tuple of axis label string (LaTeX formatted), filename label string (safe, '*' replaced by 'rsc')
    """
    if '*' in var:
        vlbl = r'$d_{ij}^{\ast}$'
#        vlbl = r'$R_{ij}^{\ast}$'
    elif 'x' in var:
        vlbl = r'$x_{ij}$'
    else:
        vlbl = r'$R_{ij}$'
    vpre = r'$\Delta$' if (('D' in var) or ('diff' in var)) else ''     # label prefix for difference
    flab = var.replace('*', 'rsc') if ('*' in var) else var      #  file label from variable choice
    return (vpre + vlbl), flab

# > flatten and correlate given pair of xij/dij arrays
def flat_corr(arr1, arr2, ijdiff, minsep=0):
    """
    Flatten and compare given pair of arrays in the region of residue separation surpassing the minimum.
    arr1   : first array, as xij or dij or Rij* (rescaled Rij); could be normalized or not
    arr2   : second array, same type/unit as the first
    ijdiff : array of (raw/direct) differences in residue indices, (i-j), same shape as other arrays
    minsep : minimum separation of residues (int) to include in the comparison
    output -> tuple of flat arr1, flat arr2, Pearson's r, RSMD  (latter 3 are 'None' if arr2 is zero)
    """
    septest = (ijdiff >= minsep).flatten()
    # transpose if array is in 'upper' form
    if np.isclose(arr1[-1,0], 0.):
        arr1 = arr1.transpose()
    arr1_flat = arr1.flatten()
    arr1_flat = arr1_flat[septest]
    if arr2.any():
        # transpose if array is in 'upper' form
        if np.isclose(arr2[-1,0], 0.):
            arr2 = arr2.transpose()
        arr2_flat = arr2.flatten()
        arr2_flat = arr2_flat[septest]
        rval = pearsonr(arr1_flat, arr2_flat).statistic
        rmsd = np.sqrt(np.square(arr1_flat-arr2_flat).mean())
        return arr1_flat, arr2_flat, rval, rmsd
    else:
        return arr1_flat, None, None, None

# > handle colorbar scaling and ticks from combined array
def cbar_props(comb, var, norm=False, sym=False, lim=None, tick_fac=10, tick_num=5):
    """
    Get colorbar properties - range, center, and ticks - based on variable and other settings. Also modifies null values in combined array to match the designnated midpoint (e.g. xij should be centered on 1).
    comb     : combined array (both compared arrays arranged together)
    var      : string variable choice, containing one of 'xij' or 'dij' or 'Rij*', optionally prefixed with 'D' or 'diff'
    norm     : boolean setting for normalized arrays
    sym      : boolean setting for symmetric colorbars
    lim      : custom setting for colorbar limits (min,max); 'None' otherwise
    tick_fac : factor (float) for rounding colorbar ticks; if power of 10, rounds to that decimal place
    tick_num : number (int) of ticks to place above and below the midpoint; total ticks = 2*tick_num (if midpoint is bracketed)
    output  -> tuple of new comb array, colorbar normalizer object, list of colorbar ticks
    """
    if lim:
        mn, mx = lim        # custom min/max setting
        amn, amx = lim
    else:
        mn, mx = comb.min(), comb.max()     # raw minimum and maximum
        amn, amx = comb[~ np.isclose(comb,0.)].min(), comb[~ np.isclose(comb,0.)].max()     # nonzero minimum and maximum
    tfac = tick_fac     # factor for rounding colorbar ticks
    tnum = tick_num     # number of ticks above/below midpoint in colorbar
    if (('D' in var) or ('diff' in var)):
        mid = 0.
    elif (('*' in var) or ('x' in var)):
        mid = (0.5*(amn+amx)) if norm else 1.
        comb[np.isclose(comb,0.)] = mid     # set inert / omitted values to 1 (i.e. neutrality for xij and rescaled Rij)
    else:
        # keep default settings when plotting Rij (dij) ; always positive, no anticipated centering
        return comb, None, None
    if sym:
        vmx = (2*mid-amn) if ((amx-mid) < (mid-amn)) else amx          # min/max for symmetric colorbars
        vmn = max(0, 2*mid-amx) if ((amx-mid) > (mid-amn)) else amn    #
    else:
        vmx = amx if (amx > mid) else (2*mid-amn)          # min/max for asymmetric colorbars
        vmn = amn if (amn < mid) else max(0, 2*mid-amx)    #
    cnorm = TwoSlopeNorm(mid, vmin=vmn, vmax=vmx)
    rnd = lambda v: np.round(v*tfac)/tfac       # rounding to limited digits based on 'tick_fac'
    ct1 = list(np.linspace(rnd(vmn),mid,tnum)) if (vmn < mid) else [rnd(vmn)]
    ct2 = list(np.linspace(mid,rnd(vmx),tnum)) if (vmx > mid) else [rnd(vmx)]
    cticks = ct1 + ct2      # arranging colorbar ticks evenly
    return comb, cnorm, cticks

# > obtain mean over |i-j| for given array
def sep_means(arr, minsep=0):
    """
    Calculate mean values from given array along each diagonal, corresponding to residue separations, beyond the minimum.
    arr    : array as xij or dij or Rij* (rescaled Rij); could be normalized or not
    minsep : minimum separation of residues (int), i.e. diagonal index, to include in the list of means
    output -> tuple of separations |i-j| used, mean values at each separation  (both are 'None' if array is just zeros)
    """
    if arr.any():
        # start separation list at 'minsep' as in xij/dij
        seplist = np.arange(minsep, len(arr))
        # ensure positive diagonals here (transpose if array is in 'upper' form)
        if np.isclose(arr[0,-1], 0.):
            arr = arr.transpose()
        assert (not np.isclose(arr[0,-1], 0.))
        svlist = np.asarray( [[s,np.diagonal(arr, s).mean()] for s in seplist] )
        seps = svlist[:,0]
        vals = svlist[:,1]
#        print(f'minsep list:  {np.diagonal(arr,minsep)}')
#        print(f'minsep value:  {vals[0]:.5g}')
        return seps, vals
    else:
        return None, None

# > fitting procedure for trend across |i-j|
def sep_fit(seps, vals, prefac='fit', fitfunc=(lambda s,v,a: a*np.power(s,v))):
    """
    Perform curve fitting of separation values (e.g. dij vs |i-j|) to homopolymer, as in  A * |i-j|^nu.
    seps    : list of separations |i-j| of interest
    vals    : list of mean values (of xij or dij or Rij*) corresponding to separations
    prefac  : either string (any) indicating 2-parameter fit (both exponent and prefactor),
                or value (float) of specified prefactor A for 1-parameter fit (just exponent)
    fitfunc : function to fit, taking arguments (ijseparation, exponentnu, prefactorA); default is A*(s^nu)
    output -> tuple of fitted prefactor value, fitted exponent value, fit function (general 2-parameter form)
    """
    func2 = fitfunc
    # prefactor given as string -> use 2-parameter fitting (i.e. prefactor is found through fitting)
    if type(prefac) is str:
        # 2-parameter fit (exponent as well as pre-factor)
        fit = curve_fit(func2, seps, vals, full_output=True)
        v, a = fit[0]
    # prefactor given as number -> use 1-parameter fitting for just exponent, with prefactor fixed at specified value
    else:
        # simple 1-parameter fit (just exponent, presumed pre-factor)
        assert (type(prefac) in [float, np.float64, np.float32])
        a = prefac
        func1 = lambda s,v: func2(s,v,a)
        fit = curve_fit(func1, seps, vals, full_output=True)
        v = fit[0][0]
    return (a, v, func2)


##  MAIN PLOT FUNCTIONS

# > correlation / scatter plot
def corr_plot(arr1, arr2, var, ijdiff, minsep=0, valnorm=False, arrlabels=('arr1', 'arr2'), savedir=False, show=True, \
        fsize=(9,7), title=False, name=None, **plotopts):
    """
    Plot correlation / direct comparison of flattened arrays, ensuring matching indices, i.e. arr1[i,j] vs arr2[i,j].
    arr1      : first array, as xij or dij or Rij* (rescaled Rij); could be normalized or not
    arr2      : second array, same type/unit as the first
    var       : string variable choice, containing one of 'xij' or 'dij' or 'Rij*', optionally prefixed with 'D' or 'diff'
    ijdiff    : array of (raw/direct) differences in residue indices, (i-j), same shape as other arrays
    minsep    : minimum separation of residues (int) to include in the comparison
    valnorm   : boolean setting for normalizing arrays to their maximum values (separately)
    arrlabels : tuple of strings as labels for the two arrays
    savedir   : string specifying directory for saving the plot, if desired; otherwise 'False' or 'None'
    show      : boolean to show plots; if 'True', plot will show whether or not it is saved
    fsize     : tuple of numbers for the figure size / shape; passed to 'figsize' keyword in 'plt.subplots'
    title     : boolean setting for plot title; if 'True', title will be automatically formatted
    name      : string (or 'None') for the name / label of the sequence / result being plotted (used in title and filename)
    **        : additional keyword arguments passed to 'plot' function (color, linewidth, marker, ...)
    output -> tuple of Pearson's r, RMSD
    """
    # adjustment for normalized arrays
    arr1, arr2, nlabel, ax_unit = check_norm(arr1, arr2, valnorm, var)
    # handling labels according to variable selection
    vl, flab = var_label(var)
    # flatten and correlate arrays
    flat1, flat2, rval, rmsd  = flat_corr(arr1, arr2, ijdiff, minsep=minsep)
    # plot flattened arrays against each other, only if there are two to compare
    if rval:
        nlong = 'Normalized' if nlabel else ''
        print(f"\n\n{name}:\t{arrlabels[0]} vs. {arrlabels[1]}\t{nlong} {var}  | pearson-r = {rval:.4g} | rmsd = {rmsd:.4g}\n")
        fig, ax = plt.subplots(figsize=fsize)
        ax.plot(flat2, flat1, **plotopts)
        minval = min(flat1.min(), flat2.min())
        maxval = max(flat1.max(), flat2.max())
        ax.plot([minval,maxval], [minval,maxval], 'k--', lw=1.5, zorder=0)
        ax.set_xlabel(arrlabels[1] + "  " + vl + ax_unit)
        ax.set_ylabel(arrlabels[0] + "  " + vl + ax_unit)
        ax.minorticks_on()
        if title:
            if title == 'nameonly':
                ax.set_title(f'{name}', fontweight='bold')
            else:
                nm = f'{name}\n' if name else ''
                ax.set_title(nm + nlabel + vl + "  correlation" + f" :  r={rval:.4g} , RMSD={rmsd:.4g}")
        fig.tight_layout()
        if savedir:
            assert (type(savedir) is str)
            corr_file = f"{name}_{flab}_{nlabel}corr_minsep{minsep}.pdf"
            makedirs(savedir, exist_ok=True)
            fig.savefig(path.join(savedir, corr_file))
        if show:
            plt.show()
        plt.close()
    return rval, rmsd

# > map across ij space
def map_plot(arr1, arr2, var, minsep=0, valnorm=False, arrlabels=('arr1', 'arr2'), savedir=False, show=True, \
        fsize=(9,7), title=False, name=None, sym_cbar=False, lim_cbar=None, rnd_cbar=50, num_cbar=5, **imopts):
    """
    Plot map of combined arrays, separated across diagonal, for direct visual comparison.
    arr1      : first array, as xij or dij or Rij* (rescaled Rij); could be normalized or not
    arr2      : second array, same type/unit as the first
    var       : string variable choice, containing one of 'xij' or 'dij' or 'Rij*', optionally prefixed with 'D' or 'diff'
    minsep    : minimum separation of residues (int) to include in the comparison
    valnorm   : boolean setting for normalizing arrays to their maximum values (separately)
    arrlabels : tuple of strings as labels for the two arrays
    savedir   : string specifying directory for saving the plot, if desired; otherwise 'False' or 'None'
    show      : boolean to show plots; if 'True', plot will show whether or not it is saved
    fsize     : tuple of numbers for the figure size / shape; passed to 'figsize' keyword in 'plt.subplots'
    title     : boolean setting for plot title; if 'True', title will be automatically formatted
    name      : string (or 'None') for the name / label of the sequence / result being plotted (used in title and filename)
    sym_cbar  : boolean setting for symmetric colorbar values
    lim_cbar  : custom limits (min,max) for colorbar; 'None' to adjust to data limits
    rnd_cbar  : factor for rounding ticks in colorbar - passed to 'tick_fac' in 'cbar_props()'
    num_cbar  : number of ticks above/below midpoint of colorbar
    **        : additional keyword arguments passed to 'imshow' function (cmap, origin, ...)
    output -> None
    """
    # adjustment for normalized arrays
    arr1, arr2, nlabel, ax_unit = check_norm(arr1, arr2, valnorm, var)
    # handling labels according to variable selection
    vl, flab = var_label(var)
    # transpose if array 1 is in 'upper' form
    if np.isclose(arr1[-1,0], 0.):
        arr1 = arr1.transpose()
    # transpose if array 2 is in 'lower' form
    if np.isclose(arr2[0,-1], 0.):
        arr2 = arr2.transpose()
    # arrange combined array, and get colorbar properties (and modified combined array, 'comb')
    comb = arr1 + arr2      # combined arrays
    cblab = vl + ax_unit    # colorbar label
    comb, cnorm, cticks = cbar_props(comb, var, norm=valnorm, sym=sym_cbar, lim=lim_cbar, tick_fac=rnd_cbar, tick_num=num_cbar)
    # make map
    fig, ax = plt.subplots(figsize=fsize)
    img = ax.imshow(comb, interpolation='none', norm=cnorm, **imopts)
#    fig.colorbar(img, ax=ax, label=cblab, ticks=cticks)
    fig.colorbar(img, label=cblab, ticks=cticks, fraction=0.046, pad=0.04)
    ax.set_xlabel(r"residue  $j$")
    ax.set_ylabel(r"residue  $i$")
    ax.minorticks_on()
    if (np.abs(arr2)>0).any():
        org = imopts['origin'] if ('origin' in imopts) else 'upper'
        up_lbl = f" top={arrlabels[1]} , bottom={arrlabels[0]}"
        down_lbl = f" top={arrlabels[0]} , bottom={arrlabels[1]}"
        tb_label = up_lbl if (org=='upper') else down_lbl
    else:
        tb_label = f" {arrlabels[0]}"
    if title:
        if title == 'nameonly':
            ax.set_title(f'{name}', fontweight='bold')
        else:
            nm = f'{name}\n' if name else ''
            ax.set_title(nm + tb_label)
    fig.tight_layout()
    if savedir:
        assert (type(savedir) is str)
        dij_file = f"{name}_{flab}_{nlabel}map_minsep{minsep}.pdf"
        makedirs(savedir, exist_ok=True)
        fig.savefig(path.join(savedir, dij_file))
    if show:
        plt.show()
    plt.close()
    return

# > trend across separation |i-j|  (averaged at each separation)
def sep_plot(arr1, arr2, var, minsep=0, valnorm=False, arrlabels=('arr1', 'arr2'), savedir=False, show=True, \
        fit_prefac=None, fit_outdir=None, fsize=(9,7), title=False, name=None, opts1={}, opts2={}, fitopts1={}, fitopts2={}):
    """
    arr1       : first array, as xij or dij or Rij* (rescaled Rij); could be normalized or not
    arr2       : second array, same type/unit as the first
    var        : string variable choice, containing one of 'xij' or 'dij' or 'Rij*', optionally prefixed with 'D' or 'diff'
    minsep     : minimum separation of residues (int) to include in the comparison
    valnorm    : boolean setting for normalizing arrays to their maximum values (separately)
    arrlabels  : tuple of strings as labels for the two arrays
    savedir    : string specifying directory for saving the plot, if desired; otherwise 'False' or 'None'
    show       : boolean to show plots; if 'True', plot will show whether or not it is saved
    fit_prefac : string or number (or 'None') to enable fitting, and to specify prefactor
    fit_outdir : string specifying directory for saving fit parameters, if desired; otherwise 'False' or 'None'
    fsize      : tuple of numbers for the figure size / shape; passed to 'figsize' keyword in 'plt.subplots'
    title      : boolean setting for plot title; if 'True', title will be automatically formatted
    name       : string (or 'None') for the name / label of the sequence / result being plotted (used in title and filename)
    opts1      : dictionary of keywords for 'plot' of first array vs. |i-j|
    opts2      : dictionary of keywords for 'plot' of second array vs. |i-j|
    fitopts1   : dictionary of keywords for 'plot' of fitted curve from first array vs. |i-j|
    fitopts2   : dictionary of keywords for 'plot' of fitted curve from second array vs. |i-j|
    output -> tuple of fitted prefactor 1, fitted exponent 1, fitted prefactor 2, fitted exponent 2, curve fit function
    """
    # normalization and labels
    arr1, arr2, nlabel, ax_unit = check_norm(arr1, arr2, valnorm, var)
    vl, flab = var_label(var)
    # establish separation and value lists
    seps1, vals1 = sep_means(arr1, minsep=minsep)
    seps2, vals2 = sep_means(arr2, minsep=minsep)
    # plot across separations
    fig, ax = plt.subplots(figsize=fsize)
    ax.plot(seps1, vals1, label=arrlabels[0], **opts1)
    if vals2 is not None:
        ax.plot(seps2, vals2, label=arrlabels[1], **opts2)
    # get fits, save if desired
    if fit_prefac:
        if type(fit_prefac) is str:
            fit_label = '2par'
            print(f"2-par fitting {var} as  a*|i-j|^v  :")
        else:
            fit_label = '1par'
            print(f"1-par fitting {var} as  {fit_prefac:.3g}*|i-j|^v  :")
        fita1, fitv1, fitf1 = sep_fit(seps1, vals1, prefac=fit_prefac)
        print(f"\t * {arrlabels[0]}: a={fita1:.3g} , v={fitv1:.3g}")
        ax.plot(seps1, fitf1(seps1, fitv1, fita1), label=f"{fit_label}: a={fita1:.2g}, v={fitv1:.2g}", zorder=0, **fitopts1)
        if vals2 is not None:
            fita2, fitv2, fitf2 = sep_fit(seps2, vals2, prefac=fit_prefac)
            print(f"\t * {arrlabels[1]}: a={fita2:.3g} , v={fitv2:.3g}")
            ax.plot(seps2, fitf2(seps2, fitv2, fita2), label=f"{fit_label}: a={fita2:.2g}, v={fitv2:.2g}", zorder=0, **fitopts2)
        else:
            fita2, fitv2, fitf2 = None, None, None
        if fit_outdir:
            # prepare dataframe
            fit_dct = {'NAME':[name], f'{var}_{fit_label}_a1':[fita1], f'{var}_{fit_label}_v1':[fitv1]}
            if fita2 is not None:
                fit_dct.update({f'{var}_{fit_label}_a2':[fita2], f'{var}_{fit_label}_v2':[fitv2]})
            fit_df = pd.DataFrame(fit_dct)
            # handle file
            fname = f'{flab}_{nlabel}sepfits_{fit_label}_minsep{minsep}.csv'
            hdr = (fname not in listdir(fit_outdir))
            fit_outfile = path.join(fit_outdir, fname)
            fit_df.to_csv(fit_outfile, header=hdr, index=False, mode='a')
    else:
        fita1, fitv1, fitf1 = None, None, None
        fita2, fitv2, fitf2 = None, None, None
    ax.set_xlabel(r"residue separation  $|i-j|$")
    ax.set_ylabel(vl + ax_unit)
    ax.legend(loc='best')
    ax.minorticks_on()
    if title:
        if title == 'nameonly':
            ax.set_title(f'{name}', fontweight='bold')
        else:
            nm = f'{name}\n' if name else ''
            ax.set_title(nm + "scaling trend")
    fig.tight_layout()
    if savedir:
        assert (type(savedir) is str)
        sep_file = f"{name}_{flab}_{nlabel}sep_minsep{minsep}.pdf"
        makedirs(savedir, exist_ok=True)
        fig.savefig(path.join(savedir, sep_file))
    if show:
        plt.show()
    plt.close()
    return (fita1, fitv1, fita2, fitv2, fitf1)

