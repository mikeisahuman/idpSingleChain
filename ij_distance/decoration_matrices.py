##  Mike Phillips, 8/26/2025
##  Simple script to calculate decoration matrices using 'xModel_ij':
##  > SHDM: requires 'omega_mn' function for weights, otherwise calculates with uniform weight (i.e. w2_ij=1)
##  > SCDM: requires degrees of ionization alpha +/- ; default is assuming full ionization (alpha=1)
##  > SCDDM, SDDM: requires d.o.i. alpha +/- < 1 ; will return zeros if fully ionizaed

"""
Note: this tool uses a single sequence for all i,j pairs; does not allow for distinct dye modifications at each i,j.

Runs as a script _only_ if output directory ('OUTDIR') is provided, otherwise can be used as a module.

Command Line Arguments:
    (1) sequence name
    (2) output directory for saving results, if desired  (builds and saves _all_ decoration arrays)
"""

import Sequence as S, xModel_ij as M
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import myPlotOptions as mpo
import hydropathy_lambdas as HL
from time import perf_counter
from os import path
import sys


args = sys.argv

sname = 'sv1'
if len(args) > 1:
    sname = args.pop(1)

OUTDIR = None
if len(args) > 1:
    OUTDIR = args.pop(1)


PLOTS = True    # make and save plots?  will save to 'OUTDIR' if provided


##  SEQUENCE FILE AND HEADINGS
seq_file = "./example_sequences/IDPseqs.csv"
headName = "NAME"
headSeq = "SEQUENCE"


##  FILE FORMATS FOR SAVING ALL DECORATION ARRAYS
if OUTDIR:
    shdm_out = path.join(OUTDIR, f"{sname:}_SHDMarray.npy")
    scdm_out = path.join(OUTDIR, f"{sname:}_SCDMarray.npy")
    scddm_out = path.join(OUTDIR, f"{sname:}_SCDDMarray.npy")
    sddm_out = path.join(OUTDIR, f"{sname:}_SDDMarray.npy")
    scddmi_out = path.join(OUTDIR, f"{sname:}_SCDDM_ipolars_array.npy")
    sddmi_out = path.join(OUTDIR, f"{sname:}_SDDM_ipolars_array.npy")



##  KEY PARAMETERS
wmn=(lambda m,n: 1)     # for SHDM: residue-specific function for two-body interaction weights (see 'hydropathy_lambdas.py')

#wmn_flag = 'default'    # to use uniform scaling of above function
wmn_flag = 'mittal'     # to use mittal hydropathy form: wmn = (lam_m + lam_n)/2

#alP, alM = 1, 1     # for SCDM/SCDDM/SDDM: degrees of ionization on positive and negative residues
#alP, alM = 0.6, 0.8     # ionization must be less than 1 in order for SCDDM/SDDM to be non-trivial
alP, alM = 0.6, 0.6

x = 1       # they also depend on 'x' (swelling factor), but this should generally be kept at 1

int_polars = {"S":1, "Y":1, "Q":1, "N":1, "T":1}    # for SCDDM/SDDM: intrinsic polar residues to be included
#int_polars = {}     # default: no intrinsic polars

pHval = None        # could optionally set pH and pK exclusions for sequence charge (affects SCDM/SCDDM/SDDM)
pKexclude = []      #


##  ESTABLISH SEQUENCE & MODEL

def make_model(sname, seq_file, headName, headSeq, pH=pHval, pKex=pKexclude, ipolars=int_polars, wmn_flag='default'):
    pars = {'l':1, 'lB':1, 'kappa':0, 'w2':1, 'w3':1, 'pH':pH, 'pKex':pKex, 'context':True}

    seqob = S.Sequence(name=sname, alias="", aminos=None, info=True, file=seq_file, headName=headName, headSeq=headSeq)
    md = M.xModel_ij(seqob, intrinsic_polars=ipolars)
    md.setPars(pars)

    if wmn_flag == 'default':
        wmn=(lambda m,n: 1)
    else:
        # custom two-body weights 'wmn' require a sequence; it can be re-set here
        a = seqob.aminos
        wmn = lambda m,n: (HL.LAMBDA_M[a[m]] + HL.LAMBDA_M[a[n]])/2   # using Mittal Hydropathy scale, with amino sequence 'a'
    print(f"\n * wmn choice :  {wmn_flag:}\n")
    return md, wmn


##  BUILD _ALL_ ARRAYS SIMULTANEOUSLY: SCDM & SHDM/SCDDM/SDDM  (also with intrinsic polar counterparts)

def build_all_arrays(md, alP, alM, wmn, x=1):
    print(f"\n * ionization settings :  alP = {alP:.4g}    alM = {alM:.4g}")
    print(f"\t(factor  x={x:.4g})\n")
    t1 = perf_counter()
    # all have the same shape
    scdm = np.zeros((md.seq.N,md.seq.N))
    shdm = scdm.copy()
    scddm = scdm.copy()
    sddm = scdm.copy()
    scddm_i = scdm.copy()
    sddm_i = scdm.copy()
    for i in range(md.seq.N):
        for j in range(i):
            # SCDM is done on its own (unique functional form)
            scdm[i,j] = md.SCDM(i,j, alP=alP, alM=alM, x=x)
            # SHDM, SCDDM, SDDM all result from the same method (same functional form)
            shdm[i,j], scddm[i,j], sddm[i,j], scddm_i[i,j], sddm_i[i,j] = md.Omega(i,j, alP,alM, wmn)
    t2 = perf_counter()
    print(f"\n * Time to build all decoration matrices:\t{(t2-t1):.5g}")
    return scdm, shdm, scddm, sddm, scddm_i, sddm_i


##  SIMPLE PLOTTER TOOL (for checking any decoration matrix)

def plot_deco(mat, FSIZE=(9,7), TITLE=None, CBAR_LABEL="matrix", DLINE=True, SAVE=None, **imargs):
    N = len(mat)
    fig, ax = plt.subplots(figsize=FSIZE)
    im = ax.imshow(mat, **imargs)
    if DLINE:
        ax.plot([0,N],[0,N], 'k-', lw=2, zorder=2)
    ax.set_xlim(0,N-1)
    if ('origin' in imargs) and (imargs['origin']=='upper'):
        ax.set_ylim(N-1,0)
    else:
        ax.set_ylim(0,N-1)
    if TITLE:
        ax.set_title(sname)
    ax.set_xlabel(r'residue $j$')
    ax.set_ylabel(r'residue $i$')
    fig.colorbar(im, ax=ax, label=CBAR_LABEL)
    fig.tight_layout()
    if SAVE:
        fig.savefig(SAVE)
    else:
        plt.show()
    plt.close()
    return


##  RUN & SAVE ALL _only_ if output directory ('OUTDIR') is provided

if OUTDIR:
    md, wmn = make_model(sname, seq_file, headName, headSeq, pH=pHval, pKex=pKexclude, ipolars=int_polars, wmn_flag=wmn_flag)
    scdm, shdm, scddm, sddm, scddm_i, sddm_i = build_all_arrays(md, alP, alM, wmn, x)
    np.save(shdm_out, shdm)
    np.save(scdm_out, scdm)
    np.save(scddm_out, scddm)
    np.save(sddm_out, sddm)
    np.save(scddmi_out, scddm_i)
    np.save(sddmi_out, sddm_i)
    if PLOTS:
        plot_deco(shdm, CBAR_LABEL='SHDM', SAVE=shdm_out.replace('npy','pdf'), \
            cmap='viridis', norm=None, interpolation='none', origin='upper')
        plot_deco(scdm, CBAR_LABEL='SCDM', SAVE=scdm_out.replace('npy','pdf'), \
            cmap='seismic', norm=TwoSlopeNorm(0), interpolation='none', origin='upper')
        plot_deco(scddm, CBAR_LABEL='SCDDM', SAVE=scddm_out.replace('npy','pdf'), \
            cmap='Blues', norm=None, interpolation='none', origin='upper')
        plot_deco(sddm, CBAR_LABEL='SDDM', SAVE=sddm_out.replace('npy','pdf'), \
            cmap='Purples', norm=None, interpolation='none', origin='upper')
        plot_deco(scddm_i, CBAR_LABEL='int. SCDDM', SAVE=scddmi_out.replace('npy','pdf'), \
            cmap='Greens', norm=None, interpolation='none', origin='upper')
        plot_deco(sddm_i, CBAR_LABEL='int. SDDM', SAVE=sddmi_out.replace('npy','pdf'), \
            cmap='Oranges', norm=None, interpolation='none', origin='upper')

