##  Mike Phillips, 9/29/2023; revamped 8/11/2025
##  Calculating full xij array, converting to Rij, and checking Rg.
##  For given sequence and parameters (T, cs, w3, and w2 or wmn function),
##      determine full xij matrix within Full Ionization model (alP=alM=1; p=delta=0)
##      _or_ doi-xij matrix alongside degrees of ionization alP_ij, alM_ij, for given p & delta.
##  Then, construct Rg from xij and N,l,b.
##  > Can ultimately check agreement with Rg measurements rigorously, under various w2;wmn schemes.

"""
Note: this script uses a single sequence for all i,j pairs; does not allow for distinct dye modifications at each i,j.

Command Line Arguments:
    (1) sequence name
    (2) output directory for saving results (xij & Rij arrays, alP & alM arrays if using doi-model, Rg value)
"""

import Sequence as S, xModel_ij as xM, doiModel_ij as doiM

import pandas as pd
import numpy as np
from itertools import combinations
from time import perf_counter
import os
import sys

args = sys.argv


##  sequence name, output directory, key settings, sequence file and headings

seqname = '0'
if len(args) > 1:
    seqname = args.pop(1)

#OUTDIR = './test/dynamics/out/'
OUTDIR = None
if len(args) > 1:
    OUTDIR = args.pop(1)


CHARGE_REG = True      # enable charge regulation (doiModel_ij) with each xij?  use 'False' for full ionization (xModel_ij)


# for some sets (e.g. Wenwei..Mittal) : include partial charge on Histidine
#SX.aminos.update({"H":0.5})
#SDX.aminos.update({"H":0.5})


seq_file = "./example_sequences/EKVseqs.csv"
name_head = "NAME"
seq_head = "SEQUENCE"
w2_head = "w2_pred_01"      # set to 'None' to use default w2 value (e.g. if not available in file)

#OUTDIR = "./test/dynamics/mittal_EKV_w2pred01/"


N_head, C_head, IDP_head = None, None, None     # headings for N/C terminal adjustments: primarily for Lindorff-Larsen IDRome set
#N_head, C_head, IDP_head = 'is_nterm', 'is_cterm', 'is_idp'


print(f"\nObtaining xij & Rij arrays, and Rg:\n * sequence '{seqname:}' from file  '{seq_file:}'")
print(f" * w2 heading  '{w2_head:}'")
print(f" * N/C terminal headings  '{N_head:}' , '{C_head:}' , '{IDP_head:}'")
if CHARGE_REG:
    print("+ under self-consistent charge-regulated model  'doiModel_ij'\n")
else:
    print("+ under full-ionization model  'xModel_ij'\n")


##  prepare output sub-directories and filenames (if saving automatically)

if OUTDIR:
    os.makedirs(os.path.join(OUTDIR, 'Rij'), exist_ok=True)
    os.makedirs(os.path.join(OUTDIR, 'xij'), exist_ok=True)

    RG_FILE = lambda T,cs: os.path.join(OUTDIR, "Rij/{s}_Rg_" + f"T{T}_cs{cs}.npy")
    RIJ_FILE = lambda T,cs: os.path.join(OUTDIR, "Rij/{s}_Rij_" + f"T{T}_cs{cs}.npy")

    X_FILE = lambda T,cs: os.path.join(OUTDIR, "xij/{s}_xij_" + f"T{T}_cs{cs}.npy")
    ALP_FILE = lambda T,cs: os.path.join(OUTDIR, "xij/{s}_alp_" + f"T{T}_cs{cs}.npy")
    ALM_FILE = lambda T,cs: os.path.join(OUTDIR, "xij/{s}_alm_" + f"T{T}_cs{cs}.npy")


##  basic parameter settings

#Tijdir = None   # directory with Tij matrices (optional, for speed-up)
Tijdir = "../jon_TijOmegaij/"

# basic parameters
l = 8               # Kuhn length [A]
b = 3.8             # bond length [A]
csmm = 150          # salt concenctration [mM]
#cs = 0
T = 300             # temperature [K]
pHval = None        # pH value (set to 'None' to use basic full ionization)
pKex = []           # list of exclusions for pKa (often: 'H','C','Y')

cpmm = 1            # protein monomer concentration [mM]  :  c.ion model only
pval, dval = 0.5, 1.3       # dipole size and dielectric mismatch  :  c.ion model only

w2def = 0.5         # default two-body interaction
w3val = 0.1         # default three-body interaction
ctxval = True       # setting for context; if disabled revert SCDM,SHDM -> SCD,SHD


# useful simple functions
lB_20c = 1e10 * 9e9 * np.square(1.6e-19) / (80 * 1.38e-23 * 293)    # Bjerrum length [Angstrom] at 20C (with eps_water=80)

lB_tilde = lambda T: ( (lB_20c) * (293)/(T) ) / b       # dim.-less Bjerrum length [from Temp. in K, 'b' and 'lB_20c' in A]
cs_tilde = lambda cs: 6.022e-7 * np.power(b,3) * cs         # dim.-less ionic (~salt) concentration [from cs in milli-Molar, b in A]
kap_tilde = lambda T,cs: np.sqrt( 4 * xM.PI * lB_tilde(T) * 2 * cs_tilde(cs) )  # dim.-less Debye screening [from Temp. and cs]

Rij = lambda x, pair: np.sqrt(x*l*b*np.abs(pair[1]-pair[0])) / 10      # to get Rij [nm] from xij (and ij pair)


# model parameters for x-only (Full Ionization)
pdict_FI = {'l':1, 'lB':lB_tilde(T), 'kappa':kap_tilde(T,csmm), 'w2':w2def, 'w3':w3val, \
        'pH':pHval, 'pKex':pKex, 'context':ctxval}

# arguments for x-only (Full Ionization) minimizer
minargs_FI = {'xinit':0.2, 'xinit2':2.5, 'x_bound':(1e-3,35), 'thr':1e-6}


# model parameters for doi-x (Degree of Ionization)
pdict_DOI = {'l':1, 'lB':lB_tilde(T), 'cs':cs_tilde(csmm), 'cp':cs_tilde(cpmm), 'w2':w2def, 'w3':w3val, \
        'p':pval, 'delta':dval, 'pH':pHval, 'pKex':pKex, 'context':ctxval}

# arguments for doi-x (Degree of Ionization) minimizer
minargs_DOI = {'method':'NM-TNC', 'alBounds':(1e-6,1.0), 'xBounds':(1e-3,35), 'ref':None, \
        'init':(0.70,0.65,0.3), 'init_2':(0.10,0.15,1.10), 'thr':1e-6}


##  get w2 and N/C terminal settings
def get_w2_NC(name=seqname, file=seq_file, namehead=name_head, w2head=w2_head, Nhead=N_head, Chead=C_head, IDPhead=IDP_head):
    if w2_head:
        w2_df = pd.read_csv(file)
        match = (w2_df[namehead] == name)
        w2 = w2_df[w2head][match].iloc[0]
        if Nhead:
            Nterm = eval(w2_df[Nhead][match].iloc[0].capitalize())
        else:
            Nterm = False
        if Chead:
            Cterm = eval(w2_df[Chead][match].iloc[0].capitalize())
        else:
            Cterm = False
        if IDPhead:
            if eval(w2_df[IDPhead][match].iloc[0].capitalize()):
                Nterm, Cterm = True, True
    else:
        w2 = w2def
        Nterm, Cterm = False, False
    return w2, Nterm, Cterm


##  establish sequence object and adjust as necessary for terminals, and track pH and pKexclude
def get_seq(name=seqname, file=seq_file, namehead=name_head, seqhead=seq_head, \
        pH=pHval, pKexclude=pKex, Nterm=False, Cterm=False):
    seqob = S.Sequence(name=name, file=file, headName=namehead, headSeq=seqhead, info=False)

    if pH:
        # pH adjustments are carried out upon instantiation of Model
        if (not Nterm) and ('Nterm' not in pKexclude):
            pKexclude.append('Nterm')
        elif Nterm:
            print(" * Including N-terminal charge...")
            assert ('Nterm' not in pKexclude)
        if (not Cterm) and ('Cterm' not in pKexclude):
            pKexclude.append('Cterm')
        elif Cterm:
            print(" * Including C-terminal charge...")
            assert ('Cterm' not in pKexclude)
    else:
        # non-pH adjustments need to be performed immediately upon Sequence
        seqob.charges = list(seqob.charges)
        if Nterm:
            print(" * Adjusting N-terminal charge...")
            seqob.charges[0] += 1
        if Cterm:
            print(" * Adjusting C-terminal charge...")
            seqob.charges[-1] += -1
        seqob.charges = tuple(seqob.charges)
        seqob.characterize()
        seqob.info()
    return seqob, pH, pKexclude


##  functions to carry out calculations

# construct xij for given sequence, file, etc. : Full Ionization
def xij_mat(seq, pars=pdict_FI, minargs=minargs_FI, msg=False):
    t1 = perf_counter()
    model = xM.xModel_ij(seq, Tijdir=Tijdir)
    model.setPars(pars)
    model.parInfo()
    x_arr = np.zeros((seq.N,seq.N))
    for pair in combinations(range(1,seq.N+1),2):
        x_arr[pair[1]-1,pair[0]-1], f = model.minFij(pair[1]-1, pair[0]-1, messages=msg, **minargs)
    t2 = perf_counter()
    print(f"\nTIME to get xij matrix:\t{(t2-t1):.4g}")
    return x_arr

# construct xij for given sequence, file, etc. : Charge Regulated / counterion condensation
def doixij_mat(seq, pars=pdict_DOI, minargs=minargs_DOI, msg=False):
    t1 = perf_counter()
    model = doiM.doiModel_ij(seq, Tijdir=Tijdir)
    model.setPars(pars)
    model.parInfo()
    x_arr = np.zeros((seq.N,seq.N))
    alp_arr = x_arr.copy()
    alm_arr = x_arr.copy()
    for pair in combinations(range(1,seq.N+1),2):
        (p,m,x), f = model.minFij(pair[1]-1, pair[0]-1, messages=msg, **minargs)
        x_arr[pair[1]-1,pair[0]-1] = x
        alp_arr[pair[1]-1,pair[0]-1] = p
        alm_arr[pair[1]-1,pair[0]-1] = m
    t2 = perf_counter()
    print(f"\nTIME to get doi-xij matrix:\t{(t2-t1):.4g}")
    return x_arr, alp_arr, alm_arr

# quick calculation / extraction of xee/Ree from given xij matrix
def xee(xij):
    return max(xij[-1,0],xij[0,-1])
def Ree(xij):
    return Rij(xee(xij), 1, len(xij))

# obtaining Rij matrix from xij
def Rij_arr(xij):
    rng = range(len(xij))
    J,I = np.meshgrid(rng, rng)
    return Rij(xij, (J,I))

# calculate Rg from a given xij matrix
def Rg_from_xij(xij):
    N = len(xij)
    if np.isclose(xij[-1,0], 0.):
        xij = xij.transpose()
    tot = 0
    for i in range(1,N):
        for j in range(0,i):
            tot += (i-j) * xij[i,j]
    return ( np.sqrt(l*b*tot/100) / N )

# calculate Rg from a given dij matrix (from theory or simulation)
def Rg_from_dij(dij):
    N = len(dij)
    if np.isclose(dij[-1,0], 0.):
        dij = dij.transpose()
    tot = 0
    for i in range(1,N):
        for j in range(0,i):
            tot += np.square(dij[i,j])
    return ( np.sqrt(tot) / N )



##  EVALUATE using above functions

# first set up sequence and update parameters
w2, Nterm, Cterm = get_w2_NC(name=seqname, file=seq_file, namehead=name_head, w2head=w2_head, \
                        Nhead=N_head, Chead=C_head, IDPhead=IDP_head)
seqob, pH, pKexclude = get_seq(name=seqname, file=seq_file, namehead=name_head, seqhead=seq_head, \
                        pH=pHval, pKexclude=pKex, Nterm=Nterm, Cterm=Cterm)

newpars = {'w2':w2, 'pH':pH, 'pKex':pKexclude}
pdict_FI.update(newpars)
pdict_DOI.update(newpars)

if CHARGE_REG:
    xij, alpij, almij = doixij_mat(seqob, pars=pdict_DOI, minargs=minargs_DOI, msg=False)
else:
    xij = xij_mat(seqob, pars=pdict_FI, minargs=minargs_FI, msg=False)
Rij_res = Rij_arr(xij)
Rg_res = Rg_from_dij(Rij_res)

print(f"\n > Radius of gyration:\tRg = {Rg_res:2.5g}  [nm]\n")

##  SAVE if 'OUTDIR' provided
if OUTDIR:
    rgfile, rijfile, xfile, alpfile, almfile = [f(T,csmm) for f in (RG_FILE, RIJ_FILE, X_FILE, ALP_FILE, ALM_FILE)]
    np.save(xfile.format(s=seqname), xij)
    np.save(rijfile.format(s=seqname), Rij_res)
    np.save(rgfile.format(s=seqname), np.asarray([Rg_res]))
    if CHARGE_REG:
        np.save(alpfile.format(s=seqname), alpij)
        np.save(almfile.format(s=seqname), almij)

