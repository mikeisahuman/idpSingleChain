##  Mike Phillips, 7/25/2023
##  File for calculating 'w2' based on simulated 'x' (from Ree)
##    - i.e. for training ML model on w2 (instead of Ree directly)
##  * Re-vamped 5/21/2024 *
##      > Adapted for use with 'Sequence' and 'xModel'
##  ** COMMAND LINE arguments **
##      (1) value for w3


import Sequence as S, xModel as M
from time import perf_counter
import numpy as np
import pandas as pd
import csv

import sys

argv = sys.argv


# pH settings
#phval = 4          # value of pH
#phex = ['H','C','Y']   # exclusions in translating charge from pH
phval = None
phex = []
phex_str = ('-no' + ''.join(phex)) if (len(phex)>0) else ''

# key parameters
l2 = 3.8*8      # factor relating Ree^2 and 'x' (via N), in square Angstroms
b = 3.8         # bond length [Angstrom]
#b = 5.5
T = 310         # temperature [K]

lB = 1.8743996063760822 * (293/T) * (3.8/b)      # dimensionless Bjerrum length

#csmm = 50          # salt concentration in mM
csmm = 150        # estimate based on CALVADOS
cs = csmm * 6.022e-7 * (b**3)   # convert to dimensionless concentration
kap = np.sqrt(4*M.PI*lB*2*cs)   # get dimensionless Debye screening

w3 = 0.2    # standard value of w3

if len(argv) > 1:
    w3 = float(argv[1])
w3_str = str(w3).replace('.','')


# parameter dictionary with placeholder w2
pard = {'l':1, 'lB':lB, 'kappa':kap, 'w2':0, 'w3':w3, 'pH':phval, 'pKex':phex, 'alP':1, 'alM':1, 'p':0}


# FILE SETTINGS

# file to use pre-calculated two/three body terms
#OBfile = None
OBfile = "OBfmt_5-1500.npy"

# LL 28k set
seq_file = "../IDRome_sequences/IDRome28k.csv"
out_file = f"../out/IDRome_w2out_w3{w3_str}.csv"
name_head = "seq_name"
seq_head = "fasta"
x_head = "x"
w2_head = "w2"
scd_head = "SCD"

Nterm_head = "is_nterm"
Cterm_head = "is_cterm"
IDP_head = "is_idp"

adjust_NC = True   # apply adjustments to N/C terminals according to spreadsheet?


##########

print(f"AMINO CHARGES\n{S.amino_charges}\n")


# function for establishing sequence and handling terminals

def seq_ob(sname, Nterm=False, Cterm=False):
    print("_____   "*4)
    print(f"\nSELECTED SEQUENCE:\t'{sname:}'\n")
    sq = S.Sequence(sname, file=seq_file, headName=name_head, headSeq=seq_head, info=False)
    # must handle terminals before model; use exclusions with pH
    if phval:
        excl = phex.copy()
        if Nterm:
            print(" * Including N-terminal charge...")
        else:
            excl += ['Nterm']
        if Cterm:
            print(" * Including C-terminal charge...")
        else:
            excl += ['Cterm']
        excl = tuple(excl)
        pard.update({'pKex':excl})
    else:
        excl = None
        ql = list(sq.charges)
        if Nterm:
            print(" * Adjusting N-terminal charge...")
            ql[0] += 1
        if Cterm:
            print(" * Adjusting C-terminal charge...")
            ql[-1] += -1
        sq.charges = tuple(ql)
    md = M.xModel(sq, info=False, OBfile=OBfile)
    md.setPars(pard, pH_seqinfo=False)
    return md


# functions to calculate 'w2' using 'x' from simulation Ree; writing output to csv

def get_w2(md, x, xinit=0.5, xinit2=None, x_thr=1e-6, x_bound=(1e-3,35), xcheck=1e-4):
    scd = md.SCD()
    w2 = md.findW2(x=x, check=xcheck, minArgs={'xinit':xinit, 'xinit2':xinit2, 'x_bound':x_bound, 'thr':x_thr})
    return w2, scd


def all_w2(sfile=seq_file, ofile=out_file, xinit=0.5, xinit2=None, x_thr=1e-4, x_bound=(1e-3,35), xcheck=1e-4):
    print(f"\n > Using sequence, x file  '{sfile:}' ...")
    sf = pd.read_csv(sfile)
    line1 = list(sf.keys())
    new_line1 = line1 + [w2_head, scd_head]
    print(f"\n >> Sending output w2 to  '{ofile:}' ...\n")
    # using 'csv' package so that output is written as each sequence is calculated
    with open(ofile, 'w', newline="") as outf:
        writer = csv.writer(outf, dialect="excel")
        writer.writerow(new_line1)
        for i in range(len(sf[name_head])):
            ogsname = sf[name_head][i]
            sname = str(ogsname)
            xval = sf[x_head][i]
            row = [sf[k][i] for k in line1]
            ri = (sf[name_head] == ogsname)
            if adjust_NC:
                Nincl = sf[Nterm_head][ri].iloc[0]
                Cincl = sf[Cterm_head][ri].iloc[0]
                if sf[IDP_head][ri].iloc[0]:
                    Nincl, Cincl = True, True
            else:
                Nincl, Cincl = False, False
            ob = seq_ob(sname, Nincl, Cincl)
            w2, scd = get_w2(ob, xval, xinit=xinit, xinit2=xinit2, x_bound=x_bound, x_thr=x_thr, xcheck=xcheck)
            print(f"\n > '{sname:}'   x={xval:12},  w2={str(w2):12},  scd={scd:.6f}\n")
            new_row = row + [str(w2), str(scd)]
            writer.writerow(new_row)
    return


# find 'w2' for all sequences

t1 = perf_counter()

all_w2(sfile=seq_file, ofile=out_file, xinit=0.2, xinit2=2.0, x_thr=1e-6, x_bound=(1e-3,21), xcheck=1e-4)       # 2-point verification of x

t2 = perf_counter()

print("\n" + ("+++++     "*4) + f"\nTIME for full set:\t{t2-t1:.5g}")

