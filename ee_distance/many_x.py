##  Mike Phillips, 8/2/2023
##  File for evaluating 'x' using predicted 'w2' from Parrot / CNN ML architecture.
##  * Re-vamped 5/20/2024 *
##     > adapted for use with 'Sequence' and 'xModel'
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
#T = 310         # temperature [K]
T = 300
#T = 298

lB = lambda T,b: 1.8743996063760822 * (293/T) * (3.8/b)      # dimensionless Bjerrum length

csmm = 0
#csmm = 250          # salt concentration in mM
#csmm = 150        # estimate based on CALVADOS
cs = lambda csmm, b: csmm * 6.022e-7 * (b**3)   # convert to dimensionless concentration
kap = lambda T, csmm, b: np.sqrt(4*M.PI*lB(T,b)*2*cs(csmm,b))   # get dimensionless Debye screening

w3 = 0.2    # standard value of w3

if len(argv) > 1:
    w3 = float(argv[1])
w3_str = str(w3).replace('.','')


# parameter dictionary with placeholder w2
pard = {'l':1, 'lB':lB(T,b), 'kappa':kap(T,csmm,b), 'w2':0, 'w3':w3, 'pH':phval, 'pKex':phex, 'alP':1, 'alM':1, 'p':0}


# FILE SETTINGS

# file to use pre-calculated two/three body terms
#OBfile = None
OBfile = "OBfmt_5-1500.npy"

# IDRome 28k set
seq_file = "../IDRome_sequences/IDRome28k_w2preds_w302.csv"        # sequences and w2 values
ref_file = "../IDRome_sequences/IDRome_28k.csv"      # sequences with reference Ree, and NC terminal information (if applicable)
out_file = f"../out/IDRome_out_{w3_str}.csv"        # output file for calculated x, Ree using w2 from ML

#name_head = "names"
#seq_head = "seqs"
#w2_head = "true_w2"
#w2_head = "pred_w2"
name_head = "seq_name"
seq_head = "fasta"
#w2_head = "truew2_02"
w2_head = "predw2_02"
N_head = "N"
SCD_head = "SCD"
x_head = "x"
R_head = "Ree[nm]"

ref_name_head = "seq_name"
ref_seq_head = "fasta"
ref_w2_head = 'w2'
ref_x_head = "x"
ref_R_head = "Ree / nm"     # reference Ree heading from input file (from Lindorff-Larsen)
ref_Rg_head = "Rg / nm"

res_ref_R_head = "ref_" + R_head        # reference Ree heading used in results file
res_FE_head = "F/N"         # reference Free Energy (per monomer) for results file

Nterm_head = "is_nterm"
Cterm_head = "is_cterm"
IDP_head = "is_idp"

adjust_NC = True   # apply adjustments to N/C terminals according to spreadsheet?
incl_ref = True    # ensure reference file enabled by default
halfseq = False     # check half-segments ?  [Note: half sequence segments do not incorporate pH]

salt_PH_fromfile = False
parinfo = False      # print parameter info with each sequence?


##########

print(f"AMINO CHARGES\n{S.amino_charges}\n")


# function for establishing sequence and handling terminals

def seq_ob(sname, Nterm=False, Cterm=False):
    global phval
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
        if halfseq:
            i1 = round((halfseq-1)*sq.N/2) if (halfseq<3) else round(sq.N/4)
            i2 = round(halfseq*sq.N/2) if (halfseq<3) else round(sq.N*3/4)
            sq = S.Sequence(sname, file=None, aminos=sq.aminos[i1:i2], info=False)
            ql = ql[i1:i2]
        sq.charges = tuple(ql)
    md = M.xModel(sq, info=False, OBfile=OBfile)
    md.setPars(pard, pH_seqinfo=False)
    return md


# functions to calculate 'x' and 'Ree' using 'w2' from ML test predictions; writing output to csv

def get_x(md, w2, xinit=0.5, xinit2=None, x_bound=(1e-3,35), thr=1e-6, parinfo=False):
    scd = md.SCD()
    pard.update({"w2":w2})
    md.setPars(pard, pH_seqinfo=False)
    if parinfo:
        md.parInfo()
    xres,fres = md.minF(xinit=xinit, xinit2=xinit2, x_bound=x_bound, thr=thr)
    Ree = np.sqrt(md.seq.N*l2*xres)/10       # end-to-end distance, in nm
    return xres, Ree, scd, fres


def all_x(sfile=seq_file, rfile=ref_file, incl_ref=True, ofile=out_file, xinit=0.5, xinit2=None, x_bound=(1e-3,35), thr=1e-6, parinfo=False):
    global phval, csmm
    print(f"\n > Using sequence, w2 file  '{sfile:}' ...")
    sf = pd.read_csv(sfile)
    line1 = list(sf.keys())
    new_line1 = line1 + [SCD_head, x_head, R_head]
    if incl_ref:
        print(f"\n > With reference Ree, NC file  '{rfile:}' ...")
        rf = pd.read_csv(rfile)
        new_line1 = new_line1 + [res_ref_R_head]
    new_line1.append(res_FE_head)
    print(f"\n >> Sending output x, Ree to  '{ofile:}' ...\n")
    # using 'csv' package so that output is written as each sequence is calculated
    with open(ofile, 'w', newline="") as outf:
        writer = csv.writer(outf, dialect="excel")
        writer.writerow(new_line1)
        for i in range(len(sf[name_head])):
            ogsname = sf[name_head][i]
            sname = str(ogsname)
            w2val = sf[w2_head][i]
            row = [sf[k][i] for k in line1]
            if salt_PH_fromfile:
                phval = sf[ph_head][i]
                csmm = sf[salt_head][i]
                pard.update({'kappa':kap(T,csmm,b), 'pH':phval})
            if incl_ref:
                if "_PRED" in sname:
                    rname = sname[:sname.index("_PRED")]
                else:
                    rname = ogsname
                ri = (rf[ref_name_head] == rname)
                ref_ree = rf[ref_R_head][ri].iloc[0]
                if adjust_NC:
                    Nincl = rf[Nterm_head][ri].iloc[0]
                    Cincl = rf[Cterm_head][ri].iloc[0]
                    if rf[IDP_head][ri].iloc[0]:
                        Nincl, Cincl = True, True
                else:
                    Nincl, Cincl = False, False
            elif (not incl_ref) and (adjust_NC):
                Nincl, Cincl = True, True
            else:
                Nincl, Cincl = False, False
            ob = seq_ob(sname, Nincl, Cincl)
            x, ree, scd, f = get_x(ob, w2val, xinit=xinit, xinit2=xinit2, x_bound=x_bound, thr=thr, parinfo=parinfo)
            print(f"\n > '{sname:}'   w2={str(w2val):12},  x={x:12}")
            new_row = row + [str(scd), str(x), str(ree)]
            if incl_ref:
                new_row.append(str(ref_ree))
            if halfseq:
                new_row[line1.index(seq_head)] = ob.seq.aminos
            new_row.append(str(f))
            writer.writerow(new_row)
    return


# find 'x' for all sequences

t1 = perf_counter()

#all_x(incl_ref=incl_ref)

# two-point mode : minimizing F(x) from two initial points
all_x(sfile=seq_file, rfile=ref_file, incl_ref=incl_ref, ofile=out_file, xinit=0.2, xinit2=2.0, x_bound=(1e-3,24), thr=1e-6, parinfo=parinfo)

t2 = perf_counter()

print("\n" + ("+++++     "*4) + f"\nTIME for full set:\t{t2-t1:.5g}")

