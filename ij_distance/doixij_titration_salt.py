##  Mike Phillips, 8/2/2025
##  Examination of Rij (or xij) vs. salt - for a small list of i,j pairs
##    > under doi / ion condensation Model
##  * Motivated by available FRET data of Rij across ionic strengths for specific i,j dye placements

"""
Note: generally, disctinct sequences are needed for each i,j pair (due to dyes)
    >> so, must use a sequence name format string, instead of simple direct sequence name

Command Line Arguments:
    (1) format of sequence name with i,j labels; e.g. 'seq_{i}-{j}'
    (2) directory for saving results (xij, Rij, and salt_list)
"""


import Sequence as S, doiModel_ij as M
import numpy as np
import pandas as pd
from time import perf_counter
import sys
import os


seqname_fmt = 'proTa{i}-{j}_X'      # format string for sequence names, including i,j pair labels
if len(sys.argv) > 1:
    seqname_fmt = sys.argv[1]

DSAVE = None                        # directory for saving resulting arrays (xij and/or Rij)
if len(sys.argv) > 2:
    DSAVE = sys.argv[2]

SINGLE_OUTFILE = True       # save results as single dictionary file?  otherwise, saves many separate numpy arrays


# SETUP

# specify set of ij pairs to calculate - must correspond to sequences in 'seq_file' with naming format 'seqname_fmt'
ij_pairs = ( (2,57), (58,112), (2,111) )        # conventional indexing: i=1..N

# full set (from which specific seq. can be pulled) ; including information for terminals
seq_setname = "proTa ij set"
seq_file = "./example_sequences/proTa_ij_seqs.csv"
name_head, seq_head = 'NAME', 'SEQUENCE'

Tij_dir = "../jon_TijOmegaij/"      # directory with pre-calculated Tij arrays (if available)
#Tij_dir = None

# for retrieving w2 (if available)
w2_file = None
w2_name_head = 'names'
#w2_head = "pred_w2"
w2_head = "true_w2"


print(f"\nObtaining doi-xij titration across salt:\n * sequences '{seqname_fmt:}' from file  '{seq_file:}'")
print(f"   - using i,j pairs  {ij_pairs:}")
print(f" * w2 from file  '{w2_file:}'  (heading '{w2_head:}')"


# PARAMETERS
salt_list = np.linspace(50,250,21)      # list of salt values [mM] for which to calculate xij/Rij

b = 3.8     # bond length [A]
l = 8.0     # Kuhn length [A]

T = 295     # temperature [K]
csmm = 150  # salt concentration [mM]
cpmm = 1    # protein monomer concentration [mM]
w3 = 0.1    # 3-body term
w2_def = 0.5    # default value for 2-body term (overridden if 'w2_file' provided)
pval, dval = 0.5, 1.3       # dipole size and dielectric mismatch

lB_20c = 1e10 * 9e9 * np.square(1.6e-19) / (80 * 1.38e-23 * 293)    # Bjerrum length [Angstrom] at 20C (with eps_water=80)

lB_tilde = lambda T: ( (lB_20c) * (293)/(T) ) / b       # dim.-less Bjerrum length [from Temp. in K, 'b' and 'lB_20c' in A]
cs_tilde = lambda cs: 6.022e-7 * np.power(b,3) * cs     # dim.-less ionic (~salt) concentration [from cs in milli-Molar, b in A]

Rij = lambda x, pair: np.sqrt(x*l*b*abs(pair[1]-pair[0])) / 10      # to get Rij [nm] from xij (and ij pair)

# extract w2 from its file
if w2_file:
    w2_df = pd.read_csv(w2_file)
    w2_match = (w2_df[w2_name_head] == seqname)
    w2 = w2_df[w2_head][w2_match].iloc[0]
else:
    w2 = w2_def

# global parameters
pdict = {'l':1, 'lB':lB_tilde(T), 'cs':cs_tilde(csmm), 'cp':cs_tilde(cpmm), 'w2':w2, 'w3':w3, 'p':pval, 'delta':dval, \
        'pH':None, 'pKex':[], 'context':True}

# arguments for minimizer
minargs = {'method':'NM-TNC', 'alBounds':(1e-6,1.0), 'xBounds':(1e-3,35), 'ref':None, \
        'init':(0.70,0.65,0.3), 'init_2':(0.10,0.15,1.10), 'thr':1e-6, 'messages':False}

# salt results (taking input in mM)
def salt_curve(ijpair, cs_list=salt_list, show_time=True):
    t1 = perf_counter()
    i,j = ijpair
    # SEQUENCE & MODEL
    seqname = seqname_fmt.format(i=i, j=j)
    sq = S.Sequence(seqname, file=seq_file)
    md = M.doiModel_ij(sq, Tijdir=Tij_dir)
    md.setPars(pdict, pH_seqinfo=False)
    md.parInfo()
    # multiple minimization for all salts
    # note we should keep i>j, and convert to python indices
    resd = md.multiMin(j-1, i-1, multiPar='cs', vals=cs_tilde(cs_list), minArgs=minargs)
    resd.update({'csmm':cs_list})
    resd.update({'Rijres':Rij(resd['xij'], ijpair)})
    t2 = perf_counter()
    if show_time:
        print(f"\n * Time for salt curve of '{seqname}'  ({len(cs_list):} salt points):\t{(t2-t1):1.5g} * \n")
    return seqname, resd


if DSAVE:
    os.makedirs(DSAVE, exist_ok=True)
    for pair in ij_pairs:
        seqname, rd = salt_curve(pair, salt_list)
        if SINGLE_OUTFILE:
            np.save(os.path.join(DSAVE, f"all_doixij_salt_{seqname:}.npy"), rd)
        else:
            np.save(os.path.join(DSAVE, f"alPM_salt_{seqname:}.npy"), np.array(list(zip(rd['alP'],rd['alM']))))
            np.save(os.path.join(DSAVE, f"xij_salt_{seqname:}.npy"), rd['xij'])
            np.save(os.path.join(DSAVE, f"Rij_salt_{seqname:}.npy"), rd['Rijres'])
            np.save(os.path.join(DSAVE, f"salt_list.npy"), rd['csmm'])

