##  Mike Phillips, 5/1/2024
##  Examination of Ree (or x) vs. pH
##  * Motivated by coil-globule transition seen in a subset of LL28k sequences upon pH7->4


import Sequence as S, xModel as M
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import myPlotOptions as mpo
import sys
import os


#seqname = 'P30260_769_824'
#seqname = 'Q08499_714_809'
seqname = 'P07196_396_543'

if len(sys.argv) > 1:
    seqname = sys.argv[1]

SAVE = None     # directory for saving final plot

if len(sys.argv) > 2:
    SAVE = sys.argv[2]


# FILE SETUP

## set of all 589 sequences  (names and 'x' results only)
#seq_file = "../out files/LL 30k sim/with T=37C/CNN custom/preds28k 1pt-pred 2pt-min w3=0.2/ph4 shift/no HCY/coil-glob_finalsel.npy"
#seq_df = pd.DataFrame(np.load(seq_file, allow_pickle=True).item())

# full set (from which specific seq. can be pulled) ; including information for terminals
seq_setname = "LL seqs 28k set"
seq_file = "../out files/LL 30k sim/LL28k_set.csv"
name_head, seq_head = 'seq_name', 'fasta'
Nhead, Chead, IDPhead = 'is_nterm', 'is_cterm', 'is_idp'

# for retrieving w2
w2_file = "../out files/LL 30k sim/with T=37C/CNN custom/preds28k 1pt-pred 2pt-min w3=0.2/LL_w302_preds.csv"
w2_name_head = 'names'
w2_head = "pred_w2"
#w2_head = "true_w2"


# PARAMETERS

pKex = ['H','C','Y']        # exclusions from pH-based charge assignment (amended later with excluded terminals, as appropriate)

b = 3.8     # bond length [A]

T = 310     # temperature [K]
csmm = 150  # salt concentration [mM]
w3 = 0.2    # 3-body term

lB_tilde = lambda T: (1.8743996063760822) * (293)/(T) * (3.8/b)     # dim-less Bjerrum length [from Temp. in K, 'b' in A]
cs_tilde = lambda cs: 6.022e-7 * np.power(b,3) * cs                 # dim.-less ionic (~salt) concentration [from cs in milli-Molar, b in A]
kap_tilde = lambda T,cs: np.sqrt( 4 * M.PI * lB_tilde(T) * 2 * cs_tilde(cs) )  # dim.-less Debye screening [from Temp. and cs]

# extract w2 from its file
if w2_file:
    w2_df = pd.read_csv(w2_file)
    w2_match = (w2_df[w2_name_head] == seqname)
    w2 = w2_df[w2_head][w2_match].iloc[0]
else:
    w2 = w2_def

# extract amino sequence, and amy terminals from seq. file
seq_df = pd.read_csv(seq_file)
s_match = (seq_df[name_head] == seqname)
s_aminos = seq_df[seq_head][s_match].iloc[0]
[Nincl, Cincl, IDP] = [seq_df[hd][s_match].iloc[0] for hd in (Nhead, Chead, IDPhead)]
#[Nincl, Cincl, IDP] = [eval(seq_df[hd][s_match].iloc[0].capitalize()) for hd in (Nhead, Chead, IDPhead)]
#Nincl, Cincl, IDP = seq_df[Nhead][s_match].iloc[0], seq_df[Chead][s_match].iloc[0], seq_df[IDPhead][s_match].iloc[0]
if (not Nincl) and (not IDP):
    pKex.append('Nterm')
else:
    print(f"\nINCLUDING  N-terminal...")
if (not Cincl) and (not IDP):
    pKex.append('Cterm')
else:
    print(f"\nINCLUDING  C-terminal...")


pdict = {'l':1, 'lB':lB_tilde(T), 'kappa':kap_tilde(T,csmm), 'w2':w2, 'w3':w3, 'pH':7, 'pKex':pKex}


# SEQUENCE & MODEL
sq = S.Sequence(seqname, file=None, aminos=s_aminos)
md = M.xModel(sq)
md.setPars(pdict, pH_seqinfo=False)
minargs = {'xinit':0.2, 'xinit2':2.0, 'x_bound':(0.01,25)}

# pH results
def pHcurve(pH_list=np.linspace(7,4,100)):
    return md.multiMin(multiPar='pH', vals=pH_list, minArgs=minargs)

# pH PLOT
def pHplot(resd, SAVE=SAVE):
    fig,ax = plt.subplots(figsize=(9.3,7))
    ax.plot(resd['pH'], resd['x'])
    ax.set_xlabel("pH")
    ax.set_ylabel(r"$x$")
    #ax.set_ylabel(r"expansion/compaction factor  $x$")
    ax.set_title(seqname + "\n")
    fig.tight_layout()
    if SAVE:
        fig.savefig(os.path.join(SAVE, f"x_pH_{seqname:}.pdf"))
    else:
        plt.show()
    plt.close()
    return


# INSPECTION of Free Energy at specific pH
def FEplot(xlist=np.linspace(1e-2,5,100), pH=7, SAVE=SAVE):
    md.setPars({'pH':pH}, pH_seqinfo=False)
    md.parInfo()
    Flist = [ md.F(x, **md.pars) for x in xlist ]
    fig,ax = plt.subplots(figsize=(9.3,7))
    ax.plot(xlist, Flist)
#    ymin,ymax = ax.get_ylim()
    ymin = min(Flist)
    ax.set_ylim(ymin-0.1*abs(ymin),ymin+0.25*abs(ymin))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$F(x)$")
    ax.set_title(seqname + f"\nFree Energy at pH={pH:.3g}")
    fig.tight_layout()
    if SAVE:
        fig.savefig(os.path.join(SAVE, f"FE_pH{pH:.3g}_{seqname:}.pdf"))
    else:
        plt.show()
    plt.close()
    md.setPars(pdict, pH_seqinfo=False)
    return

