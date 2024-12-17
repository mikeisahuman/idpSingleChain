##  Mike Phillips, 12/9/2024
##  Examination of Ree (or x) vs. salt
##    > under doi / ion condensation Model
##  * Motivated by large responses in a subset of LL28k sequences upon cs 150 -> 50


import Sequence as S, doiModel as M
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import myPlotOptions as mpo
import sys
import os


seqname = None
#seqname = 'P26583_163_209'
#seqname = 'Q9H320_1_206'
#seqname = 'P55081_1_206'

if len(sys.argv) > 1:
    seqname = sys.argv[1]

PSAVE = None     # directory for saving final plot

if len(sys.argv) > 2:
    PSAVE = sys.argv[2]

DSAVE = None    # directory for saving resulting arrays (x or Ree)

if len(sys.argv) > 2:
    DSAVE = sys.argv[3]

# FILE SETUP

# full set (from which specific seq. can be pulled) ; including information for terminals
seq_setname = "LL seqs 28k set"
seq_file = "../out files/LL 30k sim/LL28k_set.csv"
name_head, seq_head = 'seq_name', 'fasta'
Nhead, Chead, IDPhead = 'is_nterm', 'is_cterm', 'is_idp'

# for retrieving w2
w2_file = "../out files/LL 30k sim/with T=37C/CNN custom/preds28k 1pt-pred 2pt-min w3=0.2/LL_w302_preds.csv"
w2_name_head = 'names'
#w2_head = "pred_w2"
w2_head = "true_w2"


# PARAMETERS

b = 3.8     # bond length [A]

p = 0.5     # dim-less dipole length
delta = 1.3 # dielectric mismatch

T = 310     # temperature [K]
csmm = 150  # salt concentration [mM]
cpmm = 1    # protein residue/monomer concentration [mM]
w2_def = 0  # default value for w2
w3 = 0.2    # 3-body term

lB_tilde = lambda T: (1.8743996063760822) * (293)/(T) * (3.8/b)     # dim-less Bjerrum length [from Temp. in K, 'b' in A]
cs_tilde = lambda cs: 6.022e-7 * np.power(b,3) * cs                 # dim.-less ionic (~salt) concentration [from cs in milli-Molar, b in A]

# extract w2 from its file
if w2_file:
    w2_df = pd.read_csv(w2_file)
    w2_match = (w2_df[w2_name_head] == seqname)
    w2 = w2_df[w2_head][w2_match].iloc[0]
else:
    w2 = w2_def

# store parameters
pdict = {'l':1, 'lB':lB_tilde(T), 'p':p, 'delta':delta, 'cp':cs_tilde(cpmm), 'cs':cs_tilde(csmm), 'w2':w2, 'w3':w3, 'pH':None, 'pKex':[]}

# extract amino sequence, and amy terminals from seq. file
seq_df = pd.read_csv(seq_file)
s_match = (seq_df[name_head] == seqname)
s_aminos = seq_df[seq_head][s_match].iloc[0]
[Nincl, Cincl, IDP] = [seq_df[hd][s_match].iloc[0] for hd in (Nhead, Chead, IDPhead)]

if IDP:
    Nincl, Cincl = True, True

# enable adjustment of Sequence object with terminals
def term_fix(seq, Nterm, Cterm):
    qlist = list(seq.charges)
    if Nterm:
        print("Adjusting N-terminal...")
        qlist[0] += 1
    if Cterm:
        print("Adjusting C-terminal...")
        qlist[-1] += -1
    seq.charges = tuple(qlist)
    seq.characterize()
    seq.info()
    return seq


# SEQUENCE & MODEL
sq = S.Sequence(seqname, file=None, aminos=s_aminos)
sq = term_fix(sq, Nincl, Cincl)
md = M.doiModel(sq)
md.setPars(pdict, pH_seqinfo=False)
md.parInfo()
minargs = {'ref':(0.5,0.5,1.0), 'init':(0.70,0.65,0.3), 'init_2':(0.10,0.15,1.10), 'xBounds':(0.01,25)}

# salt results (taking input in mM)
def salt_curve(cs_list=np.linspace(50,250,120)):
    resd = md.multiMin(multiPar='cs', parVals=cs_tilde(cs_list), minArgs=minargs)
    resd.update({'csmm':cs_list})
    resd.update({'Ree':np.sqrt(md.seq.N*0.38*0.80*resd['pmx'][:,2])})
    return resd

# salt PLOT
def salt_plot(resd, asRee=False, SAVE=PSAVE):
    if asRee:
        var = 'Ree'
        ylbl = r"$R_{ee}$  [nm]"
    else:
        var = 'x'
        ylbl = r"$x$"
    ylist = resd[var] if (var == 'Ree') else resd['pmx'][:,2]
    fig,ax = plt.subplots(figsize=(9.3,7))
    ax.plot(resd['csmm'], ylist)
    ax.set_xlabel(r"$c_s$  [mM]")
    ax.set_ylabel(ylbl)
    ax.set_title(seqname + "\n")
    fig.tight_layout()
    if SAVE:
        fig.savefig(os.path.join(SAVE, f"{var}_salt_{seqname:}.pdf"))
    else:
        plt.show()
    plt.close()
    return ylist


# INSPECTION of Free Energy at specific salt
def FEplot(xlist=np.linspace(1e-2,5,100), alp=0.9, alm=0.9, cs=150, SAVE=PSAVE):
    md.setPars({'cs':cs_tilde(cs)}, pH_seqinfo=False)
    md.parInfo()
    Flist = [ md.Ftot((alp,alm,x), **md.pars) for x in xlist ]
    fig,ax = plt.subplots(figsize=(9.3,7))
    ax.plot(xlist, Flist)
#    ymin,ymax = ax.get_ylim()
    ymin = min(Flist)
    ax.set_ylim(ymin-0.1*abs(ymin),ymin+0.25*abs(ymin))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$F(x)$")
    ax.set_title(seqname + f"\nFree Energy at cs={cs:.3g} mM")
    fig.tight_layout()
    if SAVE:
        fig.savefig(os.path.join(SAVE, f"FE_cs{cs:.3g}_{seqname:}.pdf"))
    else:
        plt.show()
    plt.close()
    md.setPars(pdict, pH_seqinfo=False)
    return


rd = salt_curve(np.linspace(50,250,5))
xres = salt_plot(rd)
Rres = salt_plot(rd, asRee=True)

if DSAVE:
    np.save(os.path.join(DSAVE, f"x_salt_{seqname:}.npy"), xres)
    np.save(os.path.join(DSAVE, f"Ree_salt_{seqname:}.npy"), Rres)
    np.save(os.path.join(DSAVE, f"salt_list.npy"), rd['csmm'])

