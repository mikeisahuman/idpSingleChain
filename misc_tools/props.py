##  Mike Phillips, 3/9/2022
##  File providing functions related to characterizing polyampholytes / degree of ionization
##  - check SCD of arbitrary sequence
##  - count 'flips' (or fraction) in arbitrary sequence: (+) -> (-)  or  (-) -> (+)
##  - find average 'block' sizes, for (+) & (-) charges separately
##  - transform sequence: charge parity + reversal (inversion)
##      - quantify changes from transformation:
##          > compare original vs. new SCD (should be unchanged)
##          > compare flips (should be unchanged), and block sizes (may be chnaged)
##          > count number (fraction) of changes from original to transformed
##  * Update, 2/24/2025 *
##  -> now using 'Sequence' object (if referring to sequence by name, from CSV file)

import Sequence as S
import numpy as np

sfile = './example_sequences/SVseqs.csv'
name_h = 'NAME'
seq_h = 'SEQUENCE'

# handle sequence inputs - direct as list/tuple, or by name (e.g. 'sv1')
def refine(seq):
    # optionally reference sequence by name (from default spreadsheet defined in 'seqIon')
    if type(seq) == str:
        return (S.Sequence(name=seq, file=sfile, headName=name_h, headSeq=seq_h, info=False).charges)
    # assume 'seq' is now a 'list' or 'tuple'
    else:
        return seq


# sequence charge decoration metric
def SCD(seq, mode="all"):
    seq = refine(seq)
    N = len(seq)
    tot = 0
    for m in range(1,N):
        for n in range(m):
            qm = seq[m]
            qn = seq[n]
            if mode == "++":
                if qm < 0 or qn < 0:
                    continue
            elif mode == "--":
                if qm > 0 or qn > 0:
                    continue
            elif mode in ("+-", "-+"):
                if round(np.sign(qm)) == round(np.sign(qn)):
                    continue
            tot += qm * qn * ( (m-n)**(0.5) )
    return (tot/N)


# fraction of 'flips'
def Flips(seq):
    seq = refine(seq)       # ensure sequence
    Nmax = len(seq) - 1     # maximum number of flips (spaces between charges)
    flips = 0       # initialize counter
    for n in range(Nmax):
        # increment if sequential charges are unequal
        if seq[n] != seq[n+1]:
            flips += 1
    return (flips/Nmax)


# average block sizes, for each charge sign (and zero/neutral)
def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
# returns lists of block sizes for each charge type; ordering: (+), (-), (0)
def BlockLists(seq):
    seq = refine(seq)
    # initialize current sign
    cur = sign(seq[0])
    # intialize lists of block sizes, current block sizes
    blocksP = []
    blocksM = []
    blocks0 = []
    if cur > 0:
        bkP = 1
        bkM = 0
        bk0 = 0
    elif cur < 0:
        bkP = 0
        bkM = 1
        bk0 = 0
    else:
        bkP = 0
        bkM = 0
        bk0 = 1
    for n in range(1,len(seq)):
        nxt = sign(seq[n])
        if nxt == cur:
            if nxt > 0:
                bkP += 1
            elif nxt < 0:
                bkM += 1
            else:
                bk0 += 1
        else:
            if cur > 0:
                if bkP > 0:
                    blocksP.append(bkP)
                bkP = 0
                if nxt < 0:
                    bkM += 1
                else:
                    bk0 += 1
            elif cur < 0:
                if bkM > 0:
                    blocksM.append(bkM)
                bkM = 0
                if nxt > 0:
                    bkP += 1
                else:
                    bk0 += 1
            else:
                if bk0 > 0:
                    blocks0.append(bk0)
                bk0 = 0
                if nxt > 0:
                    bkP += 1
                elif nxt < 0:
                    bkM += 1
        cur = nxt
    # put last values into lists
    if len(blocksP)==0 or cur > 0:
        blocksP.append(bkP)
    if len(blocksM)==0 or cur < 0:
        blocksM.append(bkM)
    if len(blocks0)==0 or cur == 0:
        blocks0.append(bk0)
    return (blocksP, blocksM, blocks0)
# returns mean block sizes for all types; ordering: (+), (-), (0)
def Blocks(seq):
    (blocksP, blocksM, blocks0) = BlockLists(seq)
    # calculate mean (simple average)
    meanP = sum(blocksP)/len(blocksP)
    meanM = sum(blocksM)/len(blocksM)
    mean0 = sum(blocks0)/len(blocks0)
    return (meanP, meanM, mean0)


# transform a sequence: charge parity + inversion
def Transform(seq):
    seq = refine(seq)
    new = []
    for s in seq:
        if s > 0:
            new.append(-1)
        elif s < 0:
            new.append(1)
        else:
            new.append(0)
    new.reverse()
    return tuple(new)

# difference (fraction) between two similar (same size) sequences
def Diff(seq1, seq2):
    seq1, seq2 = refine(seq1), refine(seq2)
    if len(seq1) != len(seq2):
        print("\nERROR: given sequences are not equal length.")
        return
    else:
        N = len(seq1)
    diffcount = 0
    for n in range(N):
        if seq1[n] != seq2[n]:
            diffcount += 1
    return (diffcount/N)

