##  Mike Phillips, 3/14/2022
##  Searching for a very interesting sequence.
##  Monte-Carlo type random search / sequence generation.
##  Inputs: numbers of +/-/0 charges; SCD asymmetry cut, or SCD range.
##  Output: random sequence of charges, and string of aminos (see 'unTranslate').
##  -> repeat & track SCD asymmetry (++) - (--)
##  * Update, 7/27/2022 *
##  -> search for sequences in range of SCD
##  * Update, 2/24/2025 *
##  -> now referring to 'props' with 'Sequence' formalism
##  command line arguments:
##      (1) boolean string for pausing upon each sequence (recommend using 'True' generally)

import props
import numpy as np
import sys

rand = np.random.random

PAUSE = True
args = sys.argv
if len(args) > 1:
    PAUSE = eval(args.pop(1))

# make random sequence with given counts (overall length is total)
def makeSeq(Np=5, Nm=5, N0=0):
    N = Np + Nm + N0        # total length of sequence
    seq = []        # empty placeholder
    # build sequence list
    for n in range(N):
        # choose random number for each element
        r = rand()
        # handling outcome probabilities -> _flat_ probabilities, (nearly) independent of counts
        probs = np.array([ int(num>0) for num in (Np, Nm, N0) ])
        probs = probs / probs.sum()
        cprobs = list(np.cumsum(probs))     # cumulative probs
        for res in range(len(cprobs)):
            # establish result as corresponding index in (Np, Nm, N0)
            if r < cprobs[res]:
                break
        # handle outcomes
        if res == 0:
            # add positive charge, reduce count remaining
            seq.append(1)
            Np -= 1
            continue
        elif res == 1:
            # add negative charge, reduce count
            seq.append(-1)
            Nm -= 1
            continue
        else:
            # add zero charge, reduce count
            seq.append(0)
            N0 -= 1
    return tuple(seq)

# make sequences with large asymmetry
def asymSeq(Np=5, Nm=5, N0=0, asym_init=5.1, pause=True, ultimateCut=int(5e6), updateAsym=True):
    asym_cut = asym_init    # initialize cutoff asymmetry (absolute value)
    count = 0   # track number of sequences tested
    print(f"\nSeaching for sequences with N={Np+Nm+N0} (Np={Np}, Nm={Nm}) and minimum asymmetry SCD(++)-SCD(--)={asym_cut:.3g} ... ")
    while count < ultimateCut:
        seq = makeSeq(Np, Nm, N0)
        asym = props.SCD(seq, "++") - props.SCD(seq, "--")
        scd = props.SCD(seq)
        count += 1
        if abs(asym) > asym_cut:
            print("\n\n   CURRENT SEQUENCE:\n%s\n%s" % (str(seq), unTranslate(seq)))
            print("   ASYMMETRY [(++)-(--)]  =  %3.5g" % (asym))
            print("   SCD =  %3.5g\n" % (scd))
            print("(sequences tested so far: %i)" % count)
            if pause:
                pause_input = input("Continue search?\t> ")
                pause_input = pause_input.lower()[0]
            else:
                pause_input = "y"
                print("Search continuing...")
            if pause_input != "y":
                return (seq, unTranslate(seq))
            else:
                if updateAsym:
                    asym_cut = abs(asym)
                continue
    if count == ultimateCut:
        print("\n\nSEARCH CONCLUDED  -  ultimate cutoff reached:\tcount = %i\n" % count)
    return (seq, unTranslate(seq))

# make sequences with SCD falling in given range
def scdSeq(Np=5, Nm=5, N0=0, SCD_rng=(-1,-2), pause=True, ultimateCut=int(5e6)):
    scd_min, scd_max = min(SCD_rng), max(SCD_rng)
    count = 0   # track number of sequences tested
    print(f"\nSeaching for sequences with N={Np+Nm+N0} (Np={Np}, Nm={Nm}) and SCD within [{SCD_rng[0]:.3g}, {SCD_rng[1]:.3g}] ... ")
    while count < ultimateCut:
        seq = makeSeq(Np, Nm, N0)
        scd = props.SCD(seq)
        asym = props.SCD(seq, "++") - props.SCD(seq, "--")
        count += 1
        if scd_min <= scd <= scd_max:
            print("\n\n   CURRENT SEQUENCE:\n%s\n%s" % (str(seq), unTranslate(seq)))
            print("   * SCD =  %3.5g *" % (scd))
            print("   { ASYMMETRY [(++)-(--)]  =  %3.5g }\n" % (asym))
            print("(sequences tested so far: %i)" % count)
            if pause:
                pause_input = input("Continue search?\t> ")
                pause_input = pause_input.lower()[0]
            else:
                pause_input = "y"
                print("Search continuing...")
            if pause_input != "y":
                print("Search ended.\n")
                return (seq, unTranslate(seq))
            else:
                continue
    if count == ultimateCut:
        print("\n\nSEARCH CONCLUDED  -  ultimate cutoff reached:\tcount = %i\n" % count)
    return (seq, unTranslate(seq))

# function to revert sequence of charges to string of letters
def unTranslate(seq, P="K", M="E", Z="A"):
    res = ""    # initialize with empty string
    for s in seq:
        if s == 1:
            res += P
        elif s == -1:
            res += M
        else:
            res += Z
    return res



# RUN

#asymSeq(Np=25, Nm=25, N0=0, asym_init=7, pause=PAUSE, ultimateCut=int(5e6), updateAsym=False)

scdSeq(Np=25, Nm=25, N0=0, SCD_rng=(-14,-16.5), pause=PAUSE, ultimateCut=int(1e6))

