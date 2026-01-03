##  Mike Phillips, 6/18/2025
##  Single-chain model for inter-residue factor 'xij', cf. Rij^2 = |i-j|*l*b*xij
##   + counter-ions / degrees of ionization (effective charge) 'a+/-' at each ij self-consistent with 'xij'
##  Class / Object definition - encapsulating all functions
##  Each Model object holds a Sequence.
##      > with capability to calculate Free Energy, minimization for 'xij', etc.
##      > holds all necessary parameters, and enables (re-)setting them as desired
##
##  Require: sequence (name / aminos)
##           desired ij pair (NOT well-designed for full set of pairs, but it is possible to obtain full ij arrays)
##           key parameters > lB/l (i.e. Temperature)
##                          > salt/ionic concentration for electrostatic screening
##                          > protein concentration for screening and entropy related to counter-ions
##                          > two-body and three-body volume interactions w2, w3 (~ hydrophobicity related)
##                          > dipole size p/l, dielectric mismatch delta=eps_w/eps_l  (only applicable if deg. of ion. < 1)
##                          > optional custom pH and list of exclusions from pKa calcuations
##                          > boolean settings for dipole terms and ion condensation details (from F4)
##                          > dipole length Pint for intrinsic polars
##
##  'T' matrix optimized for speed by saving calculated values in array with object.


import numpy as np
from scipy.special import erfcx as SCIerfcx     # SCALED complimentary error function: erfcx(u)=exp(u^2)*erfc(u)
from scipy.optimize import minimize, root_scalar, root
from time import perf_counter       # for timing of minimization
import os       # for checking directory for Tij and Oij files


# GLOBAL Data Type
DTYPE = np.float64
PI = DTYPE(np.pi)       # appropriate value for 'pi'
erfcx = lambda u: SCIerfcx(u, dtype=DTYPE)  # incorporating data type in supplied 'erfcx(u)')


# default settings for intrinsic polars
Pint_def = 3.8*(0.5)*(0.5)      # default dipole moment for intrinsic polars: half charges separated by half bond length

#polars = {"B":1}       # fictitious polar (neutral charge) amino acid
#polars = {"S":1, "Y":1, "Q":1, "N":1, "T":1}      # yes: ser tyr gln asn ; maybe possible: cys gly thr , trp his
#polars = {}       # blank dictionary - to neglect intrinsically polar residues

def polar_check(polars):
    pol_msg = "_NOT_" if (len(polars) == 0) else "_INDEED_"
    pol_secondl = "" if (len(polars) == 0) else f"\tpolars = {polars:}\n"
    return f"\nFROM 'xModel_ij' :\n\tyou are {pol_msg:} including intrinsic polars!\n{pol_secondl:}"


# dictionary of _all_ default parameter settings; any/all can be overridden as desired with 'setPars'
#   > if including intrinsic polars: must specify the intrinsic dipole moment
#   > Pint = d*q  ['d' in units matching 'l', 'q' in units of elementary charge]
default_pars = {'l':3.8, 'lB':7.12, 'cs':0, 'cp':6e-7, 'w2':0, 'w3':0.1, 'p':1.9, 'delta':1.3, 'Pint':Pint_def, \
        'pH':None, 'pKex':(), 'dipoleD2factor':True, 'F4factor':True, 'F4screen':False, 'F4p':None, 'kill_kappa':False, \
        'wmn':(lambda m,n:1), 'wii':None, 'wio':None, 'woi':None, 'woo':None, 'context':True}


#####   Class Defnition : object for encapsulating, evaluating 'simple' inter-residue IDP model #####
class doiModel_ij:
    def __init__(self, seq, info=False, Tijdir=None, intrinsic_polars={}):
        self.polars = intrinsic_polars
        print(polar_check(self.polars))
        self.seq = seq      # sequence object is a necessary input
        self.seq.polars = self.translate_polar()    # include sequence as polars
        if info:
            self.seq.info()
        self.full_range = range(0,seq.N)   # reference range for summations
        # set placeholder array of T(i,j), to be filled as it gets called for each (i,j) pair
        self.mat_T = np.zeros((seq.N, seq.N))
        # option: load Tij from prior calculation (e.g. from GPU)
        if Tijdir and (f"{self.seq.N:}_TijArray.npy" in os.listdir(Tijdir)):
            Tpath = os.path.join(Tijdir, f"{self.seq.N:}_TijArray.npy")
            print(f"\n * Loading Tij array:  '{Tpath:}'\n")
            j,i = np.meshgrid(self.full_range, self.full_range)
            ijdiff = (i-j)      # prepare difference grid
            ijpos = (ijdiff > 0)    # valid value check (positive)
            Tmat = DTYPE( np.load(Tpath) )
            Tmat = Tmat.reshape((self.seq.N,self.seq.N))        # ensure expected shape
            Tmat[ijpos] = Tmat[ijpos]/ijdiff[ijpos]     # rescale with difference
            self.mat_T = DTYPE( Tmat.copy() )
        # load in baseline parameters
        self.allpars = {}   # blank initialization to load defaults
        self.setPars()
   #   #   #   #   #   #

    #   translate polars (only relevant if polars are included in theory, cf. 'polars' dictionary)
    def translate_polar(self):
        lst = [(self.polars[c] if (c in self.polars) else 0) for c in self.seq.aminos]
        return tuple(lst)

    #   set basic parameters / methods
    def setPars(self, pars={}, pH_seqinfo=True):
        pdict = self.allpars.copy() if self.allpars else default_pars
        pdict.update(pars)
        self.allpars = pdict        # store dictionary of ALL parameters
        self.pars = {k:pdict[k] for k in pdict if k not in ('pH','pKex')}       # all except pH pars (for Free Energy arguments)
        if pdict['pH']:
            self.seq.charges = self.seq.PHtranslate(pH=pdict['pH'], pKexclude=pdict['pKex'])
            self.seq.characterize()
            if pH_seqinfo:
                self.seq.info()
        return pdict

    #   print table of parameters
    def parInfo(self, exclude=('Pint',"wmn","wii","wio","woi","woo")):
        print("\nxModel parameters:")
        print("\n\tPAR\tVALUE")
        includedKeys = [k for k in self.allpars.keys() if k not in exclude]
        for par in includedKeys:
            try:
                print("\t{:5}\t{:3.5g}".format(par, self.allpars[par]))
            except:
                print("\t{:5}\t{:6}".format(par, str(self.allpars[par])))
        print("")
        return

    #   Debye screening  (dim.less, kappa*l)
    def kapl(self, alP, alM, cp, cs, l, lB):
        if alP < 0 or alM < 0:
            return np.nan
        else:
            return DTYPE( l * np.sqrt( 4*PI*lB* ((self.seq.fracp*alP+self.seq.fracm*alM)*cp + 2*cs ) ) )

    ## Free Energy : as 'beta F / N'
    #   combinatorial entropy - placement of counterions on polymer chain
    def F1(self, alP, alM):
        res = 0
        if 0.0 < alP < 1.0:
            res += self.seq.fracp * alP*np.log(alP, dtype=DTYPE)
            res += self.seq.fracp * (1-alP)*np.log(1-alP, dtype=DTYPE)
        if 0.0 < alM < 1.0:
            res += self.seq.fracm * alM*np.log(alM, dtype=DTYPE)
            res += self.seq.fracm * (1-alM)*np.log(1-alM, dtype=DTYPE)
        return DTYPE(res)

    #   translational entropy - wandering counterions (~plasma)
    def F2(self, alP, alM, cp, cs, l):
        l3 = np.power(l,3)
        cp, cs = cp*l3, cs*l3
        res = 0
        quantP = self.seq.fracp*alP*cp + cs
        quantM = self.seq.fracm*alM*cp + cs
        if quantP > 0.0:
            res += quantP * ( np.log(quantP, dtype=DTYPE) - 1 )
        else:
            return np.nan
        if quantM > 0.0:
            res += quantM * ( np.log(quantM, dtype=DTYPE) - 1 )
        else:
            return np.nan
        return DTYPE(res / cp)

    #   ion density fluctuations - from correlation length xi^-3 (xi~1/kappa)
    def F3(self, kl, cp, l):
        res = kl*kl*kl       # ionic fluctuations simply give kappa^3
        return DTYPE( (-1/(12*PI)) * res / (cp*l*l*l) )

    #   ion pair energy - reduction in energy from attraction of opposite charges
    def F4(self, alP, alM, l, lB, p, dlt, kl, extraFac=True, screen=False):
        if np.isclose(dlt,0.0) or np.isclose(p,0.0):
            return DTYPE(0)
        dlt_extra = 0.5 if extraFac else 0      # correction factor from effective dipole form 1/d to 1/p  (i.e. delta*(1+1/(2*delta)) = delta+0.5)
        res = - ( self.seq.fracp*(1-alP) + self.seq.fracm*(1-alM) )
        # explicit 'p' -> interpretation of 'delta' as just ratio of dielectrics: eps_bulk/eps_local
        res *= lB * (dlt + dlt_extra) / p
        if screen:
            res *= np.exp(-kl*p/l, dtype=DTYPE)
        return DTYPE(res)

    #   polymer free energy - chain entropy, intra-chain interactions (potential energy), specific to pair ij
    def F5_ij(self, alP, alM, x, i, j, l, lB, p, delta, kl, w2, w3, Pint, dipoleD2factor=True, \
            wmn=(lambda m,n:1), wii=None, wio=None, woi=None, woo=None, context=True):
        # Note: i,j used as Python indices, i.e. ranging 0..N-1
        if any([(x<0.), (alP<0.), (alP>1.), (alM<0.), (alM>1.)]):
            return 0
        if i<0 or j<0 or i>(self.seq.N-1) or j>(self.seq.N-1):
            print(f"\n* Warning: must keep i,j values as sequence indices!\t[given (i,j)=({i:},{j:})]\n")
            raise ValueError
        if j == i:
            return 0
        elif j > i:
            # functions take arguments with i>j, but symmetric by construction; set that way if reversed
            i,j = j,i
        w_dlt = delta if dipoleD2factor else 1      # only include delta^2 in dipole strengths if option is specified
        wcd, wdd = self.Wcddd(l,lB,kl,p,w_dlt)
        wcdi, wddi = self.Wcddd(l,lB,kl,Pint,w_dlt)
        om, ocd, odd, ocdi, oddi = self.Omega(i,j, alP,alM, wmn, wii,wio,woi,woo, context)
        t = self.T(i,j) if w3!=0 else 0
        q = self.Q(x,i,j, kl,alP,alM, context)
        res = 1.5 * (x - np.log(x, dtype=DTYPE))
        res += (2*lB/(PI*l)) * q
        res += (np.power(3/(2*PI*x), 1.5)) * (w2*om + wcd*ocd + wdd*odd + wcdi*ocdi + wddi*oddi)
        res += w3 * (np.power(3/(2*PI*x), 3)) * t / 2
        return DTYPE(res/self.seq.N)

    #   charge-dipole and dipole-dipole interaction weights (after directional averaging and casting as delta functions)
    def Wcddd(self, l, lB, kl, p, delta):
        wcd = - delta*delta * (PI/3) * np.square(lB*p/(l*l)) * (2 + kl) * np.exp(-2*kl, dtype=DTYPE)
        wdd = - delta*delta * (PI/9) * np.square(lB*p*p/(l*l*l)) * (4 + kl*(8 + kl*(4 + kl))) * np.exp(-2*kl, dtype=DTYPE)
        return wcd, wdd

    #   2-body TERM
    def Om_term(self, i, j, m, n,  wii,wio,woi,woo, context):
        if j > i:
            i,j = j,i
        if n >= m:
            return 0
        term = 0
        denfac = np.power(m-n, -2.5)
        if (i >= m >= (j+1)) and ((m-1) >= n >= j):
            term += np.square(m-n) * denfac * (wii if wii else 1)
        if (i >= m >= j) and ((j-1) >= n) and context:
            term += np.square(m-j) * denfac * (woi if woi else 1)
        if (m >= (i+1)) and ((j-1) >= n) and context:
            term += np.square(i-j) * denfac * (woo if woo else 1)
        if (m >= (i+1)) and (i >= n >= j) and context:
            term += np.square(i-n) * denfac * (wio if wio else 1)
        return term

    #   3-body TERM
    def T_term(self, i, j, l, m, n):
        if j > i:
            i,j = j,i
        if n >= m or m >= l:
            return 0
        term = 0
        denfac = np.power((l-m)*(m-n), -1.5)
        if (i >= l >= (j+3)) and ((l-1) >= m >= (j+2)) and ((m-1) >= n >= (j+1)):
            term += (l-n) * denfac
        if (i >= l >= (j+1)) and (j >= m >= 1) and ((m-1) >= n):
            term += np.square(l-j)/(l-m) * denfac
        if (i >= l >= (j+2)) and ((l-1) >= m >= (j+1)) and (j >= n):
            term += ( (l-m) + np.square(m-j)/(m-n) ) * denfac
        if (l >= (i+1)) and (j >= m >= 1) and ((m-1) >= n):
            term += np.square(i-j)/(l-m) * denfac
        if (l >= (i+1)) and ((l-1) >= m >= i) and (j >= n):
            term += np.square(i-j)/(m-n) * denfac
        if (l >= (i+1)) and ((i-1) >= m >= (j+1)) and (j >= n):
            term += (np.square(i-m)/(l-m) + np.square(m-j)/(m-n)) * denfac
        if (l >= (i+1)) and (i >= m >= (j+2)) and ((m-1) >= n >= (j+1)):
            term += (np.square(i-m)/(l-m) + (m-n)) * denfac
        if (l >= (i+2)) and ((l-1) >= m >= (i+1)) and (i >= n >= (j+1)):
            term += np.square(i-n)/(m-n) * denfac
        return term

    #   coulomb TERM
    def Q_term(self, x, i, j, kl, alP, alM, m, n, context, derivative=False):
        if j > i:
            i,j = j,i
        if n >= m:
            return 0
        term = 0
        cseq = self.seq.charges
        if cseq[m] > 0:
            qm = cseq[m] * alP
        elif cseq[m] < 0:
            qm = cseq[m] * alM
        else:
            return 0        # no contribution if either charge is 0
        if cseq[n] > 0:
            qn = cseq[n] * alP
        elif cseq[n] < 0:
            qn = cseq[n] * alM
        else:
            return 0        # no contribution if either charge is 0
        if derivative:
            fac = qm * qn * self.derA(m,n,x,kl)
        else:
            fac = qm * qn * self.A(m,n,x,kl)
        if (i >= m >= (j+1)) and ((m-1) >= n >= j):
            term += np.square(m-n) * fac
        if (i >= m >= j) and ((j-1) >= n) and context:
            term += np.square(m-j) * fac
        if (m >= (i+1)) and ((j-1) >= n) and context:
            term += np.square(i-j) * fac
        if (m >= (i+1)) and (i >= n >= j) and context:
            term += np.square(i-n) * fac
        return term

    #   2-body volume exclusions
    def Omega(self, i, j, alP, alM, wmn=(lambda m,n: 1), wii=None,wio=None,woi=None,woo=None, context=True):
        # Note: i,j used as Python indices, i.e. ranging 0..N-1
        Otot = 0
        Ocd = 0
        Odd = 0
        Ocdi = 0
        Oddi = 0
        cseq = self.seq.charges
        pseq = self.seq.polars
        for m in self.full_range[1:]:
            for n in range(0,m):
                otmn = self.Om_term(i,j, m,n, wii,wio,woi,woo, context)
                Otot += otmn * wmn(m,n)       # volume exclusion part (non-electrostatic)
                # charge and dipole assignments from ion condensation
                if cseq[m] > 0:
                    cm = abs(cseq[m]) * alP
                    dm = abs(cseq[m]) * (1-alP)
                elif cseq[m] < 0:
                    cm = abs(cseq[m]) * alM
                    dm = abs(cseq[m]) * (1-alM)
                else:
                    cm = 0
                    dm = 0
                if cseq[n] > 0:
                    cn = abs(cseq[n]) * alP
                    dn = abs(cseq[n]) * (1-alP)
                elif cseq[n] < 0:
                    cn = abs(cseq[n]) * alM
                    dn = abs(cseq[n]) * (1-alM)
                else:
                    cn = 0
                    dn = 0
                # dipole contributions from ion condensation
                Ocd += otmn * (cm*dn + cn*dm)
                Odd += otmn * dm*dn
                # dipole contributions from intrinsic polars
                idm = pseq[m]
                idn = pseq[n]
                Ocdi += otmn * (cm*idn + idm*cn)
                Oddi += otmn * idm*idn
        Otot = DTYPE( Otot / (i-j) )
        Ocd = DTYPE( Ocd / (i-j) )
        Odd = DTYPE( Odd / (i-j) )
        Ocdi = DTYPE( Ocdi / (i-j) )
        Oddi = DTYPE( Oddi / (i-j) )
        return Otot, Ocd, Odd, Ocdi, Oddi

    #   3-body volume exclusions
    def T(self, i, j):
        # Note: i,j used as Python indices, i.e. ranging 0..N-1
        if (self.mat_T[i,j] == 0) or (np.isnan(self.mat_T[i,j])):
            Btot = 0
            for l in self.full_range[2:]:
                for m in range(1,l):
                    for n in range(0,m):
                        Btot += self.T_term(i,j, l,m,n)
            self.mat_T[i,j] = DTYPE( Btot / (i-j) )
        return self.mat_T[i,j]

    #   electrostatic (Coulomb) attractions among chain monomers
    def Q(self, x, i, j, kl, alP, alM, context=True, derivative=False):
        tot = 0
        for m in self.full_range[1:]:
            for n in range(0,m):
                tot += self.Q_term(x,i,j, kl,alP,alM, m,n, context, derivative)
        return DTYPE( tot / (i-j) )

    #   function with details, screening, etc.
    def A(self, m, n, x, kl):
        res = 0.5 * ( np.sqrt(6*PI/x) ) * ( np.power(m-n, -1.5) )
        res += - kl * (0.5*PI/((m-n))) * erfcx( kl * np.sqrt(x*(m-n)/6) )
        return DTYPE(res)

    #   derivative of above function (d/dx) -> for analytic solution of 'w2' given some data point
    def derA(self, m, n, x, kl):
        res = - (1/(4*x)) * ( np.sqrt(6*PI/x) ) * ( np.power(m-n, -1.5) )
        res += (1/12) * np.square(kl) * ( np.sqrt(6*PI/x) ) * ( np.power(m-n,-0.5) )
        res += - (PI/12) * np.power(kl,3) * erfcx( kl * np.sqrt(x*(m-n)/6) )
        return DTYPE(res)

    #   SCD check based on Q (end-to-end, with division by N; i.e. N>>1)
    def SCD(self, alP=1, alM=1, x=1, kl=0):
        fac = 0.5 * ( np.sqrt(6*PI) ) * self.seq.N / (self.seq.N-1)
        return ( self.Q(x,self.seq.N-1,0,kl,alP,alM) / fac )

    #   SCDM extracted from Q (any i,j)
    def SCDM(self, i, j, alP=1, alM=1, x=1, kl=0):
        fac = 0.5 * ( np.sqrt(6*PI) )
        return ( self.Q(x,i,j,kl,alP,alM) / fac )

    #   SHDM extracted from Omega (any i,j)
    def SHDM(self, i, j, alP=1, alM=1, wmn=(lambda m,n: 1), wii=None,wio=None,woi=None,woo=None, context=True):
        om, ocd, odd, ocdi, oddi = self.Omega(i,j, alP,alM, wmn, wii,wio,woi,woo, context)
        return om

    #   FULL FREE ENERGY - all terms together; list/tuple/array for state variables
    def Ftot_ij(self, pmx, i, j, l=3.8, lB=7.12, cs=0.0, cp=6e-7, w2=0, w3=0.1, p=1.9, delta=1.3, Pint=0.95, \
            dipoleD2factor=True, F4factor=True, F4screen=False, F4p=None, kill_kappa=False, \
            wmn=(lambda m,n:1), wii=None, wio=None, woi=None, woo=None, context=True):
        # model/state variables are extracted as triplet
        (alP, alM, x) = pmx
        if any([(x<0.), (alP<0.), (alP>1.), (alM<0.), (alM>1.)]):
            return 0
        # Note: i,j used as Python indices, i.e. ranging 0..N-1
        if i<0 or j<0 or i>(self.seq.N-1) or j>(self.seq.N-1):
            print(f"\n* Warning: must keep i,j values as sequence indices!\t[given (i,j)=({i:},{j:})]\n")
            raise ValueError
        if j == i:
            return 0
        elif j > i:
            # functions take arguments with i>j, but symmetric by construction; set that way if reversed
            i,j = j,i
        # option: force different value of dipole length in F4 (ion pair formation); necessary if p=0 elsewhere
        dipole_F4 = F4p if F4p else p
        kl = 0 if kill_kappa else self.kapl(alP,alM,cp,cs,l,lB)
        res = self.F1(alP,alM)
        res += self.F2(alP,alM,cp,cs,l)
        res += self.F3(kl,cp,l)
        res += self.F4(alP,alM,l,lB,dipole_F4,delta,kl,F4factor,F4screen)
        res += self.F5_ij(alP,alM,x, i,j, l,lB, p,delta, kl, w2,w3, Pint, dipoleD2factor, wmn, wii,wio,woi,woo, context)
        return DTYPE(res)

    #   minimizer machine: handles all details of minimization while retaining raw 'result' object from SciPy 'minimize'
    def min_mach_Fij(self, i, j, method="NM-TNC", alBounds=(1e-6,1.0), xBounds=(1e-3,35), \
            ref=(0.5,0.5,1.0), init=(0.70,0.65,0.3), init_2=(0.10,0.15,1.10), scale=1, thr=1e-6, messages=False):
        # Note: i,j used as Python indices, i.e. ranging 0..N-1
        # establish core items
        var_bounds = (alBounds, alBounds, xBounds)          # combined variable bounds
        print_fmt = "(alP, alM, xij) = ({P:1.5g}, {M:1.5g}, {X:1.5g})"   # format for printing results
        def_init = (0.70,0.65,0.3)      # default initial point (in case none is given)
        # intro message (optional)
        if messages:
            t1 = perf_counter()     # begin timer
            print_fmt = "al+, al-, xij = {:, :, :}"
            self.parInfo()
            print(f"\nMINIMIZING Fij for pair (i,j) = ({i+1:},{j+1:}) ...")
            if init_2:
                print("\n ** 2-point Minimization **\t" + \
                    "pt1 = ({:.3g},{:.3g},{:.3g}) , pt2 = ({:.3g},{:.3g},{:.3g})\n".format(*init, *init_2))
            elif init:
                print("\n ** 1-point Minimization **\tpt1 = ({:.3g},{:.3g},{:.3g})\n".format(*init))
            else:
                print("\n  ** No initial point specified, using default **\tpt = ({:.3g},{:.3g},{:.3g})\n".format(*def_init))
        if init is None:
            init = def_init
        # define function with parameters and residue pair
        Fref = self.Ftot_ij(ref, i,j, **self.pars) if ref else 0     # reference value of Free Energy (if ref. pt. provided)
        minfunc = lambda pmx: ( ( self.Ftot_ij(pmx, i,j, **self.pars) - Fref ) * scale )  # shifted free energy for minimizing
        # handle methods
        mthd = method.lower()
        if mthd == "nelder-mead":   # direct Nelder-Mead approach
            result = minimize(minfunc, init, \
                        method="Nelder-Mead", bounds=var_bounds, \
                        options={"maxiter":30000, "xatol":thr, "fatol":thr, "adaptive":True})
        elif mthd == "tnc":         # direct TNC (truncated Newton) approach
            result = minimize(minfunc, init, \
                        method="tnc", bounds=var_bounds, options={"xtol":thr, "ftol":thr, "gtol":thr, "maxiter":30000})
        elif mthd == "nm-tnc":      # hybrid method: Nelder-Mead to find basin, TNC to hone in
            if messages:
                print("\t entering Nelder-Mead algorithm ...\n")
            res0 = minimize(minfunc, init, \
                        method="Nelder-Mead", bounds=var_bounds, \
                        options={"maxiter":30000, "xatol":thr, "fatol":thr, "adaptive":True})
            tncinit = res0.x        # N-M solution used as 'seed' (initial pt.) for TNC
            if messages:
                print("\t passing result " + (print_fmt.format(*(tncinit))) + " to TNC for refinement ...\n")
            result = self.min_mach_Fij(i,j, method="TNC", alBounds=alBounds, xBounds=xBounds, \
                ref=ref, init=tncinit, init_2=None, scale=scale, thr=thr, messages=messages)
        else:
            print("\n\nERROR: given method '{method}' is unsupported.\n\n")
            return
        # minimize from second initial point, if given (keep only if it's better)
        if init_2 is not None:
            res1 = result
            res2 = self.min_mach_Fij(i,j, method=mthd, alBounds=alBounds, xBounds=xBounds, \
                ref=ref, init=init_2, init_2=None, scale=scale, thr=thr, messages=messages)
            final_res = res1 if (res1.fun < res2.fun) else res2
            if messages:
                print("[FE1 = {res1.fun:2.5g}, FE2 = {res2.fun:2.5g}]")
                print("  >> FINAL RESULT : " + print_fmt.format(*(final_res.x)) + f' , FE = {final_res.fun:2.5g}' + "\n")
            return final_res
        else:
            final_res = result
        # results messages (optional)
        if messages:
            t2 = perf_counter()     # end timer
            print(f"\nDONE - elapsed time:\t{(t2-t1):2.5f}")
            print(f"\n[Extra Message: '{final_res.message}']")
            if final_res.success:
                print("\n\tSUCCESSFULLY found:\t" + print_fmt.format(*(final_res.x)) + "\n")
            else:
                print("\n\t** FAILED to find minimum;\t" + print_fmt.format(*(final_res.x)) + "\n")
        return final_res

    #   wrapper to minimize free energy: single-chain inter-residue 'Fij', with parameters set prior to calling
    def minFij(self, i,j, **minArgs):
        # use 'machine' for heavy lifting, but return only key information: minimized point (p,m,x) and Free Energy value f
        # reason: to enable use of raw 'result' object within machine, final return consistent in form with 'xModel_ij'
        rawres = self.min_mach_Fij(i,j, **minArgs)
        return rawres.x, rawres.fun

    #   minimize repeatedly for some varying parameter (pass 'minArgs' as keyword-arg dictionary to 'optimize')
    def multiMin(self, i, j, multiPar="cs", vals=[0], minArgs={}, seedNext=True, pH_seqinfo=False):
        # keep initial parameter value
        ival = self.allpars[multiPar]
        # prepare result list and fill by minimizing at each parameter setting
        pmxlist = []
        flist = []
        for i in range(len(vals)):
            prs = {multiPar:vals[i]}
            self.setPars(prs, pH_seqinfo=pH_seqinfo)
            pmxres, fres = self.minFij(i,j, **minArgs)
            pmxlist.append(pmxres)
            flist.append(fres)
            # use previous result as initial point for next iteration
            if seedNext:
                minArgs.update( {"init":pmxres} )
        # re-set parameter to initial setting
        self.setPars({multiPar:ival}, pH_seqinfo=pH_seqinfo)
        # arrange dictionary of results for transparency (separating out alpha +/- and xij)
        pmx = np.asarray(pmxlist)
        resdct = {multiPar:np.asarray(vals), 'alP':pmx[:,0], 'alM':pmx[:,1], 'xij':pmx[:,2], 'f':np.asarray(flist)}
        return resdct

    #   solve for W2 _numerically_ : using given calibration 'x' for pair i,j under established parameter settings
    def findW2(self, i, j, x_calib, x_thr=1e-6, w2bracket=(-10,10), minArgs={}, diff_metric=(lambda x,y: (y-x)), message=False):
        if j > i:
            i,j = j,i
        # use given 'w2' as seed / initial point
        w2_init = self.pars["w2"]
        # make function of 'w2' only, which will be numerically solved for zero by repeated optimization
        def w2Func(w2):
            # update parameter set
            self.pars.update( {"w2":w2} )
            # optimize at that point
            (alP, alM, x), f = self.minFij(i,j, **minArgs)
            # return difference (from some metric)
            return diff_metric(x, x_calib)
        if message:
            t1 = perf_counter()
            print("\nFinding 'w2' by iterating full optimization...")
        # find 'w2' at which the specified difference metric returns zero for given 'x_calib' and minimum location 'x'
        res = root_scalar(w2Func, x0=w2_init, bracket=w2bracket, xtol=x_thr, method="brentq")
        w2_res = res.root
        if message:
            t2 = perf_counter()
            print(f"DONE :\tw2={w2_res:2.4f}   (elapsed time: {(t2-t1):2.3f})")
            print(f"converged\t'{res.converged:}'\n[flag:  '{res.flag:}']")
        return w2_res

    #   solve for 'delta' and 'w2' _simultaneously_ by using two sets of parameters
    def findDandW2(self, i, j, x_calib1, x_calib2, pars1={'cs':0}, pars2={'cs':6e-7}, x_thr=1e-6, minArgs={}, \
            diff_metric=(lambda x,y: (y-x)), message=False):
        if j > i:
            i,j = j,i
        # use given 'delta' and 'w2' as seed / initial point
        dlt_init = self.pars["delta"]
        w2_init = self.pars["w2"]
        # make _vector_ function of 'delta' and 'w2', which will be numerically solved for zero by repeated optimization
        def dw2Func(vec):
            (d, w2) = vec       # delta and w2 from vector argument
            pars1.update( {"delta":d, "w2":w2} )    # update parameters and set them
            self.setPars(pars1, pH_seqinfo=False)   #
            (alP1, alM1, x1), f1 = self.minFij(i,j, **minArgs)      # minimum at first par. set
            pars2.update( {"delta":d, "w2":w2} )    #
            self.setPars(pars2, pH_seqinfo=False)   #
            (alP2, alM2, x2), f2 = self.minFij(i,j, **minArgs)      # minimum at second par. set
            # check difference (from some metric)
            return [diff_metric(x1, x_calib1), diff_metric(x2, x_calib2)]
        if message:
            t1 = perf_counter()
            print("\nFinding 'delta' & 'w2' by iterating full optimization...")
        # find pair ('delta', 'w2') at which the given metric returns zero _vector_
        res = root(dw2Func, (dlt_init, w2_init), method="df-sane", tol=x_thr)
        (d_res, w2_res) = tuple(res.x)
        if message:
            t2 = perf_counter()
            print(f"DONE :\tdlt={d_res:2.4f} ,  w2={w2_res:2.4f}   (elapsed time: {(t2-t1):2.3f})")
            print(f"converged\t'{res.success:}'\n[message:  '{res.message:}']")
        # return as _dictionary_
        return {"delta":d_res, "w2":w2_res}

    #   using derivative of Fij and Fkl to solve for 2-body and 3-body terms (w2 and w3) *simultaneously*
    def findW2W3(self, i, j, k, l, xij_calib, xkl_calib, x_thr=1e-6, minArgs={}, \
            diff_metric=(lambda x,y: (y-x)), message=False):
        if j > i:
            i,j = j,i
        if l > k:
            k,l = l,k
        # use given 'w2' and 'w3' as seed / initial point
        w2_init, w3_init = self.pars['w2'], self.pars['w3']
        # make _vector_ function of w2, w3 to make zero upon iterated minimization
        def w2w3Func(vec):
            (w2,w3) = vec
            (alPij, alMij, xij), fij = self.minFij(i,j, **minArgs)
            (alPkl, alMkl, xkl), fkl = self.minFij(k,l, **minArgs)
            # check difference (from some metric)
            return [diff_metric(xij, xij_calib1), diff_metric(xkl, xkl_calib2)]
        if message:
            t1 = perf_counter()
            print("\nFinding 'w2' & 'w3' by iterating full optimization...")
        # find pair ('w2', 'w3') at which the given metric returns zero _vector_
        res = root(w2w3Func, (w2_init, w3_init), method="df-sane", tol=x_thr)
        (w2_res, w3_res) = tuple(res.x)
        if message:
            t2 = perf_counter()
            print(f"DONE :\tw2={w2_res:2.4f} ,  w3={w3_res:2.4f}   (elapsed time: {(t2-t1):2.3f})")
            print(f"converged\t'{res.success:}'\n[message:  '{res.message:}']")
        # return as _dictionary_
        return {'w2':w2_res, 'w3':w3_res}

