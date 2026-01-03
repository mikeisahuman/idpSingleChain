##  Mike Phillips, 6/18/2025
##  Single-chain model for inter-residue factor 'xij', cf. Rij^2 = |i-j|*l*b*xij
##  Class / Object definition - encapsulating all functions
##  Each Model object holds a Sequence.
##      > with capability to calculate Free Energy, minimization for 'xij', etc.
##      > holds all necessary parameters, and enables (re-)setting them as desired
##
##  Require: sequence (name / aminos)
##           desired ij pair (NOT well-designed for full set of pairs, but it is possible to obtain full ij arrays)
##           key parameters > lB/l (i.e. Temperature)
##                          > screening kappa*l (i.e. ionic concentration)
##                          > two-body and three-body volume interactions w2, w3 (~ hydrophobicity related)
##                          > degrees of ionization (alpha_+/-)
##                          > optional custom pH and list of exclusions from pKa calcuations
##                          > dipole size p/l, dielectric mismatch delta=eps_w/eps_l  (only applicable if deg. of ion. < 1)
##                          > dipole length Pint for intrinsic polars
##
##  'T' matrix optimized for speed by saving calculated values in array with object.


import numpy as np
from scipy.special import erfcx as SCIerfcx     # SCALED complimentary error function: erfcx(u)=exp(u^2)*erfc(u)
from scipy.optimize import minimize
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
default_pars = {'l':3.8, 'lB':7.12, 'kappa':0, 'w2':0, 'w3':0.1, 'pH':None, 'pKex':(), 'alP':1, 'alM':1, 'p':0, 'delta':0, \
        'Pint':Pint_def, 'wmn':(lambda m,n:1), 'wii':None, 'wio':None, 'woi':None, 'woo':None, 'context':True}


#####   Class Defnition : object for encapsulating, evaluating 'simple' inter-residue IDP model #####
class xModel_ij:
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
                print("\t{:5}\t{:1.5g}".format(par, self.allpars[par]))
            except:
                print("\t{:5}\t{:6}".format(par, str(self.allpars[par])))
        print("")
        return

    ## Free Energy : as 'beta F / N'
    #   polymer free energy - chain entropy, intra-chain interactions (potential energy), specific to pair ij
    def Fij(self, x, i, j, l=1, lB=2, kappa=0, w2=0, w3=0, alP=1, alM=1, p=0, delta=0, Pint=0, \
            wmn=(lambda m,n:1), wii=None, wio=None, woi=None, woo=None, context=True):
        # Note: i,j used as Python indices, i.e. ranging 0..N-1
        if x < 0.0:
            return 0
        if i<0 or j<0 or i>(self.seq.N-1) or j>(self.seq.N-1):
            print(f"\n* Warning: must keep i,j values as sequence indices!\t[given (i,j)=({i:},{j:})]\n")
            raise ValueError
        if j == i:
            return 0
        elif j > i:
            # functions take arguments with i>j, but symmetric by construction; set that way if reversed
            i,j = j,i
        kl = kappa*l
        wcd, wdd = self.Wcddd(l,lB,kl,p,delta)
        wcdi, wddi = self.Wcddd(l,lB,kl,Pint,delta)
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

    #   minimize free energy: single-chain inter-residue 'Fij', for given set of parameters (set prior to calling this)
    def minFij(self, i, j, xinit=0.2, xinit2=2.5, x_bound=(1e-3,35), thr=1e-6, messages=False):
        # Note: i,j used as Python indices, i.e. ranging 0..N-1
        # intro message (optional)
        if messages:
            t1 = perf_counter()     # begin timer
            print_fmt = "xij = {x:1.5g}"
            self.parInfo()
            print(f"\nMINIMIZING Fij for pair (i,j) = ({i+1:},{j+1:}) ...")
            if xinit2:
                print(f"\n ** 2-point Nelder-Mead Minimization **\t  (x1, x2) = ({xinit:.3g}, {xinit2:.3g})\n")
            elif xinit:
                print(f"\n ** 1-point Nelder-Mead Minimization **\t  x1 = {xinit:.3g}\n")
            else:
                print("\n  ** Coarse grid search followed by Nelder-Mead Minimization **\n")
        # define function with parameters and residue pair
        func = lambda x: self.Fij(x, i, j, **self.pars)
        # coarse search for minimum as initial point, if 'xinit' not explicitly given
        if not xinit:
            xlist = np.linspace(x_bound[0], x_bound[1], round(20*x_bound[1]))
            flist = [func([xv]) for xv in xlist]
            ind = np.argmin(flist)
            xinit = xlist[ind]
        # minimize from initial point
        opt1 = minimize(func, [xinit], bounds=[x_bound], tol=thr, method='Nelder-Mead')
        # minimize from second initial point, if given (keep only if it's better)
        if xinit2:
            opt2 = minimize(func, [xinit2], bounds=[x_bound], tol=thr, method='Nelder-Mead')
            opt = opt1 if (opt1.fun < opt2.fun) else opt2
        else:
            opt = opt1
        xres = opt.x[0]
        fres = opt.fun
        # results messages (optional)
        if messages:
            t2 = perf_counter()     # end timer
            print(f"\nDONE  -  elapsed time:  {t2-t1:.5g}")
            print(f"    [Extra Message: '{opt.message:}']")
            if opt.success:
                print("\n\tSUCCESSFULLY found:\t  " + print_fmt.format(x=xres) + "\n")
            else:
                print("\n\tFAILED to find minimum;\t  " + print_fmt.format(x=xres) + "\n")
        return xres, fres

    #   minimize repeatedly for some varying parameter (pass 'minArgs' as keyword-arg dictionary to 'optimize')
    def multiMin(self, i, j, multiPar="kappa", vals=[0], minArgs={}, seedNext=True, pH_seqinfo=False, \
            multiALP=None, multiALM=None):
        # keep initial parameter value
        ival = self.allpars[multiPar]
        # check for custom degrees of ionization
        alp_ival = self.allpars['alP']
        if multiALP:
            if len(multiALP) != len(vals):
                print("\n - ERROR: custom ALP array is not the same length as 'vals' array.")
                return
        alm_ival = self.allpars['alM']
        if multiALM:
            if len(multiALM) != len(vals):
                print("\n - ERROR: custom ALM array is not the same length as 'vals' array.")
                return
        # prepare result list and fill by minimizing at each parameter setting
        xlist = []
        flist = []
        for i in range(len(vals)):
            prs = {multiPar:vals[i]}
            if multiALP:
                prs.update({'alP':multiALP[i]})
            if multiALM:
                prs.update({'alM':multiALM[i]})
            self.setPars(prs, pH_seqinfo=pH_seqinfo)
            xres, fres = self.minFij(i,j, **minArgs)
            xlist.append(xres)
            flist.append(fres)
            # use previous result as initial point for next iteration
            if seedNext:
                minArgs.update( {"xinit":xres} )
        # re-set parameter to initial setting
        self.setPars({multiPar:ival, 'alP':alp_ival, 'alM':alm_ival}, pH_seqinfo=pH_seqinfo)
        # arrange dictionary of results for transparency
        resdct = {multiPar:np.asarray(vals), 'xij':np.asarray(xlist), 'f':np.asarray(flist)}
        if multiALP:
            resdct.update({'alP':np.asarray(multiALP)})
        if multiALM:
            resdct.update({'alM':np.asarray(multiALM)})
        return resdct

    #   using derivative of Fij to solve for 2-body term
    def findW2(self, i, j, x, check=1e-4, message=False, minArgs={}):
        if j > i:
            i,j = j,i
        # grab pars from dict (to enable direct comparison between solved w2 at given x, and optimized x at solved w2)
        pkeys = ('l', 'lB', 'kappa', 'w3', 'alP', 'alM', 'p', 'delta', 'Pint', 'wmn', 'wii', 'wio', 'woi', 'woo', 'context')
        l, lB, kappa, w3, alP, alM, p, delta, Pint, wmn, wii, wio, woi, woo, context = [ self.pars[k] for k in pkeys ]
        kl = kappa*l
        wcd, wdd = self.Wcddd(l,lB,kl,p,delta)
        wcdi, wddi = self.Wcddd(l,lB,kl,Pint,delta)
        om, ocd, odd, ocdi, oddi = self.Omega(i,j, alP,alM, wmn, wii,wio,woi,woo, context)
        t = self.T(i,j) if w3!=0 else 0
        q = self.Q(x,i,j, kappa,alP,alM, context, derivative=True)
        # numerator
        num = 1.5*(1-1/x)       # Gaussian chain term
        num += - w3 * (3/x) * np.power(3/(2*PI*x),3) * t / 2        # 3-body term
        num += (2*lB/(PI*l)) * q    # electrostatics _derivative_ (d/dx)
        # denominator
        den = (3/(2*x)) * np.power(3/(2*PI*x),1.5)      # factor on 2-body term
        # divide
        res = num / den
        # subtract charge-dipole & dipole-dipole pieces
        res -= wcd*ocd + wdd*odd + wcdi*ocdi + wddi*oddi
        # divide by 2-body volume sum factor to get final result
        res /= om
        # use result of 'w2' to minimize Fij for 'x' to confirm self-consistency
        tmp = self.pars['w2']
        self.setPars({'w2':res}, pH_seqinfo=False)
        if check:
            f = self.Fij(x, i, j, **self.pars)
            x_min, f_min = self.minFij(i,j, **minArgs)
            if (abs(x - x_min) > check) and (f_min < f):
                print(f"\nWARNING at (i,j)=({i+1:},{j+1:}): 2-body solution w2={res:.5f} did _not_ reproduce expected minimum x={x:.5f}!!")
                print(f"\t[instead gave x_opt={x_min:.5f}]\n")
                self.setPars({'w2':tmp}, pH_seqinfo=False)
                return None
            elif message:
                print(f"\nOptimal 'x' consistent with input at (i,j)=({i+1:},{j+1:}): 2-body w2={res:.5f}.")
                print(f"\t[from minimizer: x={x_min:}]\n")
        return res

    #   using derivative of Fij and Fkl to solve for 2-body and 3-body terms (w2 and w3) *simultaneously*
    def findW2W3(self, i, j, k, l, xij, xkl, check=1e-4, message=False, minArgs={}):
        if j > i:
            i,j = j,i
        if l > k:
            k,l = l,k
        # grab pars from dict
        pkeys = ('l', 'lB', 'kappa', 'alP', 'alM', 'p', 'delta', 'Pint', 'wmn', 'wii', 'wio', 'woi', 'woo', 'context')
        b, lB, kappa, alP, alM, p, delta, Pint, wmn, wii, wio, woi, woo, context = [ self.pars[k] for k in pkeys ]
        kl = kappa*b
        wcd, wdd = self.Wcddd(b,lB,kl,p,delta)
        wcdi, wddi = self.Wcddd(b,lB,kl,Pint,delta)
        pifac = 1.5/PI
        # matrix elements for ij and kl segments
        om_ij, ocd_ij, odd_ij, ocdi_ij, oddi_ij = self.Omega(i,j, alP,alM, wmn, wii,wio,woi,woo, context)
        dips_ij = ocd_ij*wcd + odd_ij*wdd + ocdi_ij*wcdi + oddi_ij*wddi
        t_ij = self.T(i,j)
        q_ij = self.Q(xij,i,j, kappa,alP,alM, context, derivative=True)
        om_kl, ocd_kl, odd_kl, ocdi_kl, oddi_kl = self.Omega(k,l, alP,alM, wmn, wii,wio,woi,woo, context)
        dips_kl = ocd_kl*wcd + odd_kl*wdd + ocdi_kl*wcdi + oddi_kl*wddi
        t_kl = self.T(k,l)
        q_kl = self.Q(xkl,k,l, kappa,alP,alM, context, derivative=True)
        # right-hand-side quantities
        rhs = np.asarray( [(1-1/xij), (1-1/xkl)] )
        rhs += - np.asarray( [(dips_ij)*np.power(pifac/xij,1.5)/xij, (dips_kl)*np.power(pifac/xkl,1.5)/xkl] )
        rhs += (8*pifac/9)*(lB/b) * np.asarray( [q_ij, q_kl] )
        # left-hand-side matrix
        mat = np.asarray( [[om_ij*np.power(pifac/xij,1.5)/xij, t_ij*np.power(pifac/xij,3)/xij], [om_kl*np.power(pifac/xkl,1.5)/xkl, t_kl*np.power(pifac/xkl,3)/xkl]] )
        # solution given by inversion and multiplication
        resw2, resw3 = np.linalg.inv(mat) @ rhs
        # use result of w2,w3 to minimize for xij,xkl to confirm self-consistency
        tmp2, tmp3 = self.pars['w2'], self.pars['w3']
        self.setPars({'w2':resw2, 'w3':resw3}, pH_seqinfo=False)
        if check:
            fij = self.Fij(xij, i, j, **self.pars)
            xij_min, fij_min = self.minFij(i,j, **minArgs)
            fkl = self.Fij(xkl, k, l, **self.pars)
            xkl_min, fkl_min = self.minFij(k,l, **minArgs)
            bad_ij = ( (abs(xij - xij_min) > check) and (fij_min < fij) )
            bad_kl = ( (abs(xkl - xkl_min) > check) and (fkl_min < fkl) )
            if bad_ij or bad_kl:
                print(f"\nWARNING at (i,j)=({i+1:},{j+1:}) & (k,l)=({k+1:},{l+1:}): 2-body & 3-body solutions w2={resw2:.5f} & w3={resw3:.5f} did _not_ reproduce expected minima xij={xij:.5f} & xkl={xkl:.5f}!!")
                print(f"\t[instead gave xij_min={xij_min:.5f} & xkl_min={xkl_min:.5f}]\n")
                self.setPars({'w2':tmp2, 'w3':tmp3}, pH_seqinfo=False)
                return (None,None)
            elif message:
                print(f"\nOptimal xij,xkl consistent with input at (i,j)=({i+1:},{j+1:}) & (k,l)=({k+1:},{l+1:}): 2-body & 3-body solutions w2={resw2:.5f} & w3={resw3:.5f}.")
                print(f"\t[from minimizer: xij={xij_min:} & xkl={xkl_min:}]\n")
        return {'w2':resw2, 'w3':resw3}

