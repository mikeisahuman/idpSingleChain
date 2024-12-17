##  Mike Phillips, 4/16/2024
##  Single-chain model for end-to-end factor 'x', cf. Ree^2 = N*l*b*x.
##  Class / Object definition - encapsulating all functions
##  Each Model object holds a Sequence.
##      > with capability to calculate Free Energy, minimization for 'x', etc.
##      > holds all necessary parameters, and enables (re-)setting them as desired


import numpy as np
from scipy.special import erfcx as SCIerfcx     # SCALED complimentary error function: erfcx(u)=exp(u^2)*erfc(u)
from scipy.optimize import minimize


# GLOBAL Data Type
DTYPE = np.float64
PI = DTYPE(np.pi)       # appropriate value for 'pi'
erfcx = lambda u: SCIerfcx(u, dtype=DTYPE)  # incorporating data type in supplied 'erfcx(u)'


# GLOBAL settings for intrinsic polars
Pint_def = 3.8*(0.5)*(0.5)      # default dipole moment for intrinsic polars: half charges separated by half bond length
#polars = {"B":1}       # fictitious polar (neutral charge) amino acid
#polars = {"S":1, "Y":1, "Q":1, "N":1, "T":1}      # yes: ser tyr gln asn ; maybe possible: cys gly thr , trp his
polars = {}       # blank dictionary - to neglect intrinsically polar residues

pol_msg = "_NOT_" if (len(polars) == 0) else "_INDEED_"
pol_secondl = "" if (len(polars) == 0) else "\tpolars = {polars:},  Pint = {Pintrinsic:}\n"
print(f"\nFROM 'xModel' MODULE:\n\tyou are {pol_msg:} including intrinsic polars!")
print(pol_secondl)


# dictionary of _all_ default parameter settings; any/all can be overridden as desired with 'setPars'
#   > if including intrinsic polars: must specify the intrinsic dipole moment
#   > Pint = d*q  ['d' in units matching 'l', 'q' in units of elementary charge]
default_pars = {'l':3.8, 'lB':7.12, 'kappa':0, 'w2':0, 'w3':0.1, 'pH':None, 'pKex':(), 'alP':1, 'alM':1, 'p':0, 'delta':0, 'Pint':Pint_def}


#####   Class Definition : object for encapsulating, evaluating 'simple' isolated IDP model #####
class xModel:
    def __init__(self, seq, info=False, OBfile=None):
        self.seq = seq      # sequence object is a necessary input
        self.seq.polars = self.translate_polar()    # include sequence as polars
        if info:
            self.seq.info()
        # load 2- and 3-body sums if 'OBfile' specified, otherwise calculate immediately
        self.OBfile = OBfile
        if OBfile:
            OBarr = np.load(OBfile)
            try:
                OBind = np.where(OBarr[:,0]==self.seq.N)[0][0]
                self.Onon, self.B = OBarr[OBind, 1:]
            except IndexError:
                OBfile = False
        if not OBfile:
            Onon = 1
            B = 0
            for l in range(2,self.seq.N):
                Onon += np.power(l, -0.5)
                for m in range(1,l):
                    Onon += np.power(l-m, -0.5)
                    for n in range(0,m):
                        B += (l-n) * np.power((l-m)*(m-n), -1.5)
            self.Onon = DTYPE( Onon / self.seq.N )
            self.B = DTYPE( B / self.seq.N )
        # load in baseline parameters
        self.allpars = {}   # blank initialization to load defaults
        self.setPars()
    #   #   #   #   #   #

    #   translate polars (only relevant if polars are included in theory, cf. 'polars' dictionary)
    def translate_polar(self):
        lst = [(polars[c] if (c in polars) else 0) for c in self.seq.aminos]
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
    def parInfo(self, exclude=('Pint',)):
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
    #   polymer free energy - chain entropy, intra-chain interactions (potential energy)
    def F(self, x, l=1, lB=2, kappa=0, w2=0, w3=0, alP=1, alM=1, p=0, delta=0, Pint=0):
        if x < 0.0:
            return 0
        kl = kappa*l
        res = 1.5 * (x - np.log(x, dtype=DTYPE))
        res += w3 * np.power(3/(2*PI*x), 3) * self.B / 2
        res += np.power(3/(2*PI*x), 1.5) * self.Omega(w2, alP, alM, l, lB, kl, p, delta, Pint)
        res += (2*lB/(PI*l)) * self.Q(alP, alM, x, kl)
        return DTYPE(res / self.seq.N)

    #   2-body volume exclusions (including any effective c-d & d-d interactions)
    def Omega(self, w2, alP, alM, l, lB, kl, p, delta, Pint):
        om = w2*self.Onon
        if (delta > 1e-3):
            if (p > 1e-3):
                ocd, odd = self.Ocddd(alP,alM)
                wcd, wdd = self.Wcddd(l,lB,kl,p,delta)
                om += wcd*ocd + wdd*odd
            if polars and Pint:
                ocdi, oddi = self.Ocddd_int(alP,alM)
                wcdi, wddi = self.Wcddd(l,lB,kl,Pint,delta)
                om += wcdi*ocdi + wddi*oddi
        return DTYPE(om)

    #   charge-dipole and dipole-dipole interaction weights (after directional averaging and casting as delta functions)
    def Wcddd(self, l, lB, kl, p, delta):
        wcd = - delta*delta * (PI/3) * np.square(lB*p/(l*l)) * (2 + kl) * np.exp(-2*kl, dtype=DTYPE)
        wdd = - delta*delta * (PI/9) * np.square(lB*p*p/(l*l*l)) * (4 + kl*(8 + kl*(4 + kl))) * np.exp(-2*kl, dtype=DTYPE)
        return wcd, wdd

    #   charge-dipole and dipole-dipole short-range interaction metrics
    def Ocddd(self, alP, alM):
        N = self.seq.N
        pseq = self.seq.charges
        ocd = 0
        odd = 0
        for m in range(1,N):
            if pseq[m] == 0:
                continue
            for n in range(0,m):
                if pseq[n] == 0:
                    continue
                if pseq[m] > 0:
                    cm = abs(pseq[m])*alP
                    dm = abs(pseq[m])*(1-alP)
                else:
                    cm = abs(pseq[m])*alM
                    dm = abs(pseq[m])*(1-alM)
                if pseq[n] > 0:
                    cn = abs(pseq[n])*alP
                    dn = abs(pseq[n])*(1-alP)
                else:
                    cn = abs(pseq[n])*alM
                    dn = abs(pseq[n])*(1-alM)
                ocd += (cm*dn + dm*cn) / np.sqrt(m-n)
                odd += dm*dn / np.sqrt(m-n)
        return DTYPE(ocd / N), DTYPE(odd / N)

    #   intrinsic charge-dipole and dipole-dipole short-range interaction metrics
    def Ocddd_int(self, alP, alM):
        N = self.seq.N
        pseq = self.seq.charges
        polarseq = self.seq.polars
        ocd = 0
        odd = 0
        for m in range(1,N):
            for n in range(0,m):
                dm = polarseq[m]
                dn = polarseq[n]
                if pseq[m] > 0:
                    cm = abs(pseq[m]) * alP
                elif pseq[m] < 0:
                    cm = abs(pseq[m]) * alM
                else:
                    cm = 0
                if pseq[n] > 0:
                    cn = abs(pseq[n]) * alP
                elif pseq[n] < 0:
                    cn = abs(pseq[n]) * alM
                else:
                    cn = 0
                ocd += (cm*dn + dm*cn) / np.sqrt(m-n)
                odd += dm*dn / np.sqrt(m-n)
        return DTYPE(ocd / N), DTYPE(odd / N)

    #   electrostatic attractions among chain monomers
    def Q(self, alP, alM, x, kl=0, derivative=False):
        N = self.seq.N
        pseq = self.seq.charges
        Afunc = self.derA if derivative else self.A
        total = 0
        for m in range(1,N):
            if pseq[m] == 0:
                continue
            for n in range(0,m):
                if pseq[n] == 0:
                    continue
                if pseq[m] > 0:
                    qm = pseq[m] * alP
                else:
                    qm = pseq[m] * alM
                if pseq[n] > 0:
                    qn = pseq[n] * alP
                else:
                    qn = pseq[n] * alM
                total += qm * qn * np.square(m-n) * Afunc(m,n,x,kl)
        return DTYPE(total / N)
    #   function with details, screening, etc.
    def A(self, m, n, x, kl):
        res = 0.5 * np.sqrt(6*PI/x) * np.power(m-n,-1.5)
        res += - kl * (0.5*PI/(m-n)) * erfcx( kl * np.sqrt(x*(m-n)/6) )
        return DTYPE(res)
    #   derivative of above function (d/dx) -> for (quasi-)analytic solution of 'w2' given some data point
    def derA(self, m, n, x, kl):
        res = - (1/(4*x)) * np.sqrt(6*PI/x) * np.power(m-n,-1.5)
        res += (1/12) * (kl*kl) * np.sqrt(6*PI/x) / np.sqrt(m-n)
        res += - (PI/12) * (kl*kl*kl) * erfcx( kl * np.sqrt(x*(m-n)/6) )
        return DTYPE(res)

    #   SCD check based on Q
    def SCD(self, alP=1, alM=1, x=1, kl=0):
        fac = 0.5 * np.sqrt(6*PI)
        return ( self.Q(alP,alM,x,kl) / fac )

    #   minimize free energy for simple models : either single-chain 'F' (see above), or Higgs-Joanny formulation (see below)
    def minF(self, xinit=0.5, xinit2=None, x_bound=(1e-3,35), thr=1e-6,  info=False, HJ=False):
        if info:
            self.parInfo()
        # define parameters and free energy for given 'x' model
        if not HJ:
            pdt = self.pars
            func = lambda xl: self.F(xl[0], **pdt)
        else:
            pdt = {k:self.pars[k] for k in ('l', 'lB', 'kappa', 'w2', 'w3', 'alP', 'alM')}
            func = lambda xl: self.F_hj(xl[0], **pdt)
        # coarse search for minimum as initial point, if 'xinit' not explicitly given
        if not xinit:
            xlist = np.linspace(x_bound[0], x_bound[1], round(20*x_bound[1]))
            flist = [func([xv]) for xv in xlist]
            ind = np.argmin(flist)
            xinit = xlist[ind]
        # minimize from initial point
        # > alternatively, consider using 'direct' minimizer
#        opt = direct(func, (x_bound,), len_tol=thr, maxfum=int(1e4))
        opt = minimize(func, (xinit,), method="Nelder-Mead", bounds=(x_bound,), tol=thr)
        xres = (opt.x)[0]
        fres = opt.fun
        # minimize from second initial point, if given (keep only if it's better)
        if xinit2:
            opt2 = minimize(func, (xinit2,), method="Nelder-Mead", bounds=(x_bound,), tol=thr)
            if opt2.fun < opt.fun:
                xres = (opt2.x)[0]
                fres = opt2.fun
        return xres, fres

    #   minimize 'F' repeatedly for some varying parameter (pass 'minArgs' as keyword-arg dictionary to 'minF')
    def multiMin(self, multiPar="kappa", vals=[0], minArgs={}, seedNext=True, pH_seqinfo=False):
        # keep initial parameter value
        ival = self.allpars[multiPar]
        # prepare result list
        xlist = []
        flist = []
        for pval in vals:
            self.setPars({multiPar:pval}, pH_seqinfo=pH_seqinfo)
            xres, fres = self.minF(**minArgs)
            xlist.append(xres)
            flist.append(fres)
            if seedNext:
                minArgs.update( {"xinit":xres} )
        self.setPars({multiPar:ival}, pH_seqinfo=pH_seqinfo)
        return {multiPar:np.asarray(vals), 'x':np.asarray(xlist), 'f':np.asarray(flist)}

    #   solve for 2-body term from standard 'x' model (or Higgs-Joanny; see below), and check self-consistency
    def findW2(self, x=1, check=1e-4, minArgs={}):
        HJ = minArgs['HJ'] if 'HJ' in minArgs else False
        res = self.solveW2fromX(x) if not HJ else self.solveW2fromHJ(x)
        # store result in parameters (at least for checking)
        tmp = self.pars['w2']       # current value of w2, for reverting if self-consistency check fails
        self.setPars({'w2':res})
        # use result of 'w2' to minimize 'F' w/r/t 'x', to confirm self-consistent result
        if check:
            x_sol, f_sol = self.minF(**minArgs)
            if abs(x - x_sol) > check:
                print("WARNING: 2-body solution w2=%1.5f did _not_ reproduce expected minimum x=%2.4f!!" % (res,x))
                print("\t[instead gave x_opt=%2.4f]\n" % x_sol)
                self.setPars({'w2':tmp})
                return None
        return res

    #   solve for 2-body term from standard 'x' model
    def solveW2fromX(self, x=1):
        l, lB, kappa, w3, alP, alM, p, delta, Pint = [ self.pars[k] for k in ('l', 'lB', 'kappa', 'w3', 'alP', 'alM', 'p', 'delta', 'Pint') ]
        kl = kappa*l
        # numerator
        num = 1.5*(1-1/x)       # Gaussian chain term
        num += - w3 * (3/x) * np.power(3/(2*PI*x),3) * self.B / 2
        num += (2*lB/(PI*l)) * self.Q(alP,alM,x,kl, derivative=True)      # electrostatics _derivative_ (d/dx)
        # denominator
        den = (3/(2*x)) * np.power(3/(2*PI*x),1.5)      # factor on 2-body term(s)
        # divide
        res = num / den
        # subtract other 2-body terms (i.e. dipoles) : just full 'Omega' term, with w2=0
        res += - self.Omega(0, alP, alM, l, lB, kl, p, delta, Pint)
        # divide by 2-body volume sum factor to get final result
        res /= self.Onon
        return res

    #   polymer free energy - compositional, from Higgs-Joanny formulation [cf. JCP 1991] ; amended with 3-body term
    def F_hj(self, x, l=1, lB=2, kappa=0, w2=0, w3=0, alP=1, alM=1):
        total = alP*self.seq.fracp + alM*self.seq.fracm
        diff = alP*self.seq.fracp - alM*self.seq.fracm
        kl = kappa*l
        amph = - PI*(lB*lB/(l*l))*(total*total)/kl
        elec = 4*PI*(lB/l)*(diff*diff)/(kl*kl)
        v_star = w2 + amph + elec
        om = self.Onon
        o3 = w3 * np.power(2*PI*x/3, -3) * self.B / 2       # with 3-body term
        return ( (3/2)*(x-np.log(x)) + np.power(2*PI*x/3, -1.5) * v_star * om + o3)

    #   finding two-body exclusion 'w2' (or 'v') under Higgs-Joanny model, given 'x'
    def solveW2fromHJ(self, x=1):
        l, lB, kappa, w3, alP, alM = [ self.pars[k] for k in ('l', 'lB', 'kappa', 'w3', 'alP', 'alM') ]
        total = alP*self.seq.fracp + alM*self.seq.fracm
        diff = alP*self.seq.fracp - alM*self.seq.fracm
        kl = kappa*l
        amph = - PI*(lB*lB/(l*l))*(total*total)/kl
        elec = 4*PI*(lB/l)*(diff*diff)/(kl*kl)
        om = self.Onon
        res = np.power(x, 2.5) - np.power(x, 1.5)
        res -= w3 * np.power(2*PI*np.sqrt(x)/3, -3) * self.B       # 3-body term : derivative
        res *= np.power(2*PI/3, 1.5) / om
        return (res - amph - elec)

#####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####

