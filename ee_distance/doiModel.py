##  Mike Phillips, 6/25/2024
##  Single-chain model for end-to-end factor 'x', cf. Ree^2 = N*l*b*x
##   alongside degrees of ionization 'alp' 'alm'  (fraction of charge on +/- ions).
##  Class / Object definition - encapsulating all functions
##  Each Model object holds a Sequence.
##      > with capability to calculate Free Energy, minimization for triplet ('alp','alm','x'), etc.
##      > holds all necessary parameters, and enables (re-)setting them as desired
##      > check Free Energy values, term-by-term or total
##      > minimize Free Energy with 'minF' (under current parameter setting)
##      > minimize FE under varying parameter values with 'multiMin' or 'multiMin_T'
##      > determine w2 by matching x at given salt cs with 'findW2'
##      > determine delta & w2 by matching x values at two given salt values with 'findDandW2'


import numpy as np
from scipy.special import erfcx as SCIerfcx     # SCALED complimentary error function: erfcx(u)=exp(u^2)*erfc(u)
from scipy.optimize import minimize
from scipy.optimize import root_scalar
from scipy.optimize import root
from time import perf_counter       # for timing


##  Global Data Type
DTYPE = np.float64
PI = DTYPE(np.pi)       # appropriate value for 'pi'
erfcx = lambda u: SCIerfcx(u, dtype=DTYPE)

# GLOBAL settings for intrinsic polars
Pint_def = 3.8*(0.5)*(0.5)      # default dipole moment for intrinsic polars: half charges separated by half bond length
#polars = {"B":1}       # fictitious polar (neutral charge) amino acid
#polars = {"S":1, "Y":1, "Q":1, "N":1, "T":1}      # yes: ser tyr gln asn ; maybe possible: cys gly thr , trp his
polars = {}       # blank dictionary - to neglect intrinsically polar residues

pol_msg = "_NOT_" if (len(polars) == 0) else "_INDEED_"
pol_secondl = "" if (len(polars) == 0) else "\tpolars = {polars:},  Pint = {Pint_def:}\n"
print(f"\nFROM 'doiModel' MODULE:\n\tyou are {pol_msg:} including intrinsic polars!")
print(pol_secondl)


# dictionary of _all_ default parameter settings; any/all can be overridden as desired with 'setPars'
#   > if including intrinsic polars: must specify the intrinsic dipole moment
#   > Pint = d*q  ['d' in units matching 'l', 'q' in units of elementary charge]
#   > concentrations 'cs' (salt) and 'cp' (residue/monomer) should be in units 1/[l^3]  : if milli-Molar, use *6.022e-7 for cubic Angstrom
#   > dipole size 'p' involves pair separation distance and ion pair charge  [units of 'l', and elementary charge]
#   > dielectric mismatch 'delta' is dimensionless by definition : ratio of (dim.less) dielectric constants, eps_water/eps_local
default_pars = {'l':3.8, 'lB':7.12, 'cs':0, 'cp':6e-7, 'w2':0, 'w3':0.1, 'p':1.9, 'delta':1.3, 'pH':None, 'pKex':(), 'Pint':Pint_def, \
                'dipoleD2factor':True, 'F4factor':True, 'F4screen':False, 'F4p':None, 'kill_kappa':False, 'ignoreBases':False}


#####   Class Definition : object for encapsulating, evaluating 'degree of ionization' isolated IDP model #####
class doiModel:
    def __init__(self, seq, info=False, OBfile=None):
        self.seq = seq      # sequence object is a necessary input
        self.seq.polars = self.translate_polar()    # include sequence as polars
        if info:
            self.seq.info()
        # load non-electrostatic 2- and 3-body sums if 'OBfile' specified, otherwise calculate immediately
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
        # simple functions for charge-dipole & dipole-dipole weights
        self.wcd = lambda l,lB,p,kl: DTYPE( - (PI/3) * np.square(lB*p/(l*l)) * np.exp(-2*kl, dtype=DTYPE) * (2+kl) )
        self.wdd = lambda l,lB,p,kl: DTYPE( - (PI/9) * np.square(lB*p*p/(l*l*l)) * np.exp(-2*kl, dtype=DTYPE) * (4+8*kl+4*kl*kl+kl*kl*kl) )
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
    def parInfo(self, exclude=('Pint', 'F4factor', 'F4override', 'kill_kappa', 'ignoreBases')):
        print("\ndoiModel parameters:")
        print("\n\tPAR\tVALUE")
        includedKeys = [k for k in self.allpars.keys() if k not in exclude]
        for par in includedKeys:
            try:
                print("\t{:5}\t{:1.5g}".format(par, self.allpars[par]))
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

    ## Free Energy Terms : always 'beta F / N' -> some overall factors of density '1/cp'
    #   combinatorial entropy - placement of counterions on polymer chain
    def F1(self, alP, alM):
        res = 0
        if 0.0 < alP < 1.0:
            res += self.seq.fracp * alP*np.log(alP, dtype=DTYPE)
            res += self.seq.fracp * (1-alP)*np.log(1-alP, dtype=DTYPE)
#        else:
#            return np.nan
        if 0.0 < alM < 1.0:
            res += self.seq.fracm * alM*np.log(alM, dtype=DTYPE)
            res += self.seq.fracm * (1-alM)*np.log(1-alM, dtype=DTYPE)
#        else:
#            return np.nan
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

    #   polymer free energy - chain entropy, intra-chain interactions (potential energy)
    def F5(self, alP, alM, x, l, lB, p, dlt, kl, w2, w3, Pint, dipoleD2factor=True, ignoreBases=False):
        if x < 0.0:
            return np.nan
        om = self.Onon
        D2 = (dlt*dlt) if dipoleD2factor else 1
        wcd = self.wcd(l,lB,p,kl) * D2
        wdd = self.wdd(l,lB,p,kl) * D2
        wcdi = self.wcd(l,lB,Pint,kl) * D2
        wddi = self.wdd(l,lB,Pint,kl) * D2
        ocd, odd, ocdi, oddi = self.eOmega(alP, alM, ignoreBases)
        res = 1.5 * (x - np.log(x, dtype=DTYPE))
        res += w3 * np.power(3/(2*PI*x),3) * self.B / 2
        res += np.power(3/(2*PI*x),1.5) * (w2*om + wcd*ocd + wdd*odd + wcdi*ocdi + wddi*oddi)
        res += (2*lB/(PI*l)) * self.Q(alP, alM, x, kl)
        return DTYPE(res / self.seq.N)

    #   2-body short-range _electrostatic_ interactions (effective c-d & d-d interactions, arising from condensation and intrinsic)
    def eOmega(self, alP, alM, ignoreBases=False):
        N = int(self.seq.N)
        chseq = self.seq.charges
        poseq = self.seq.polars
        ocd = 0     # interactions involving dipoles that arise from condensation of ions
        odd = 0     #
        ocdi = 0    # interactions involving intrinsic polars
        oddi = 0    #
        for m in range(1,N):
            if (chseq[m] == 0) and (poseq[m] == 0):
                continue
            for n in range(0,m):
                if (chseq[n] == 0) and (poseq[n] == 0):
                    continue
                # for charge-dipole interaction, charge is always positive (and so are dipoles)
                if chseq[m] > 0:
                    cm = abs(chseq[m])*alP
                    if ignoreBases:
                        dm = 0
                    else:
                        dm = abs(chseq[m])*(1-alP)
                elif chseq[m] < 0:
                    cm = abs(chseq[m])*alM
                    dm = abs(chseq[m])*(1-alM)
                else:
                    cm,dm = 0,0
                if chseq[n] > 0:
                    cn = abs(chseq[n])*alP
                    if ignoreBases:
                        dn = 0
                    else:
                        dn = abs(chseq[n])*(1-alP)
                elif chseq[n] < 0:
                    cn = abs(chseq[n])*alM
                    dn = abs(chseq[n])*(1-alM)
                else:
                    cn,dn = 0,0
                # intrinsic dipoles: rely on dictionary 'polars' for weights, keep positive
                idm, idn = abs(poseq[m]), abs(poseq[n])
                div = np.sqrt(m-n)
                ocd += (cm*dn + dm*cn) / div
                odd += dm*dn / div
                ocdi += (cm*idn + idm*cn) / div
                oddi += idm*idn / div
        return DTYPE(ocd/N), DTYPE(odd/N), DTYPE(ocdi/N), DTYPE(oddi/N)

    #   electrostatic (charge-charge) attractions among chain monomers
    def Q(self, alP, alM, x, kl, derivative=False):
        N = int(self.seq.N)
        chseq = self.seq.charges
        if derivative:
            Afunc = self.derA
        else:
            Afunc = self.A
        total = 0
        for m in range(1,N):
            if chseq[m] == 0:
                continue
            for n in range(0,m):
                if chseq[n] == 0:
                    continue
                if chseq[m] > 0:
                    qm = chseq[m] * alP
                else:
                    qm = chseq[m] * alM
                if chseq[n] > 0:
                    qn = chseq[n] * alP
                else:
                    qn = chseq[n] * alM
                total += qm * qn * np.square(m-n) * Afunc(m,n,x,kl)
        return DTYPE(total/N)

    #   function with details, screening, etc.
    def A(self, m, n, x, kl):
        res = 0.5 * np.sqrt(6*PI/x) * np.power(m-n, -1.5)
        res += - kl * (0.5*PI/((m-n))) * erfcx( kl * np.sqrt(x*(m-n)/6) )
        return DTYPE(res)

    #   derivative of above function (d/dx) -> for (quasi-)analytic solution of 'w2' given some data point
    def derA(self, m, n, x, kl):
        res = - (1/(4*x)) * np.sqrt(6*PI/x) * np.power(m-n, -1.5)
        res += (1/12) * (kl*kl) * np.sqrt(6*PI/((m-n)*x))
        res += - (PI/12) * (kl*kl*kl) * erfcx( kl * np.sqrt(x*(m-n)/6) )
        return DTYPE(res)

    #   shortcut for SCD check
    def SCD(self, alP=1, alM=1, x=1, kl=0):
        fac = 0.5 * np.sqrt(6*PI)
        return ( self.Q(alP,alM,x,kl) / fac )

    #   FULL FREE ENERGY - use list/tuple/array for optimization variables
    def Ftot(self, pmx, l=3.8, lB=7.12, cs=0.0, cp=6e-7, w2=0, w3=0, p=0, delta=1, Pint=0, \
                dipoleD2factor=False, F4factor=True, F4screen=False, F4p=None, kill_kappa=False, ignoreBases=False):
        (alP, alM, x) = pmx
        # option: force different value of dipole length in F4 (ion pair formation); necessary if p=0 elsewhere
        dipole_F4 = F4p if F4p else p
        kl = 0 if kill_kappa else self.kapl(alP,alM,cp,cs,l,lB)
        res = self.F1(alP,alM)
        res += self.F2(alP,alM,cp,cs,l)
        res += self.F3(kl,cp,l)
        res += self.F4(alP,alM,l,lB,dipole_F4,delta,kl,F4factor,F4screen)
        res += self.F5(alP,alM,x,l,lB,p,delta,kl,w2,w3,Pint,dipoleD2factor,ignoreBases)
        return DTYPE(res)

    #   testing function - set some parameters, get some total Free Energy values
    def test(self, pmx_list=((1,1,1),(0.5,0.5,1)), seqinfo=True, showseq=False, parinfo=True, customFunc=None):
        if seqinfo:
            self.seq.info(showSeq=showseq)
        if parinfo:
            self.parInfo()
        for pmx in pmx_list:
            print(f"\n\twith (alP,alM,x) = {pmx}")
            print(f"\t-> gives: Ftot = {self.Ftot(pmx, **self.pars):2.5g}")
            if customFunc:
                print(f"\t-> customFunc = {customFunc(pmx):.5g}")
            print("")
        return

    #   minimize total Free Energy for given set of parameters (should be set prior to calling this)
    def minF(self, method="NM-TNC", alBounds=(1e-6,1.0), xBounds=(1e-3,20), \
                    ref=(0.5,0.5,1.0), init=(0.70,0.65,0.3), init_2=(0.10,0.15,1.10), \
                    showPars=("lB","cs","delta"), SCALE=1, TOL=1e-8, messages=False):
        # > possible option for global minimization: consider using 'direct' minimizer

        pdict = self.pars.copy()    # safe copy of model parameter dictionary

        var_bounds = (alBounds, alBounds, xBounds)          # variable bounds
        print_fmt = "(alP,alM,x) = ({P:1.5g}, {M:1.5g}, {X:1.5g})"   # format for printing results
        # handle choice of printing selected parameters
        if len(showPars) > 0:
            pinfo = "for (" + (", ".join(showPars)) + ") = ("
            pinfo += (", ".join([f"{pdict[p]:1.5g}" for p in showPars])) + ") "
        else:
            pinfo = ""
        # intro message
        if messages:
            print(f"\nOPTIMIZING {pinfo}...")
        t1 = perf_counter()     # begin timer
        Fref = self.Ftot(ref, **pdict)     # reference value of Free Energy
        minfunc = lambda trip: ( ( self.Ftot(trip, **pdict) - Fref ) * SCALE )
        if method.lower() == "nelder-mead":
            result = minimize(minfunc, init, \
                        method="Nelder-Mead", bounds=var_bounds, \
                        options={"maxiter":30000, "xatol":TOL, "fatol":TOL, "adaptive":True})
        elif method.lower() == "nm-tnc":    # hybrid method: Nelder-Mead to find basin, TNC to hone in
            if messages:
                print("\t entering Nelder-Mead algorithm ...\n")
            res0 = minimize(minfunc, init, \
                        method="Nelder-Mead", bounds=var_bounds, \
                        options={"maxiter":30000, "xatol":TOL, "fatol":TOL, "adaptive":True})
            new0 = res0.x       # N-M solution used as 'seed' (initial pt.) for TNC
            if messages:
                print("\t passing result " + (print_fmt.format(*tuple(new0))) + " to TNC for refinement ...\n")
            res1x, res1f = self.minF(method="TNC", alBounds=alBounds, xBounds=xBounds, \
                ref=ref, init=new0, showPars=showPars, SCALE=SCALE, TOL=TOL, messages=messages)
            # construct representative 'result' object
            result = res0
            result.x = res1x
            result.fun = res1f
        elif method.lower() == "nm-tnc-2":  # two-step approach with hybrid method - compare basins explicitly
            if init_2:
                init1 = init
                init2 = init_2
            else:
                print(" # second initial point was unspecified... ")
                return
            res1x, res1f = self.minF(method="NM-TNC", alBounds=alBounds, xBounds=xBounds, \
                ref=ref, init=init1, showPars=showPars, SCALE=SCALE, TOL=TOL, messages=messages)
            res2x, res2f = self.minF(method="NM-TNC", alBounds=alBounds, xBounds=xBounds, \
                ref=ref, init=init2, showPars=showPars, SCALE=SCALE, TOL=TOL, messages=messages)
#            FE1 = self.Ftot(res1, **pdict)
#            FE2 = self.Ftot(res2, **pdict)
            FE1, FE2 = res1f, res2f
            real_res = (res1x,res1f) if (FE1 < FE2) else (res2x,res2f)
            if messages:
                print("[FE1 = {FE1:2.5g}, FE2 = {FE2:2.5g}]")
                print("  >> FINAL CHOICE : " + print_fmt.format(*real_res) + "\n")
            return real_res
        elif method.lower() == "tnc":
            result = minimize(minfunc, init, \
                        method="tnc", bounds=var_bounds, options={"xtol":TOL, "ftol":TOL, "gtol":TOL, "maxiter":30000})
        else:
            print("\n\nERROR: given method '{method}' is unsupported.\n\n")
            return
        t2 = perf_counter()     # end timer
        # results messages (optional)
        if messages:
            print(f"\nDONE - elapsed time:\t{(t2-t1):2.5f}")
            print(f"\n[Extra Message: '{result.message}']")
            if result.success:
                print("\n\tSUCCESSFULLY found:\t" + print_fmt.format(*tuple(result.x)) + "\n")
            else:
                print("\n\t** FAILED to find minimum;\t" + print_fmt(*tuple(result.x)) + "\n")
        return tuple(result.x), result.fun

    #   minimize FE repeatedly for some varying parameter (pass 'minArgs' as keyword-arg dictionary to 'minF')
    def multiMin(self, multiPar="cs", parVals=[0], minArgs={}, seedNext=False):
        pdict = self.pars       # alias for actual parameter dictionary, to modify underlying parameters
        opar = pdict[multiPar]  # storing original parameter value for reference, to re-set
        # all keyword arguments to 'minF' function
        args = {}
        args.update(minArgs)
        # prepare result list
        pmxlist = []
        flist = []
        for pval in parVals:
#            print(f"ARGS passed to 'minF' function :  {args}\n")
            pdict.update({multiPar:pval})
            # adjust scale of FE for minimization, if not specified - increases robustness of minimization
            if "SCALE" not in args:
                scale = abs(self.SCD())
                if multiPar == "lB":
                    scale *= pval
                args.update({"SCALE":scale})
            pres, fres = self.minF(**args)
            pmxlist.append(pres)
            flist.append(fres)
            # adjust initial point with current result, if using 'seedNext' mode
            if seedNext:
                args.update( {"init":pres} )
        # re-set chosen parameter, returning to original setting
        pdict.update({multiPar:opar})
        return {multiPar:np.asarray(parVals), 'pmx':np.asarray(pmxlist), 'f':np.asarray(flist)}

    #   minimize repeatedly while varying Temperature, entering through one or more model parameters (e.g. lB & w2)
    def multiMin_T(self, Tvals=[273], Tfuncs={'lB':(lambda T:1)}, minArgs={}, seedNext=False):
        pdict = self.pars
        opars = [pdict[tp] for tp in Tfuncs]
        # all keyword arguments to 'minF' function
        args = {}
        args.update(minArgs)
        # prepare result list
        pmxlist = []
        flist = []
        for t in Tvals:
#            print(f"ARGS passed to 'minF' function :  {args}\n")
            pdict.update({p:Tfuncs[p](t) for p in Tfuncs})
            if "SCALE" not in args:
                scale = abs(self.SCD())
                if "lB" in Tfuncs:
                    scale *= Tfuncs["lB"](t)
                args.update({"SCALE":scale})
            pres, fres = self.minF(**args)
            pmxlist.append(pres)
            flist.append(fres)
            if seedNext:
                args.update( {"init":pres} )
        # re-set parameters affected by T, returning to original setting
        pdict.update({tp:opars[tp] for tp in Tfuncs})
        return {'T':np.asarray(Tvals), 'funcs':list(Tfuncs.keys()), 'pmx':np.asarray(pmxlist), 'f':np.asarray(flist)}

    #   solve for W2 _numerically_ : calibrate datapoint 'x' at a particular salt value 'cs'  [same units as in 'pars']
    def findW2(self, x_calib, cs_calib, x_thr=1e-8, diff_metric=(lambda x,y: (y-x)), notes=False, \
            minArgs={"method":"NM-TNC", "xBounds":(1e-3, 20), "ref":(0.5,0.5,1.5), "init":(0.8,0.8,2.5)}):
        # grab parameters as they are, and enable updating
        pdict = self.pars
        # ensure use of given salt value
        pdict.update( {"cs":cs_calib} )
        # given 'w2' used as seed / initial point
        w2_init = pars["w2"]
        # make function of 'w2' only, which will be numerically solved for zero by repeated optimization
        def w2Func(w2):
            # update parameter set first
            pdict.update( {"w2":w2} )
            # optimize at that point
            (alP, alM, x), f = self.minF(**minArgs)
            # check difference (from some metric)
            return diff_metric(x, x_calib)
        if notes:
            t1 = perf_counter()
            print("\nFinding 'w2' by iterating full optimization...")
        # now use SciPy to iterate -> find 'w2' at which the given metric returns zero
#        res = root_scalar(w2Func, x0=w2_init, x1=-w2_init/2, xtol=x_thr, rtol=x_thr, method="secant")
        res = root_scalar(w2Func, x0=w2_init, bracket=(-10,10), xtol=x_thr, method="brentq")
        w2_res = res.root
        if notes:
            t2 = perf_counter()
            print(f"DONE :\tw2={w2_res:2.4f}   (elapsed time: {(t2-t1):2.3f})")
            print(f"converged\t'{res.converged:}'\nflag\t'{res.flag:}'")
        return w2_res

    #   solve for 'delta' and 'w2' _simultaneously_ : calibrate using datapoints (cs_dlt, x_dlt) & (cs_w2, x_w2)
    def findDandW2(self, x_calibs=[], cs_calibs=[], thr=1e-6,  diff_metric=(lambda x,y: (y-x)), notes=False, \
            minArgs={"method":"NM-TNC", "xBounds":(1e-3, 20), "ref":(0.8,0.8,1.5), "init":(0.7,0.7,2.5)}):
        # grab parameters as they are (update with 'setPars' separately)
        pars1 = self.pars.copy()
        # copy parameters for second calibration point (simultaneous)
        pars2 = pars1.copy()
        # ensure use of given salt values
        pars1.update( {"cs":cs_calibs[0]} )
        pars2.update( {"cs":cs_calibs[1]} )
        # given 'delta' and 'w2' used as seed / initial point
        dlt_init = pars1["delta"]
        w2_init = pars1["w2"]
        # make _vector_ function of 'delta' and 'w2', which will be numerically solved for zero by repeated optimization
        def dw2Func(vec):
            (d, w2) = vec       # unpack single vector argument
            pars1.update( {"delta":d, "w2":w2} )
            self.setPars(pars1)
            (alP1, alM1, x1), f1 = self.minF(**minArgs)
            pars2.update( {"delta":d, "w2":w2} )
            self.setPars(pars2)
            (alP2, alM2, x2), f2 = self.minF(**minArgs)
            # check difference (from some metric)
            return [diff_metric(x1, x_calibs[0]), diff_metric(x2, x_calibs[1])]
        if notes:
            t1 = perf_counter()
            print("\nFinding 'delta' & 'w2' by iterating full optimization...")
        # now use SciPy to iterate -> find pair ('delta', 'w2') at which the given metric returns zero _vector_
        res = root(dw2Func, (dlt_init, w2_init), method="df-sane", tol=thr)
        (d_res, w2_res) = tuple(res.x)
        if notes:
            t2 = perf_counter()
            print(f"DONE :\tdlt={d_res:2.4f} ,  w2={w2_res:2.4f}   (elapsed time: {(t2-t1):2.3f})")
        # return as _dictionary_
        return {"delta":d_res, "w2":w2_res}

