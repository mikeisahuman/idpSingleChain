##  Mike Phillips, 2/8/2024
##  Very simple file for calculating two- and three-body terms,
##  i.e. 'Omega' and 'B', at arbitrary N (and saving)
##
##  Note: large .npy file saved by appending under 'with' environment
##      > when loading, must use a similar 'with' (and option 'rb' in 'open'), and call 'np.load' within loop

import numpy as np
from time import perf_counter

Nlist = range(5,101)

outfile = None
#outfile = "./out files/OB calc/OBtest_arr.npy"


#   quick function to calculate 2- and 3-body terms, Onon and B : MEMORY INTENSIVE when N is large
def fast_OB(N):
    rng = np.arange(N)
    l,m,n = np.meshgrid(rng, rng, rng)
    Odif = m[:,0,:] - n[:,0,:]
    Opow = np.power(Odif[Odif>0], -0.5)
    Ores =  Opow.sum() / N
    Bdif = l - n
    Bfac = (l - m) * (m - n)
    Bcheck = (Bdif > 0) & (Bfac > 0)
    Bpow = Bdif[Bcheck] * np.power(Bfac[Bcheck], -1.5)
    Bres =  Bpow.sum() / N
    return Ores, Bres

#   slow version function to calculate 2- and 3-body terms, Onon and B
def slow_OB(N, timeit=False):
    t1 = perf_counter()
    Ores = 1
    Bres = 0
    for l in range(2,N):
        Ores += np.power(l, -0.5)
        for m in range(1,l):
            Ores += np.power(l-m, -0.5)
            for n in range(0,m):
                Bres += (l-n) * np.power((l-m)*(m-n), -1.5)
    Ores = Ores / N
    Bres = Bres / N
    t2 = perf_counter()
    if timeit:
        print(f"\n > time for N={N:} :  {t2-t1:}\n")
    return N, Ores, Bres


if outfile:
    with open(outfile, 'w+b') as wf:
        for N in Nlist:
            print(f"\nCalculating (Omega, B)  for  N={N:} ...")
            np.save(wf, np.asarray(slow_OB(N)))
        print("\nDONE with all N !")

