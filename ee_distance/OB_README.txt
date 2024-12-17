Two- and Three-body terms 'Omega' and 'B'

Calculated from double/triple summation, divided by length N.


Saved as 'OBarr_{n1}-{n2}.npy' where n1,n2 indicate length range (inclusive).
	-> was saved under 'with' environment


Loading: must use 'with' environment
	-> i.e.
	OBlist = []
	with open('OBarr_{n1}-{n2}.npy', 'rb') as f:
		for N in range(n1,n2+1):
			OBlist.append( np.load(f) )


Alternative: formatted list obtained as above is saved separately as 'OBfmt_{n1}-{n2}.npy'

Each entry in formatted list is a triplet with (N, Omega, B).
	-> to retrieve particular entry N, use 'np.where', i.e.
	index = np.where(OBfmt[:,0]==N)[0][0]
	Omega, B = OBfmt[index,1:]
