# Object Classes and Scripts for Inter-Residue Distance
i.e. getting compaction / swelling factor $x_{ij}$ between *any* pair of residues $i,j$ from theory related to mean-square inter-residue distance, $\langle R_{ij}^2 \rangle = |i-j| l b x_{ij}$

## One sequence analyzer / wrapping
'Sequence' will load and report basic information for a selected IDP sequence, e.g. referring to a CSV file

## Two physical models
1. Full Ionization model 'xModel_ij'
2. Counter-ion Condensation / Degree of Ionization / Beyond Monopole model 'doiModel_ij'

## Scripts
* 'titration' scripts determine inter-residue distance across a range of conditions, like salt or pH, with others kept fixed; for one or several $i,j$ pairs
* 'many' scripts calculate the same quantity under fixed conditions for a large number of sequences, e.g. the 28058 that comprise the human IDRome; for one or several $i,j$ pairs
* 'xij_map' script builds the entire inter-residue profile map, across all $i,j$ pairs, under fixed conditions
* 'decoration_matrices' script builds interaction matrices $SHDM$, $SCDM$, $SCDDM$, $SDDM$, $T_{ij}$ ; does not perform minimization for $x_{ij}$
* 'extract_w2ij' script calculates the necessary two-body interaction $\omega_2$ at each $i,j$ based on given $x_{ij}$ map, under fixed conditions

## Plotters
* 'xij_plotter_module' provides a suite of tools for loading, normalizing, and plotting $x_{ij}$ map(s) from previous calculations (or simulation)
* 'xij_plotter_script' uses tools from the module to make publication-quality plots of $x_{ij}$ map(s) in various viewpoints
* 'ijpairs_plotter' will plot results of specific $i,j$ pair(s) as a function of some variable, e.g. results from 'titration' scripts
