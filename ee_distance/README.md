# Object Classes and Scripts for End-to-End Distance
i.e. getting compaction / swelling factor $x$ from theory related to mean-square end-to-end distance, $\langle R_{ee}^2 \rangle = N\,l\,b\,x$

## One sequence analyzer / wrapping
'Sequence' will load and report basic information for a selected IDP sequence, e.g. referring to a CSV file

## Two physical models
1. Full Ionization model 'xModel'
2. Counter-ion Condensation / Degree of Ionization / Beyond Monopole model 'doiModel'

## Scripts
* 'titration' scripts determine end-to-end distance across a range of conditions, like salt or pH, with others kept fixed
* 'many' scripts calculate the same quantity under fixed conditions for a large number of sequences, e.g. the 28058 that comprise the human IDRome
