# idpSingleChain

Tools for analyzing isolated protein sequences.

---

## Intrinsically Disordered Protein (IDP) sequence-specific physics theory for a Single Chain

### Consists of two main sectors
1. Mean-square end-to-end distance $\langle R_{ee}^2 \rangle$  
   i.e. overall estimation of chain compaction / swelling (conformation) as a function of sequence and conditions (salt, temperature, pH)
2. Full distance map between any two amino acids / reisdues in the chain $\langle R_{ij}^2 \rangle$  
   i.e. detailed estimation of chain compaction / swelling within each section of the chain, as a function of sequence and conditions (salt, temperature, pH)

### Each has two models of underlying physics
* 'xModel' : Full Ionization model
   - simple assignment of charge +1 / -1 / 0 for each amino acid in the chain, according to their properties
* 'doiModel' : Counter-ion Condensation / degree of ionization / beyond monopole theory
   - ionization of amino acids is determined self-consistently with conformation, accounting for dipole interactions

---

#### Based upon decades of work in polymer / protein physics:

* Doi & Edwards, Theory of Polymer Dynamics (1988)
* Muthukumar, Theory of counter-ion condensation on flexible polyelectrolytes: Adsorption mechanism (2004), https://doi.org/10.1063/1.1701839
* Firman & Ghosh, Sequence charge decoration dictates coil-globule transition in intrinsically disordered proteins (2018), https://doi.org/10.1063/1.5005821
* Huihui & Firman & Ghosh, Modulating charge patterning and ionic strength as a strategy to induce conformational changes in intrinsically disordered proteins (2018), https://doi.org/10.1063/1.5037727
* Huihui & Ghosh, An analytical theory to describe sequence-specific inter-residue distance profiles for polyampholytes and intrinsically disordered proteins (2020), https://doi.org/10.1063/5.0004619
* Phillips & Muthukumar & Ghosh, Beyond monopole electrostatics in regulating conformations of intrinsically disordered proteins (2024), https://doi.org/10.1093/pnasnexus/pgae367
* Houston & Phillips & Torres & Gaalswyk & Ghosh, Physics-Based Machine Learning Trains Hamiltonians and Decodes the Sequence–Conformation Relation in the Disordered Proteome (2024), https://doi.org/10.1021/acs.jctc.4c01114
