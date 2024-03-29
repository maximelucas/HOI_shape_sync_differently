# Higher-order interactions shape collective dynamics differently in hypergraphs and simplicial complexes

[![DOI](https://zenodo.org/badge/466099254.svg)](https://zenodo.org/badge/latestdoi/466099254)

This repository contains the code used for the analysis presented in the paper:  
"[Higher-order interactions shape collective dynamics differently in hypergraphs and simplicial complexes][2],"  
by Yuanzhao Zhang*, Maxime Lucas*, and Federico Battiston  
*Nat. Commun.* **14**, 1605 (2023)

<img src="https://user-images.githubusercontent.com/7493360/220310428-c26eb720-1519-4a1d-acac-5228191db7c2.jpeg" width="45%">

### Contents
- `utils.py`: useful functions used in the notebooks.
- `*.ipynb`: the notebooks used to reproduce the results and figures.
- `data/`: the brain connectome data used in Figs. 5 to 7 (available from https://neurodata.io/project/connectomes/). 

### Dependencies

The code was tested for Python 3.9 and the dependencies specified in [requirements.txt](requirements.txt).

In particular, the code heavily relies on the [XGI library](https://github.com/ComplexGroupInteractions/xgi). Most of the functions in `utils.py` have now been integrated into XGI.

The code to compute the multiorder Laplacian is based on the [repository](https://github.com/maximelucas/multiorder_Laplacian) associated with the paper, "Multiorder Laplacian for synchronization in higher-order networks" (2020), by Lucas, M., Cencetti, G., & Battiston, F. [Phys. Rev. Res., 2(3)][1], 033410.

[1]: https://doi.org/10.1103/PhysRevResearch.2.033410
[2]: https://doi.org/10.1038/s41467-023-37190-9
