
# Learning Reconstructive Embeddings in Reproducing Kernel Hilbert Spaces via the Representer Theorem
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16812805.svg)](https://doi.org/10.5281/zenodo.16812805)
[![DOI](https://zenodo.org/badge/DOI/10.1109/OJCS.2026.3682462.svg)](https://doi.org/10.1109/OJCS.2026.3682462)


﻿﻿<img width="1536" height="1024" alt="imagen" src="https://github.com/user-attachments/assets/226c640d-52cc-4c73-829f-f4ad44d9492b" />


Motivated by the growing interest in representation learning approaches that uncover the latent structure of high-dimensional data, this project proposes new algorithms for reconstruction-based manifold learning within Reproducing-Kernel Hilbert Spaces (RKHS). Each observation is first reconstructed as a linear combination of the other samples in the RKHS, by optimizing a vector form of the Representer Theorem for their autorepresentation property. A separable operator-valued kernel extends the formulation to vector-valued data while retaining the simplicity of a single scalar similarity function. A subsequent kernel-alignment objective projects the data into a lower-dimensional latent space whose Gram matrix aims to match the high-dimensional reconstruction kernel, thus transferring the auto-reconstruction geometry of the RKHS to the embedding. Therefore, the proposed algorithms represent a principled approach to the autorepresentation property, exhibited by many natural data, by using and adapting well-known results of Kernel Learning Theory. Numerical experiments on both simulated (concentric circles and swiss-roll) and real (cancer molecular activity and IoT network intrusions) datasets provide empirical evidence of the practical effectiveness of the proposed approach.

This repository contains the supplementary code for the paper "Learning Reconstructive Embeddings in Reproducing Kernel Hilbert Spaces via the Representer Theorem".


## Citation

If you use this code in your research, please cite:

```bibtex
@article{feito_casares_2025a,
  author        = {Feito-Casares, Enrique and Melgarejo Meseguer, Francisco Manuel and Rojo-{\'A}lvarez, Jos{\'e} Luis},
  journal       = {IEEE Open Journal of the Computer Society},
  title         = {{Learning Reconstructive Embeddings in Reproducing Kernel Hilbert Spaces via the Representer Theorem }},
  year          = {2026},
  month         = apr,
  volume        = {7},
  ISSN          = {2644-1268},
  pages         = {659-669},
  doi           = {10.1109/OJCS.2026.3682462}
}

@online{feito_casares_2025b,
  title        = {Learning Reconstructive Embeddings in Reproducing Kernel Hilbert Spaces via the Representer Theorem (Supplementary Code)},
  author       = {Feito-Casares, Enrique and Melgarejo Meseguer, Francisco Manuel and Rojo-{\'A}lvarez, Jos{\'e} Luis},
  year         = {2025},
  month        = aug,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.16812805},
  url          = {https://doi.org/10.5281/zenodo.16812805}
}
```

## Project Structure
```
KernelEmbedding/
├── README.md                  # This file
├── Experiments_A_B_C.ipynb    # Experiments A: Concentric Circles, B: Swiss Roll, C: Cancer Biomolecules
├── Experiment_D.ipynb         # Experiment D: IoT Network Intrusion
├── requirements.txt           # Python dependencies
├── data/                      # Dataset storage directory
│   ├── swiss_roll.mat         # Synthetic Swiss Roll data
│   ├── CiCIOT/                # CIC-IoT-2023 dataset (network intrusions)
│   └── NCI-CANCER/            # NCI60 cancer biomolecule dataset
├── Experiments_A_B_C.ipynb    # Experiments A: Concentric Circles, B: Swiss Roll, C: Cancer Biomolecules
├── Experiment_D.ipynb         # Experiment D: IoT Network Intrusions (CIC-IoT-2023)
├── ke_toolbox/                # Kernel Embedding toolbox
│   ├── __init__.py
│   ├── dataset.py             # Data loading and preprocessing functions
│   ├── kernels.py             # Kernel function implementations and utilities
│   ├── main.py                # Main functions and optimization pipeline
│   ├── optimization.py        # RKHS reconstruction optimization algorithms
│   ├── requirements.txt       # Python dependencies
│   └── utils.py               # Synthetic data generation and device management
```
## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the experiments:
   - Open and run the notebooks `Experiments_A_B_C.ipynb` or `Experiment_D.ipynb` in Jupyter to reproduce the experiments from the paper.

## Data References
The following datasets were used in the experiments:

### NCI60 Dataset
The NCI60 dataset contains 4547 drug candidates with their cancer inhibition potentials in 60 cell line targets. Raw data provided by the NCI's Division of Cancer Treatment and Diagnosis (DCTD) offer a wide range of data for use by the scientific community.

**References:**
- Su, H., Heinonen, M., & Rousu, J. (2010, September). Structured output prediction of anti-cancer drug activity. In IAPR International Conference on Pattern Recognition in Bioinformatics (pp. 38-49). Springer Berlin Heidelberg. [http://link.springer.com/chapter/10.1007/978-3-642-16001-1_4](http://link.springer.com/chapter/10.1007/978-3-642-16001-1_4)
- Su, Hongyu; Rousu, Juho. Multilabel Classification through Random Graph Ensembles. Machine Learning, DOI: [https://doi.org/10.1007/s10994-014-5465-9](https://doi.org/10.1007/s10994-014-5465-9).

### CICIoT2023 Dataset
Citation: Neto, E.C.P.; Dadkhah, S.; Ferreira, R.; Zohourian, A.; Lu, R.; Ghorbani, A.A. CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment. Sensors 2023, 23, 5941.[https://doi.org/10.3390/s23135941](https://doi.org/10.3390/s23135941)


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Partially funded by the Autonomous Community of Madrid (ELLIS Madrid Node). Also partially supported by project PID2022-140786NB-C32 (LATENTIA) and PID2023-152331OA-I00 (HERMES) from the Spanish Ministry of Science and Innovation (AEI/10.13039/501100011033). This work was supported by the CyberFold project, funded by the European Union through the NextGenerationEU instrument (Recovery, Transformation, and Resilience Plan), and managed by Instituto Nacional de Ciberseguridad deEspaña (INCIBE), under reference number ETD202300129

<p align="center">
  <img src="https://github.com/user-attachments/assets/e8280cfd-ea9c-4bd6-af51-9d1c1c268279" alt="EU Funding" height="80">
</p>

With the collaboration of

<p align="center">
  <img src="https://github.com/user-attachments/assets/47ab46a5-977c-4b02-945b-3aac93912cec" alt="URJC Logo" height="50">
</p>
