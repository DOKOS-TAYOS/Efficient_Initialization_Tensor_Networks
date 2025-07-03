# Efficient Finite Initialization for Tensorized Neural Networks

This repository contains the implementation and supplementary material for the paper:

> **Efficient Finite Initialization for Tensorized Neural Networks**
> Alejandro Mata Ali, Iñigo Perez Delgado, Marina Ristol Roura, Aitor Moreno Fdez. de Leceta
> [arXiv:2309.06577](https://arxiv.org/abs/2309.06577)

## Abstract

We present a novel method for initializing layers of tensorized neural networks in a way that avoids the explosion or vanishing of the parameters of the matrix they emulate. The method is intended for layers with a high number of nodes, we do not desire to store/calculate all the elements of the represented layer, and it follows a smooth distribution. This method is equally applicable to normalize general tensor networks in which we want to avoid overflows.

The core of this method is the use of the Frobenius norm and the partial lineal entrywise norm of reduced forms of the layer in an iterative partial form, so that it has to be finite and within a certain range. These norms are efficient to compute, fully or partially, for most cases of interest. In addition, the method benefits from the reuse of intermediate calculations. We apply the method to different layers and check its performance. All code is publicly available.

## Contents

* `Efficient_Finite_Initialization_for_Tensorized_Neural_Networks.pdf`: Preprint of the published paper.
* `TN_Normalizer.ipynb`: Jupyter notebook with a full implementation of the proposed initialization methods.
* `figures/`: Directory containing all illustrative figures from the paper.

## Main Features

* Frobenius-norm-based iterative normalization for general tensor networks.
* Entrywise-norm-based normalization for positive tensor networks.
* Compatible with arbitrarily large TNN layers (memory-safe).
* Designed to work with PyTorch layers or stand-alone tensor networks.
* Includes experiments showing scalability with number of nodes, physical and bond dimensions.

## Installation & Requirements

Install required Python packages:

```bash
pip install numpy torch
```

Then, open and run the notebook:

```bash
jupyter notebook TN_Normalizer.ipynb
```

## Usage

The notebook provides modular functions to initialize and normalize tensor networks. You can use the `normalize_tensor_network()` function to apply the Frobenius or linear norm-based algorithm depending on your network type.

## Citation

If you use this work, please cite:

```bibtex
@article{mata2023efficient,
  title={Efficient Finite Initialization for Tensorized Neural Networks},
  author={Mata Ali, Alejandro and Perez Delgado, I\~nigo and Ristol Roura, Marina and Moreno Fdez. de Leceta, Aitor},
  journal={arXiv preprint arXiv:2309.06577},
  year={2023}
}
```

## License


This repository is released under the MIT License.

## Contact

For questions or collaborations, contact:
**Alejandro Mata Ali**
`alejandro.mata.ali@gmail.com`

---
