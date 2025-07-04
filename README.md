# Efficient Finite Initialization with Partial Norms for Tensorized Neural Networks and Tensor Networks Algorithms

This repository contains the implementation and supplementary material for the paper:

> **Efficient Finite Initialization with Partial Norms for Tensorized Neural Networks and Tensor Networks Algorithms**
> Alejandro Mata Ali, Iñigo Perez Delgado, Marina Ristol Roura, Aitor Moreno Fdez. de Leceta
> [arXiv:2309.06577](https://arxiv.org/abs/2309.06577)

## Abstract

We present two algorithms to initialize layers of tensorized neural networks and general tensor network algorithms using partial computations of their Frobenius norms and lineal entrywise norms, depending on the type of tensor network involved. The core of this method is the use of the norm of subnetworks of the tensor network in an iterative way, so that we normalize by the finite values of the norms that led to the divergence or zero norm. In addition, the method benefits from the reuse of intermediate calculations. We have also applied it to the Matrix Product State/Tensor Train (MPS/TT) and Matrix Product Operator/Tensor Train Matrix (MPO/TT-M) layers and have seen its scaling versus the number of nodes, bond dimension, and physical dimension. All code is publicly available.

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
pip install numpy tensorkrowch torch
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
