# DiffEqFlux: High Level Scientific Machine Learning (SciML) Pre-Built Architectures

DiffEqFlux.jl is a parameter estimation system for the SciML ecosystem. It is
a high level interface that pulls together all of the tools with heuristics
and helper functions to make solving inverse problems and inferring models
as easy as possible without losing efficiency.



## Applications

The approach of this package is the efficient training of
[Universal Differential Equations](https://arxiv.org/abs/2001.04385).
Since this is a fairly general class of problems, the following
applications are readily available as specific instances of this
methodology, and are showcased in tutorials and layer functions:

- Neural ODEs
- Neural SDEs
- Neural DAEs
- Neural DDEs
- Augmented Neural ODEs
- Graph Neural ODEs
- Hamiltonian Neural Networks (with specialized second order and symplectic integrators)
- Lagrangian Neural Networks
- Continuous Normalizing Flows (CNF) and FFJORD
- Galerkin Neural ODEs

## Citation

If you use DiffEqFlux.jl or are influenced by its ideas, please cite:

```
@article{rackauckas2020universal,
  title={Universal differential equations for scientific machine learning},
  author={Rackauckas, Christopher and Ma, Yingbo and Martensen, Julius and Warner, Collin and Zubov, Kirill and Supekar, Rohit and Skinner, Dominic and Ramadhan, Ali},
  journal={arXiv preprint arXiv:2001.04385},
  year={2020}
}
```
