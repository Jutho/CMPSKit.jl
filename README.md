# CMPSKit

[![Build Status](https://github.com/Jutho/CMPSKit.jl/workflows/CI/badge.svg)](https://github.com/Jutho/CMPSKit.jl/actions)

This package is still work in progress, but is currently functional for running ground state simulations of one-dimensional non-relativistic quantum field theories using continuous matrix product states, both for infinite systems with homogeneous or periodic interactions, or for finite systems with inhomogeneous interactions.

This package is not yet registered; to install use:
```julia
import Pkg
Pkg.add("https://github.com/Jutho/CMPSKit.jl")
```

The use of this package is probably best illustrated in the notebook hosted at [cMPS-notebook](https://github.com/Jutho/cMPS-notebook), which illustrates how to use this package to reproduce the results from the paper "Variational optimization of continous matrix product states" by Benoît Tuybens, Jacopo de Nardis, Jutho Haegeman and Frank Verstraete, currently available as preprint [arXiv:2006.01801](https://arxiv.org/abs/2006.01801).
