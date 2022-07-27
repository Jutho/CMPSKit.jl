using CMPSKit
using Test
using LinearAlgebra
using Random
using OptimKit: GradientDescent, ConjugateGradient, LBFGS
using KrylovKit: GMRES, Arnoldi
Random.seed!(111134)

include("functionseries.jl")
include("exp.jl")
include("uniformcmps.jl")
include("periodiccmps.jl")
