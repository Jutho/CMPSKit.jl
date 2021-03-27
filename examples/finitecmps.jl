using Revise
using OptimKit
using CMPSKit
using LinearAlgebra
# using GRUtils
# using JLD2

H = ∫(∂ψ'*∂ψ - 100 * ψ'*ψ + 1 * (ψ')^2*ψ^2, (0,1))

D = 6
T = Float64
grid = 0:0.5:1
Qlist = [randn(T, (D,D)) for _ in grid]
Rlist = [randn(T, (D,D)) for _ in grid]
Rlist[1] = zero(Rlist[1])
Rlist[end] = zero(Rlist[end])
Q = PiecewiseLinear(grid, Qlist)
R = PiecewiseLinear(grid, Rlist)
newgrid = 0:0.2:1
Q = PiecewiseLinear(newgrid, Q.(newgrid))
R = PiecewiseLinear(newgrid, R.(newgrid))
vL = normalize!(randn(T, D))
vR = normalize!(randn(T, D))
Ψ = FiniteCMPS(Q, R, vL, vR)

alg = LBFGS(100; verbosity = 2, maxiter = 1000, gradtol = 1e-2)
groundstate(H, Ψ, alg; envnodes = 0:0.01:1)
