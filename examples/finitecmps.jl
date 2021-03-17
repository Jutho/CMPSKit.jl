using Revise
using OptimKit
using CMPSKit
using LinearAlgebra
# using JLD2

H = ∫(∂ψ'*∂ψ - 1 * ψ'*ψ + 1 * (ψ')^2*ψ^2, (0,1))

D = 6
grid = 0:0.5:1
Qlist = [randn(D,D), collect(Symmetric(randn(D,D)))]
push!(Qlist, collect(transpose(Qlist[1])))
Q₀ = PiecewiseLinear(grid, Qlist)
R₀ = PiecewiseLinear(grid, [zeros(D,D), collect(Symmetric(randn(D,D))), zeros(D,D)])
vL₀ = randn(D)
vR₀ = copy(vL₀)
newgrid = 0:0.1:1
Q = PiecewiseLinear(newgrid, Q₀.(newgrid))
R = PiecewiseLinear(newgrid, R₀.(newgrid))
vL = vL₀
vR = vR₀
Ψ = FiniteCMPS(Q, R, vL, vR)

# alg = LBFGS(100; verbosity = 2, maxiter = 1000, gradtol = 1e-2)
# (ψ, ρL, ρR), E, = optimize_fixed_boundaries(H, ψ₀, alg)
