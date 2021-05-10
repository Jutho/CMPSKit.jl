using Revise
using OptimKit
using CMPSKit
using LinearAlgebra
using GRUtils
# using JLD2

μ = 20
c = 1

H = ∫(∂ψ'*∂ψ - μ * ψ'*ψ + c * (ψ')^2*ψ^2, (0,1))

D = 16
T = Float64
grid = 0:0.5:1
Qlist = [randn(T, (D,D)) for _ in grid]
Rlist = [randn(T, (D,D)) for _ in grid]
Rlist[1] = zero(Rlist[1])
Rlist[end] = zero(Rlist[end])
Q = PiecewiseLinear(grid, Qlist)
R = PiecewiseLinear(grid, Rlist)
newgrid = 0:0.01:1
Q = PiecewiseLinear(newgrid, Q.(newgrid))
R = PiecewiseLinear(newgrid, R.(newgrid))
vR = vL = setindex!(zeros(T, D), 1, 1)
Ψ = FiniteCMPS(Q, R, vL, vR)

# alg = GradientDescent(; verbosity = 2, maxiter = 1000, gradtol = 1e-5)
# alg = ConjugateGradient(; verbosity = 2, maxiter = 1000, gradtol = 1e-5)
alg = LBFGS(30; verbosity = 2, maxiter = 1000, gradtol = 1e-5)

function plotresults(Ψ, ρL, ρR, e, E; plotgrid = 0:1e-4:1)
    fig = Figure()
    subplot(1, 2, 1)
    ekin = expval(∂ψ'*∂ψ, Ψ, ρL, ρR)
    plot(plotgrid, e.(plotgrid), plotgrid, ekin.(plotgrid))
    legend("e", "ekin")
    subplot(1, 2, 2)
    density = expval(ψ'*ψ, Ψ, ρL, ρR)
    plot(plotgrid, density.(plotgrid))
    legend("density")
    display(fig)
end

Ψ, ρL, ρR, E, e, normgrad, numfg, history =
    groundstate(H, Ψ, alg; optimnodes = 0:0.1:1, plot = plotresults);

N̂ = ∫(ψ'*ψ, (0,1))
N = expval(N̂, Ψ, ρL, ρR)
R = Ψ.Rs[1]
ΔN² = 2*integrate(localdot(leftenv(N̂, (Ψ,ρL,ρR))[1], R*ρR*R')) + N - N^2

# gs2 = groundstate2(H, Ψ, alg; optimnodes = 0:0.1:1, plot = (Ψ, ρL, ρR, e, E)->begin
# p = plot(plotgrid, e.(plotgrid))
# display(p)
# end);
