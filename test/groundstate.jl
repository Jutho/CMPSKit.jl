using Revise
using CMPSKit
using KrylovKit
using OptimKit
using LinearAlgebra
using JLD2

D = 16

Q = Constant(randn(D,D))
R1 = Constant(randn(D,D))
R2 = Constant(randn(D,D))
Ψ = InfiniteCMPS(Q, (R1, R2))

alg = ConjugateGradient(; verbosity = 2, maxiter = 10^6, gradtol = 1e-4);

H = ∫(∂ψ[1]'*∂ψ[1] - 1 * ψ[1]'*ψ[1] + 1 * (ψ[1]')^2*ψ[1]^2 +
        ∂ψ[2]'*∂ψ[2] - 1 * ψ[2]'*ψ[2] + 1 * (ψ[2]')^2*ψ[2]^2 +
        100 * (ψ[1]*ψ[2] - ψ[2]*ψ[1])' * (ψ[1]*ψ[2] - ψ[2]*ψ[1]), (-Inf,+Inf))

# function finalize!(x, f, g, numiter)
#     if mod(numiter, 1000) == 0
#         @save "tempoptresults2.jld" x
#     end
#     return x, f, g
# end
# groundstate(H, Ψ; optalg = alg, linalg = GMRES(; tol = 1e-4), finalize! = finalize!)
groundstate(H, Ψ; optalg = alg, linalg = GMRES(; tol = 1e-6))
