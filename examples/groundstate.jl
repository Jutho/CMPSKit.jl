using Revise
using CMPSKit
using KrylovKit
using OptimKit
using LinearAlgebra
using JLD2


μ = (4.75π)^2
c = 1e6

H = ∫(∂ψ'*∂ψ - μ * ψ'*ψ + c * (ψ')^2*ψ^2, (-Inf,+Inf))

D = 8
optalg = LBFGS(30; verbosity = 2, maxiter = 10^3, gradtol = 1e-1);
linalg = GMRES(; krylovdim = D^2, tol = 1e-8);
eigalg = Arnoldi(; krylovdim = D^2, tol = 1e-8);

# Single-boson Lieb-Liniger
Q = Constant(randn(D,D)/D)
R = Constant(randn(D,D)/D)
Ψ = InfiniteCMPS(Q, R)

ΨL, ρR, E, e = groundstate(H, Ψ; optalg = optalg, linalg = linalg, eigalg = eigalg)

Q = ΨL.Q
R = ΨL.Rs[1]

# Two independent boson Lieb-Liniger
I = one(Q(0))
Q1 = Constant(kron(Q(0), I))
R1 = Constant(kron(R(0), I))
Q2 = Constant(kron(I, Q(0)))
R2 = Constant(kron(I, R(0)))

Ψ2 = InfiniteCMPS(Q1+Q2, (R1, R2))

H2 = ∫(∂ψ[1]'*∂ψ[1] - 1 * ψ[1]'*ψ[1] + 1 * (ψ[1]')^2*ψ[1]^2 +
        ∂ψ[2]'*∂ψ[2] - 1 * ψ[2]'*ψ[2] + 1 * (ψ[2]')^2*ψ[2]^2 +
        100 * (ψ[1]*ψ[2] - ψ[2]*ψ[1])' * (ψ[1]*ψ[2] - ψ[2]*ψ[1]), (-Inf,+Inf))

expval(∂ψ[1]'*∂ψ[1] - 1 * ψ[1]'*ψ[1] + 1 * (ψ[1]')^2*ψ[1]^2, Ψ2) ≈ e
expval(∂ψ[1]'*∂ψ[1] - 1 * ψ[1]'*ψ[1] + 1 * (ψ[1]')^2*ψ[1]^2, Ψ2) ≈
    expval(∂ψ[2]'*∂ψ[2] - 1 * ψ[2]'*ψ[2] + 1 * (ψ[2]')^2*ψ[2]^2, Ψ2)
expval(density(H2), Ψ2) ≈ 2*e

D = 16
Q = Constant(randn(D,D))
R1 = Constant(randn(D,D))
R2 = Constant(randn(D,D))
Ψ2 = InfiniteCMPS(Q, (R1, R2))
optalg = ConjugateGradient(; verbosity = 2, maxiter = 10^6, gradtol = 1e-7);
linalg = GMRES(; krylovdim = 256, tol = 1e-9);
eigalg = Arnoldi(; krylovdim = 256, tol = 1e-9);

ΨL2, ρR2, E2, e2 = groundstate(H2, Ψ2; optalg = optalg, linalg = linalg, eigalg = eigalg)

# Free fermion
Q = Constant(randn(D,D))
R = Constant(kron([0 1; 0 0], randn(D>>1,D>>1)))
ΨF = InfiniteCMPS(Q, R)

HF = ∫(∂ψ'*∂ψ - 1 * ψ'*ψ + 1000 * (ψ')^2*ψ^2, (-Inf,+Inf))
ΨL, ρR, E, e = groundstate(H, Ψ; optalg = optalg, linalg = linalg, eigalg = eigalg)
