using Revise
using CMPSKit
using KrylovKit
using OptimKit
using LinearAlgebra
using JLD2
using Random

χ = 8
k = 1.
μ = μ1 = μ2 = 1.0
c = c1 = c2 = 1.0
A1 = A2 = 0.0
c12 = -0.0
δμ = 0.00
δA = 0.00
sign1 = 1.0
sign2 = -1.0

E2s = []
Ψ2s = []
histories2 = [] 

M = Constant(rand(ComplexF64,χ,χ))
D1 = Constant(diagm(rand(ComplexF64,χ)/χ))
D2 = Constant(diagm(rand(ComplexF64,χ)/χ))
KL = Constant(rand(ComplexF64,χ,χ))

KL = 0.5*(KL-KL')
R1 = M*D1*inv(M)
R2 = M*D2*inv(M)
Ds = (D1,D2)
RLs = (R1,R2)
QL = KL - 0.5*R1'*R1 - 0.5*R2'*R2

Ψ = InfiniteCMPS(QL, (R1,R2); gauge = :left)

d1 = ∂ψ[1] - im * (A1 + sign1 * δA) * ψ[1]
d2 = ∂ψ[2] - im * (A2 + sign2 * δA) * ψ[2]

h = k * (d1' * d1 + d2' * d2) - (μ1 + sign1 * δμ) * ψ[1]'*ψ[1] - (μ2 + sign2 * δμ) * ψ[2]'*ψ[2] + c1 * (ψ[1]')^2*ψ[1]^2 + c2 * (ψ[2]')^2*ψ[2]^2 + c12 * (ψ[1]'*ψ[2]'*ψ[2]*ψ[1] + ψ[2]'*ψ[1]'*ψ[1]*ψ[2])
H = ∫(h, (-Inf,+Inf))

alg_prep = LBFGS(; verbosity = 4, maxiter = 1000, gradtol = 1e-10)
alg1 = LBFGS(; verbosity = 4, maxiter = 40000, gradtol = 1e-5);
alg2 = LBFGS(; verbosity = 4, maxiter = 20000, gradtol = 1e-6);
linalg = GMRES(krylovdim = 80);

function expand(Ψ::InfiniteCMPS; ϵ = 0.05)
    χ = size(Ψ.Rs[1][],1)
    χ_new = 2 * χ

    I2 = Diagonal(ones(Float64, 2))
    M = eigvecs(Ψ.Rs[1][])
    D1 = diagm(diag((inv(M) * Ψ.Rs[1][] * M)))
    D2 = diagm(diag((inv(M) * Ψ.Rs[2][] * M)))
    Ds = (D1,D2)

    Ds_new = map(1:length(Ψ.Rs)) do ix
        kron(I2,Ds[ix]) 
    end
    Q_new = kron(I2, Ψ.Q[]) 
    M_new = kron(I2, M)
    Rs_new = [M_new * D * inv(M_new) for D in Ds_new]

    pert = rand(ComplexF64, χ_new, χ_new)
    pert = (pert + pert') / norm(pert + pert')

    M_new = M_new * exp(ϵ * pert)
    Ds_new = map(Ds_new) do D
        dD = Diagonal(rand(ComplexF64, χ_new))
        dD = (dD + dD') / norm(dD + dD')
        D + ϵ * dD
    end

    ΔRs = [M_new * D * inv(M_new) - R0 for (D, R0) in zip(Ds_new, Rs_new)]
    V = sum(-[R0' * ΔR + 0.5 * ΔR' * ΔR for (R0, ΔR) in zip(Rs_new, ΔRs)])
    Q_new = Q_new + V

    return InfiniteCMPS(Constant(Q_new), (Constant(Rs_new[1]), Constant(Rs_new[2])); gauge = :left), M_new, (Ds_new[1], Ds_new[2])
end

D1 = Constant(diagm(diag((inv(M[]) * Ψ.Rs[1][] * M[]))))
D2 = Constant(diagm(diag((inv(M[]) * Ψ.Rs[2][] * M[]))))
Q = Constant(((inv(M[]) * Ψ.Q[] * M[])))
Ψ = InfiniteCMPS(Q, (D1,D2))
Ψ, ρL, ρR, E, e, normgrad, numfg, history = groundstate_diagonal(H, Ψ; optalg = alg_prep, linalg = linalg)

Ψ_new = leftgauge(Ψ)[1]

Ψ, ρR, E, e, normgrad, numfg, history = groundstate_MDMinv(H, Ψ_new; optalg = alg1, linalg = linalg)

@show E
@show expval(h,Ψ)[]
@show expval(ψ[1]*ψ[2] - ψ[2]*ψ[1],Ψ)[]
@show expval(ψ[1]'*ψ[1] + ψ[2]'*ψ[2],Ψ)[]

@save "data_mu=$(μ)_c=$(c)_c12=$(c12)_D=8" Ψ ρR E e normgrad numfg history

Ψ_new, M_new, Ds_new = expand(Ψ)
D1 = Constant(diagm(diag((inv(M_new) * Ψ_new.Rs[1][] * M_new))))
D2 = Constant(diagm(diag((inv(M_new) * Ψ_new.Rs[2][] * M_new))))
Q = Constant(((inv(M_new) * Ψ_new.Q[] * M_new)))
Ψ_new = InfiniteCMPS(Q, (D1,D2))
Ψ, ρL, ρR, E, e, normgrad, numfg, history = groundstate_diagonal(H, Ψ_new; optalg = alg_prep, linalg = linalg)

Ψ_new = leftgauge(Ψ)[1]
Ψ, ρR, E, e, normgrad, numfg, history = groundstate_MDMinv(H, Ψ_new; optalg = alg2, linalg = linalg)
@show E
@show expval(ψ[1]*ψ[2] - ψ[2]*ψ[1],Ψ)[]
@show expval(ψ[1]'*ψ[1] + ψ[2]'*ψ[2],Ψ)[]

@save "data_mu=$(μ)_c=$(c)_c12=$(c12)_D=16" Ψ ρR E e normgrad numfg history

Ψ_new, M_new, Ds_new = expand(Ψ)
D1 = Constant(diagm(diag((inv(M_new) * Ψ_new.Rs[1][] * M_new))))
D2 = Constant(diagm(diag((inv(M_new) * Ψ_new.Rs[2][] * M_new))))
Q = Constant(((inv(M_new) * Ψ_new.Q[] * M_new)))
Ψ_new = InfiniteCMPS(Q, (D1,D2))
Ψ, ρL, ρR, E, e, normgrad, numfg, history = groundstate_diagonal(H, Ψ_new; optalg = alg_prep, linalg = linalg)

Ψ_new = leftgauge(Ψ)[1]
Ψ, ρR, E, e, normgrad, numfg, history = groundstate_MDMinv(H, Ψ_new; optalg = alg2, linalg = linalg)
@show E
@show expval(ψ[1]*ψ[2] - ψ[2]*ψ[1],Ψ)[]
@show expval(ψ[1]'*ψ[1] + ψ[2]'*ψ[2],Ψ)[]

@save "data_mu=$(μ)_c=$(c)_c12=$(c12)_D=32" Ψ ρR E e normgrad numfg history

Ψ_new, M_new, Ds_new = expand(Ψ)
D1 = Constant(diagm(diag((inv(M_new) * Ψ_new.Rs[1][] * M_new))))
D2 = Constant(diagm(diag((inv(M_new) * Ψ_new.Rs[2][] * M_new))))
Q = Constant(((inv(M_new) * Ψ_new.Q[] * M_new)))
Ψ_new = InfiniteCMPS(Q, (D1,D2))
Ψ, ρL, ρR, E, e, normgrad, numfg, history = groundstate_diagonal(H, Ψ_new; optalg = alg_prep, linalg = linalg)

Ψ_new = leftgauge(Ψ)[1]
Ψ, ρR, E, e, normgrad, numfg, history = groundstate_MDMinv(H, Ψ_new; optalg = alg2, linalg = linalg)
@show E
@show expval(ψ[1]*ψ[2] - ψ[2]*ψ[1],Ψ)[]
@show expval(ψ[1]'*ψ[1] + ψ[2]'*ψ[2],Ψ)[]

@save "data_mu=$(μ)_c=$(c)_c12=$(c12)_D=64" Ψ ρR E e normgrad numfg history