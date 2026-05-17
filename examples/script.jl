using CMPSKit, Test, LinearAlgebra

D = 8 # bond dimension
T = Float64 # element type

g = 10.0
μ = 1.0

Ĥ = ∫(∂ψ̂' * ∂ψ̂ - μ * ψ̂' * ψ̂ + g * ψ̂' * ψ̂' * ψ̂ * ψ̂, (-Inf, +Inf))

# initial cMPS
Q, R = Constant.((randn(T, D, D), randn(T, D, D)))
Ψ = InfiniteCMPS(Q, R)

ΨL, = leftgauge(Ψ)
ΨR, C = rightgauge(ΨL)
ρR, = rightenv(ΨL)
QL, RLs = ΨL
C = sqrt(ρR)
QR = C \ (QL * C)
RRs = (C,) .\ (RLs .* (C,))
ΨR = InfiniteCMPS(QR, RRs; gauge=:r)
HL, EL, eL, hL, infoL = leftenv(Ĥ, (ΨL, one(ρR), ρR))
HR, ER, eR, hR, infoR = rightenv(Ĥ, (ΨR, adjoint(C) * C, one(ρR)))
gradC, gradQC, gradRCs = centergradient(Ĥ, (ΨL, ΨR, C), HL, HR)

@test gradQC ≈ -sum(gradRCs .* adjoint.(RRs)) - gradC * QR'
@test gradQC ≈ -sum(adjoint.(RLs) .* gradRCs) - QL' * gradC

gradQL, gradRLs = gradient(Ĥ, (ΨL, one(C), C * C'), HL, C * HR * C')
dRLs = (.-(RLs) .* Ref(gradQL) .+ gradRLs) ./ (ρR,)
dQL = -sum(adjoint.(RLs) .* dRLs)

@test dQL * C ≈ (gradQC - QL * gradC)
@test all(dRLs .* (C,) .≈ (gradRCs .- RLs .* (gradC,)))

# test that update matches
α = 1e-6
RLsnew = RLs .- α .* dRLs
K = QL + 1 / 2 * sum(adjoint.(RLs) .* RLs)
Knew = K - α * sum(adjoint.(dRLs) .* RLs .- adjoint.(RLs) .* dRLs) / 2
QLnew = Knew - 1 / 2 * sum(adjoint.(RLsnew) .* RLsnew)

QC = QL * C
RCs = RLs .* (C,)
RCsnew = RCs .- α .* gradRCs
QCnew = QC - α * gradQC
Cnew = C - α * gradC
RLsnew′ = RCsnew ./ (Cnew,)
QLnew′ = QCnew / Cnew
Knew′ = QLnew′ + 1 / 2 * sum(adjoint.(RLsnew′) .* RLsnew′)
Knew′ = (Knew′ - adjoint(Knew′)) / 2
QLnew′ = Knew′ - 1 / 2 * sum(adjoint.(RLsnew′) .* RLsnew′)
norm(RLsnew .- RLsnew′)
norm(QLnew - QLnew′)



ΨL, ρL, ρR, E, e = CMPSKit.groundstate(Ĥ, Ψ; gradtol=1e-2, optalg=CMPSKit.LBFGS(20; gradtol=1e-2, verbosity=0, maxiter=2000))
ρR, = rightenv(ΨL)
QL, RLs = ΨL
C = sqrt(ρR)
QR = C \ (QL * C)
RRs = (C,) .\ (RLs .* (C,))
KR = QR + sum(RR * RR' for RR in RRs) / 2
KR = (KR - KR') / 2
QR = KR - sum(RR * RR' for RR in RRs) / 2
ΨR = InfiniteCMPS(QR, RRs; gauge=:r)

α = 1e-3
for n = 1:1000
    HL, E, e, hL, info_HL =
        leftenv(Ĥ, (ΨL, one(ρR), C * C'))
    HR, E, e, hR, info_HR =
        rightenv(Ĥ, (ΨR, C' * C, one(ρR)))
    dC, dQC, dRCs = centergradient(Ĥ, (ΨL, ΨR, C), HL, HR)
    @show n, E

    QL, RLs = ΨL
    QR, RRs = ΨR
    QC = QL * C # == C * QR
    RCs = RLs .* (C,) # == (C,) .* RRs

    C = C - α * dC
    QC = QC - α * dQC
    RCs = RCs .- α .* dRCs

    RLs = RCs ./ (C,)
    QL = QC / C
    KL = QL + sum(RL' * RL for RL in RLs) / 2
    KL = (KL - KL') / 2
    QL = KL - sum(RL' * RL for RL in RLs) / 2

    ΨL = InfiniteCMPS(QL, RLs; gauge=:l)
    ρR, λ, info_ρR = rightenv(ΨL, C * C')
    ρR = rmul!(ρR, 1 / tr(ρR[]))
    C = sqrt(ρR)
    QR = C \ QL * C
    RRs = (C,) .\ RLs .* (C,)
    KR = QR + sum(RR * RR' for RR in RRs) / 2
    KR = (KR - KR') / 2
    QR = KR - sum(RR * RR' for RR in RRs) / 2
    ΨR = InfiniteCMPS(QR, RRs; gauge=:r)
end



# ΨL, ΨR, C, HL, HR, E, e, normgrad, numfg, history = CMPSKit.groundstate_unconstrained2(Ĥ, Ψ; gradtol=1e-5, optalg=CMPSKit.LBFGS(20; gradtol=1e-5, verbosity=0, maxiter=2000))

QL, RLs = ΨL
C = sqrt(ρR)
QR = C \ (QL * C)
RRs = (C,) .\ (RLs .* (C,))
ΨR = InfiniteCMPS(QR, RRs; gauge=:r)
HL, EL, eL, hL, infoL = leftenv(Ĥ, (ΨL, one(ρR), ρR))
HR, ER, eR, hR, infoR = rightenv(Ĥ, (ΨR, adjoint(C) * C, one(ρR)))
gradC, gradQC, gradRCs = centergradient(Ĥ, (ΨL, ΨR, C), HL, HR)


RL = RLs[1]
RR = RRs[1]
Z = zero(RL[])
I = one(RL[])
X = randn(size(RL[]))
Y = randn(size(RL[]))
QL′ = Constant([QL[] -RL[]'*X; Z -X'*X/2])
RL′ = Constant([RL[] X; Z Z])
QR′ = Constant([QR[] Z; -Y*RR[]' -Y*Y'/2])
RR′ = Constant([RR[] Z; Y Z])
C′ = Constant([C[] Z; Z Z])

ΨL′ = InfiniteCMPS(QL′, RL′; gauge=:l)
ΨR′ = InfiniteCMPS(QR′, RR′; gauge=:r)
HL′, EL′, eL′, hL′, infoL′ = leftenv(Ĥ, (ΨL′, one(C′), C′ * adjoint(C′)))
HR′, ER′, eR′, hR′, infoR′ = rightenv(Ĥ, (ΨR′, adjoint(C′) * C′, one(C′)))

@test EL ≈ ER ≈ EL′ ≈ ER′

gradC′, gradQC′, gradRCs′ = centergradient(Ĥ, (ΨL′, ΨR′, C′), HL′, HR′)
QC′ = QL′ * C′
RC′ = RL′ * C′

α = 1e-5
Cnew = C′ - 1e-5 * gradC′
QCnew = QC′ - 1e-5 * gradQC′
RCnew = RC′ - 1e-5 * gradRCs′[1]
QLnew = QCnew / Cnew
RLnew = RCnew / Cnew
KLnew = QLnew + RLnew' * RLnew / 2
@show norm(KLnew + KLnew')
KLnew = (KLnew - KLnew') / 2
QLnew = KLnew - RLnew' * RLnew / 2

ΨLnew = InfiniteCMPS(QLnew, RLnew; gauge=:l)
Enew = expval(Ĥ, ΨLnew)


# converge some further
ΨL, = CMPSKit.groundstate(Ĥ, ΨL; gradtol=1e-6, optalg=CMPSKit.LBFGS(20; gradtol=1e-5, verbosity=0, maxiter=5000))
