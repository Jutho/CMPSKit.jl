# Environments of the CMPS
function leftenv(Ψ::FiniteCMPS{<:AbstractPiecewise}; Kmax = 50, tol = defaulttol(Ψ))
    Q, Rs, vL, vR = Ψ
    a, b = domain(Ψ)
    ρL, infoL = lefttransfer(vL*vL', nothing, Ψ; Kmax = Kmax, tol = tol)
    Z = vR'*ρL(b)*vR
    imag(Z) < defaulttol(Z) && real(Z) > 0 ||
        @warn "Non-positive left environment: Z = vR'*ρL(b)*vR = $Z"
    λ = log(abs(Z))/(b-a)/2
    return ρL, λ, infoL
end

function leftenv!(Ψ::FiniteCMPS{<:AbstractPiecewise}; Kmax = 50, tol = defaulttol(Ψ))
    ρL, λ, infoL = leftenv(Ψ; Kmax = Kmax, tol = tol)
    (a,b) = domain(Ψ)
    Ψ.Q -= λ * one(Ψ.Q)
    for i = 1:length(ρL)
        xᵢ = ρL.nodes[i]
        xⱼ = ρL.nodes[i+1]
        xc = offset(ρL[i])
        Δx = max(abs(xⱼ - xc), abs(xc - xᵢ))
        exptaylor = [exp(-2*λ*(xc-a))]
        for k = 1:Kmax
            push!(exptaylor, exptaylor[k]*(-2*λ)/k)
            abs(exptaylor[k+1]*Δx^k) < tol && break
        end
        ρL[i] = truncmul(TaylorSeries(exptaylor, xc), ρL[i];
                            Kmax = Kmax, tol = tol, dx = Δx)
    end
    return ρL, infoL
end

function rightenv(Ψ::FiniteCMPS{<:AbstractPiecewise}; Kmax = 50, tol = defaulttol(Ψ))
    Q, Rs, vL, vR = Ψ
    a, b = domain(Q)
    ρR, infoR = righttransfer(vR*vR', nothing, Ψ; Kmax = Kmax, tol = tol)
    Z = vL'*ρR(a)*vL
    imag(Z) < defaulttol(Z) && real(Z) > 0 ||
        @warn "Non-positive right environment: Z = vL'*ρR(a)*vL = $Z"
    λ = log(abs(Z))/(b-a)/2
    return ρR, λ, infoR
end

function rightenv!(Ψ::FiniteCMPS{<:AbstractPiecewise}; Kmax = 50, tol = defaulttol(Ψ))
    ρR, λ, infoR = rightenv(Ψ; Kmax = Kmax, tol = tol)
    Ψ.Q -= λ * one(Ψ.Q)
    for i = 1:length(ρR)
        xᵢ = ρR.nodes[i]
        xⱼ = ρR.nodes[i+1]
        xc = offset(ρR[i])
        Δx = max(abs(xⱼ - xc), abs(xc - xᵢ))
        exptaylor = [exp(2*λ*(xc-b))]
        for k = 1:Kmax
            push!(exptaylor, exptaylor[k]*(2*λ)/k)
            abs(exptaylor[k+1]*Δx^k) < tol && break
        end
        ρR[i] = truncmul(TaylorSeries(exptaylor, xc), ρR[i];
                            Kmax = Kmax, tol = tol, dx = Δx)
    end
    return ρR, infoR
end

function environments!(Ψ::FiniteCMPS; Kmax = 50, tol = defaulttol(Ψ))
    ρL, infoL = leftenv!(Ψ; Kmax = Kmax, tol = tol)
    ρR, λR, infoR = rightenv(Ψ; Kmax = Kmax, tol = tol)
    λR < 1000*tol ||
        @warn "Incompatible normalization between left and right environments: λR = $λR"
    localZ = localdot(ρL, ρR)
    (a, b) = domain(Ψ)
    Za = localZ(a)
    Zb = localZ(b)
    abs(Za-1) < sqrt(defaulttol(Za)) && abs(Zb-1) < sqrt(defaulttol(Zb)) ||
        @warn "Incompatible environment normalizations: Z = ⟨ρL|ρR⟩ = $Za = $Zb"
    return ρL, ρR, infoL, infoR
end

# Environments of the CMPS with the Hamiltonian
const FiniteCMPSData = Tuple{FiniteCMPS{<:PiecewiseLinear}, Piecewise, Piecewise}

function leftenv!(H::LocalHamiltonian, Ψ::FiniteCMPS; kwargs...)
    domain(H) == domain(Ψ) || throw(DomainMismatch())
    ρL, ρR, infoL, infoR = environments!(Ψ; kwargs...)
    return leftenv(H, (Ψ,ρL,ρR); kwargs...)
end

function rightenv!(H::LocalHamiltonian, Ψ::FiniteCMPS; kwargs...)
    domain(H) == domain(Ψ) || throw(DomainMismatch())
    ρL, ρR, infoL, infoR = environments!(Ψ; kwargs...)
    return rightenv(H, (Ψ,ρL,ρR); kwargs...)
end

# assumes Ψ is normalized and ⟨ρL|ρR⟩ = 1
function leftenv(H::LocalHamiltonian, Ψρs::FiniteCMPSData;
                    Kmax = 50, tol = defaulttol(Ψρs[1]))
    Ψ, ρL, ρR = Ψρs
    (a,b) = domain(Ψ)
    domain(H) == (a,b) || throw(DomainMismatch())
    hL = leftreducedoperator(H.h, Ψ, ρL)
    eL = localdot(hL, ρR)
    EL = integrate(eL, (a,b))
    HL, infoL = lefttransfer(zero(hL(a)), hL, Ψ; Kmax = Kmax, tol = tol)
    return HL, EL, eL, hL, infoL
end

function rightenv(H::LocalHamiltonian, Ψρs::FiniteCMPSData;
                    Kmax = 50, tol = defaulttol(Ψρs[1]))
    Ψ, ρL, ρR = Ψρs
    (a,b) = domain(Ψ)
    domain(H) == (a,b) || throw(DomainMismatch())
    hR = rightreducedoperator(H.h, Ψ, ρR)
    eR = localdot(ρL, hR)
    ER = integrate(eR, (a,b))
    HR, infoR = righttransfer(zero(hR(b)), hR, Ψ; Kmax = Kmax, tol = tol)
    return HR, ER, eR, hR, infoR
end

function environments!(H::LocalHamiltonian, Ψ::FiniteCMPS; kwargs...)
    ρL, ρR, infoρL, infoρR = environments!(Ψ; kwargs...)
    HL, HR, hL, hR, E, e, infoHL, infoHR = environments(H,(Ψ,ρL,ρR); kwargs...)
    return HL, HR, ρL, ρR, hL, hR, E, e, infoHL, infoHR, infoρL, infoρR
end

function environments(H::LocalHamiltonian, Ψρs::FiniteCMPSData; kwargs...)
    HL, EL, eL, hL, infoHL = leftenv(H, Ψρs; kwargs...)
    HR, ER, eR, hR, infoHR = rightenv(H, Ψρs; kwargs...)
    EL ≈ ER ||
        @warn "non-matching energy from left and right environments"
    E = (EL+ER)/2
    e = rmul!(eL + eR, 1//2)
    return HL, HR, hL, hR, E, e, infoHL, infoHR
end
