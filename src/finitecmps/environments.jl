# Environments of the CMPS
function leftenv(Ψ::FiniteCMPS{<:AbstractPiecewise}; Kmax = 50, tol = eps())
    Q, Rs, vL, vR = Ψ
    a, b = domain(Q)
    grid = nodes(Q)
    N = length(grid) - 1
    ρ₁⁰ = zero(Q(a))
    ρ₁⁰ = mul!(ρ₁⁰, vL, vL', true, true)
    ρ₁ = TaylorSeries([ρ₁⁰], a)
    ρs = Vector{typeof(ρ₁)}(undef, N)
    ρs[1] = ρ₁
    infoL = true
    for i = 1:N
        ρᵢ = ρs[i]
        xᵢ = grid[i]
        xⱼ = grid[i+1]
        Δxᵢ = grid[i+1] - grid[i]
        Qᵢ = shift!(Q[i], xᵢ)
        Rᵢs = shift!.(getindex.(Rs, i), xᵢ)

        # build Taylor coefficients (i.e. solve triangular problem)
        ρᵢ, converged = _transferleft!(ρᵢ, zero(ρᵢ), Qᵢ, Rᵢs, Δxᵢ, Kmax, tol)
        infoL &= converged
        shift!(ρᵢ, (xᵢ+xⱼ)/2)

        # initialize next ρ element
        if i < N
            ρs[i+1] = TaylorSeries([ρᵢ(xⱼ)], xⱼ)
        end
    end
    ρL = Piecewise(grid, ρs)
    Z = vR'*ρL(b)*vR
    imag(Z) < defaulttol(Z) && real(Z) > 0 ||
        @warn "Non-positive left environment: Z = vR'*ρL(b)*vR = $Z"
    λ = log(abs(Z))/(b-a)/2
    return ρL, λ, infoL
end

function leftenv!(Ψ; Kmax = 50, tol = eps())
    ρL, λ, infoL = leftenv(Ψ)
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

function rightenv(Ψ::FiniteCMPS{<:AbstractPiecewise}; Kmax = 50, tol = eps())
    Q, Rs, vL, vR = Ψ
    a, b = domain(Q)
    grid = nodes(Q)
    N = length(grid) - 1
    ρN⁰ = zero(Q(b))
    ρN⁰ = mul!(ρN⁰, vR, vR', true, true)
    ρN = TaylorSeries([ρN⁰], b)
    ρs = Vector{typeof(ρN)}(undef, N)
    ρs[N] = ρN
    infoR = true
    for i = N:-1:1
        ρᵢ = ρs[i]
        xᵢ = grid[i]
        xⱼ = grid[i+1]
        Δxᵢ = grid[i+1] - grid[i]
        Qᵢ = shift!(Q[i], xⱼ)
        Rᵢs = shift!.(getindex.(Rs, i), xⱼ)

        # build Taylor coefficients (i.e. solve triangular problem)
        ρᵢ, converged = _transferright!(ρᵢ, zero(ρᵢ), Qᵢ, Rᵢs, Δxᵢ, Kmax, tol)
        infoR &= converged
        shift!(ρᵢ, (xᵢ+xⱼ)/2)

        # initialize next ρ element
        if i > 1
            ρs[i-1] = TaylorSeries([ρᵢ(xᵢ)], xᵢ)
        end
    end
    ρR = Piecewise(grid, ρs)
    Z = vL'*ρR(a)*vL
    imag(Z) < defaulttol(Z) && real(Z) > 0 ||
        @warn "Non-positive right environment: Z = vL'*ρR(a)*vL = $Z"
    λ = log(abs(Z))/(b-a)/2
    return ρR, λ, infoR
end

function rightenv!(Ψ; Kmax = 50, tol = eps())
    ρR, λ, infoR = rightenv(Ψ)
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

function environments!(Ψ::FiniteCMPS; Kmax = 50, tol = eps())
    ρL, infoL = leftenv!(Ψ; Kmax = Kmax, tol = tol)
    ρR, λR, infoR = rightenv(Ψ; Kmax = Kmax, tol = tol)
    λR < 10*tol || @warn "Incompatible normalization between left and right environments"
    localZ = localdot(ρL, ρR)
    (a, b) = domain(Ψ)
    Za = localZ(a)
    Zb = localZ(b)
    abs(Za -1 ) < defaulttol(Za) && abs(Zb-1) < defaulttol(Zb) ||
        @warn "Incompatible environment normalizations: Z = ⟨ρL|ρR⟩ = $Za = $Zb"
    return ρL, ρR, infoL, infoR
end

# Environments of the CMPS with the Hamiltonian
const FiniteCMPSData = Tuple{FiniteCMPS, MatrixFunction, MatrixFunction}

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
function leftenv(H::LocalHamiltonian, Ψρs::FiniteCMPSData; Kmax = 50, tol = eps())
    Ψ, ρL, ρR = Ψρs
    (a,b) = domain(Ψ)
    domain(H) == (a,b) || throw(DomainMismatch())
    hL = leftreducedoperator(H.h, Ψ, ρL)
    eL = localdot(hL, ρR)
    EL = integrate(eL, (a,b))
    Q = Ψ.Q
    Rs = Ψ.Rs
    grid = nodes(hL)
    HLs = similar(hL.elements)
    N = length(HLs)
    HLs[1] = TaylorSeries([zero(ρL(a))], a)
    infoL = true
    for i = 1:N
        HLᵢ = HLs[i]
        hLᵢ = hL[i]
        xᵢ = grid[i]
        xⱼ = grid[i+1]
        Δxᵢ = grid[i+1] - grid[i]
        Qᵢ = shift!(Q[i], xᵢ)
        Rᵢs = shift!.(getindex.(Rs, i), xᵢ)
        hLᵢ = shift(hL[i], xᵢ)

        # build Taylor coefficients (i.e. solve triangular problem)
        HLᵢ, converged = _transferleft!(HLᵢ, hLᵢ, Qᵢ, Rᵢs, Δxᵢ, Kmax, tol)
        infoL &= converged
        shift!(HLᵢ, (xᵢ+xⱼ)/2)
        shift!(hL[i], (xᵢ+xⱼ)/2)

        # initialize next ρ element
        if i < N
            HLs[i+1] = TaylorSeries([HLᵢ(xⱼ)], xⱼ)
        end
    end
    HL = Piecewise(grid, HLs)
    return HL, EL, eL, hL, infoL
end

function rightenv(H::LocalHamiltonian, Ψρs::FiniteCMPSData; Kmax = 50, tol = eps())
    Ψ, ρL, ρR = Ψρs
    (a,b) = domain(Ψ)
    domain(H) == (a,b) || throw(DomainMismatch())
    hR = rightreducedoperator(H.h, Ψ, ρR)
    eR = localdot(ρL, hR)
    ER = integrate(eR, (a,b))
    Q = Ψ.Q
    Rs = Ψ.Rs
    grid = nodes(hR)
    HRs = similar(hR.elements)
    N = length(HRs)
    HRs[N] = TaylorSeries([zero(ρR(b))], b)
    infoR = true
    for i = N:-1:1
        HRᵢ = HRs[i]
        xᵢ = grid[i]
        xⱼ = grid[i+1]
        Δxᵢ = grid[i+1] - grid[i]
        Qᵢ = shift(Q[i], xⱼ)
        Rᵢs = shift.(getindex.(Rs, i), xⱼ)
        hRᵢ = shift(hR[i], xⱼ)

        # build Taylor coefficients (i.e. solve triangular problem)
        HRᵢ, converged = _transferright!(HRᵢ, hRᵢ, Qᵢ, Rᵢs, Δxᵢ, Kmax, tol)
        infoR &= converged
        shift!(HRᵢ, (xᵢ+xⱼ)/2)

        # initialize next ρ element
        if i > 1
            HRs[i-1] = TaylorSeries([HRᵢ(xᵢ)], xᵢ)
        end
    end
    HR = Piecewise(grid, HRs)
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
    E = (EL+ER/2)
    e = rmul!(eL + eR, 1//2)
    return HL, HR, hL, hR, E, e, infoHL, infoHR
end
