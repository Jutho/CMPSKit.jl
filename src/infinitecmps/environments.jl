function leftenv(Ψ::InfiniteCMPS, ρ₀ = one(Ψ.Q);
                    tol = KrylovDefaults.tol,
                    krylovdim = KrylovDefaults.krylovdim,
                    maxiter = KrylovDefaults.maxiter,
                    eager = false,
                    kwargs...)

    alg = Arnoldi(; tol = tol, krylovdim = krylovdim, maxiter = maxiter, eager = eager)
    eigsort = EigSorter(x->(abs(div(imag(x), pi/period(Ψ))), -real(x)))
    let TL = LeftTransfer(Ψ)
        _, ρs, λs, info = schursolve(ρ₀, 1, eigsort, alg) do x
            y = -∂(x) + TL(x; tol = tol/10, kwargs...)
            return truncate!(y; tol = tol/10, kwargs...)
        end
        λ, ρ = λs[1]/2, ρs[1]
        imag(λ) <= max(info.normres[1], defaulttol(λ)) ||
            @warn "Largest eigenvalue of transfer matrix not real? $λ"
        ρ = ρ + ρ'
        ρ = rmul!(ρ, 1/(norm(ρ)*sign(tr(ρ[0]))))
        ρ = truncate!(ρ; tol = tol, kwargs...)
        res = -∂(ρ) + TL(ρ) - (2*λ)*ρ
        newinfo = ConvergenceInfo(info.converged, res, norm(res), info.numiter, info.numops)
        return ρ, real(λ), newinfo
    end
end

function leftenv!(Ψ::InfiniteCMPS, ρ₀ = one(Ψ.Q); kwargs...)
    ρ, λ, info = leftenv(Ψ, ρ₀; kwargs...)
    Q0 = Ψ.Q[0]
    view(Q0, diagind(Q0)) .-= λ
    return ρ, info
end

function rightenv(Ψ::InfiniteCMPS, ρ₀ = one(Ψ.Q);
                    tol = KrylovDefaults.tol,
                    krylovdim = KrylovDefaults.krylovdim,
                    maxiter = KrylovDefaults.maxiter,
                    eager = false,
                    kwargs...)

    alg = Arnoldi(; tol = tol, krylovdim = krylovdim, maxiter = maxiter, eager = eager)
    eigsort = EigSorter(x->(abs(div(imag(x), pi/period(Ψ))), -real(x)))
    let TR = RightTransfer(Ψ)
        _, ρs, λs, info = schursolve(ρ₀, 1, eigsort, alg) do x
                y = ∂(x) + TR(x; kwargs...)
                return truncate!(y; kwargs...)
            end
        λ, ρ = λs[1]/2, ρs[1]
        imag(λ) <= max(info.normres[1], defaulttol(λ)) ||
            @warn "Largest eigenvalue of transfer matrix not real? $λ"
        ρ = ρ + ρ'
        ρ = rmul!(ρ, 1/(norm(ρ)*sign(tr(ρ[0]))))
        ρ = truncate!(ρ; tol = tol, kwargs...)
        info.normres[1] = norm(∂(ρ) + TR(ρ) - (2*real(λ))*ρ)
        return ρ, real(λ), info
    end
end

function rightenv!(Ψ::InfiniteCMPS, ρ₀ = one(Ψ.Q); kwargs...)
    ρ, λ, info = rightenv(Ψ, ρ₀; kwargs...)
    Q0 = Ψ.Q[0]
    view(Q0, diagind(Q0)) .-= λ
    return ρ, info
end

function environments!(Ψ::InfiniteCMPS, ρL₀ = one(Ψ.Q), ρR₀ = one(Ψ.Q); kwargs...)
    ρL, λL, infoL = leftenv(Ψ, ρL₀; kwargs...)
    ρR, λR, infoR = rightenv(Ψ, ρR₀; kwargs...)
    isapprox(λL, λR; atol = defaulttol(λL), rtol = defaulttol(λL)) ||
        @warn "Different left and right normalization: $λL versus $λR"

    Q0 = Ψ.Q[0]
    Q0d = view(Q0, diagind(Q0))
    Q0d .-= (λL+λR)/2

    Z = dot(ρL, ρR)
    imag(Z) < defaulttol(Z) && real(Z) > 0 ||
        @warn "Incompatible fixed point environments: Z = ⟨ρL|ρR⟩ = $Z"

    ρL = rmul!(ρL, 1/sqrt(Z))
    ρR = rmul!(ρR, 1/sqrt(Z))

    return ρL, ρR, infoL, infoR
end

function leftenv!(H::LocalHamiltonian, Ψ::InfiniteCMPS, HL₀ = zero(Ψ.Q); kwargs...)
    domain(H) == domain(Ψ) || throw(DomainMismatch())
    ρL, ρR, infoL, infoR = environments!(Ψ; kwargs...)
    return leftenv(H, (Ψ,ρL,ρR), HL₀; kwargs...)
end

function rightenv!(H::LocalHamiltonian, Ψ::InfiniteCMPS, HR₀ = zero(Ψ.Q); kwargs...)
    domain(H) == domain(Ψ) || throw(DomainMismatch())
    ρL, ρR, infoL, infoR = environments!(Ψ; kwargs...)
    return rightenv(H, (Ψ,ρL,ρR), HR₀; kwargs...)
end

const InfiniteCMPSData = Tuple{InfiniteCMPS,PeriodicMatrixFunction,PeriodicMatrixFunction}

# assumes Ψ is normalized and ⟨ρL|ρR⟩ = 1
function leftenv(H::LocalHamiltonian, Ψρs::InfiniteCMPSData, HL₀ = zero(Ψρs[1].Q);
                    tol = KrylovDefaults.tol,
                    krylovdim = KrylovDefaults.krylovdim,
                    maxiter = KrylovDefaults.maxiter,
                    eager = false,
                    kwargs...)

    (Ψ,ρL,ρR) = Ψρs
    domain(H) == domain(Ψ) || throw(DomainMismatch())

    hL = leftreducedoperator(H.h, Ψ, ρL)
    eL = real(localdot(hL, ρR))
    hL = axpy!(-dot(ρR, hL), ρL, hL)

    HL₀ = HL₀ - ρL * dot(HL₀, ρR)
    alg = GMRES(;tol = tol, krylovdim = krylovdim, maxiter = maxiter)
    let TL = LeftTransfer(Ψ)
        HL, infoL = linsolve(hL, HL₀, alg) do x
            y = ∂(x) - TL(x; tol = tol/10, kwargs...)
            y = axpy!(dot(ρR, x), ρL, y)
            truncate!(y; tol = tol/10, kwargs...)
        end
        HL = rmul!(HL + HL', 0.5)
        HL = truncate!(HL; tol = tol/10, kwargs...)
        res = hL - (∂(HL)-TL(HL))
        infoL = ConvergenceInfo(infoL.converged, res, norm(res), infoL.numiter, infoL.numops)
        return HL, eL, hL, infoL
    end
end

function rightenv(H::LocalHamiltonian, Ψρs::InfiniteCMPSData, HR₀ = zero(Ψρs[1].Q);
                    tol = KrylovDefaults.tol,
                    krylovdim = KrylovDefaults.krylovdim,
                    maxiter = KrylovDefaults.maxiter,
                    eager = false,
                    kwargs...)

    (Ψ,ρL,ρR) = Ψρs
    domain(H) == domain(Ψ) || throw(DomainMismatch())

    hR = rightreducedoperator(H.h, Ψ, ρR)
    eR = real(localdot(ρL, hR))
    hR = axpy!(-dot(ρL, hR), ρR, hR)

    HR₀ = HR₀ - ρR * dot(ρL, HR₀)
    alg = GMRES(;tol = tol, krylovdim = krylovdim, maxiter = maxiter)
    let TR = RightTransfer(Ψ)
        HR, infoR = linsolve(hR, HR₀, alg) do x
            y = -∂(x) - TR(x; tol = tol/10, kwargs...)
            y = axpy!(dot(ρL, x), ρR, y)
            truncate!(y; tol = tol/10, kwargs...)
        end
        HR = rmul!(HR + HR', 0.5)
        HR = truncate!(HR; tol = tol/10, kwargs...)
        res = hR - (-∂(HR)-TR(HR))
        infoR = ConvergenceInfo(infoR.converged, res, norm(res), infoR.numiter, infoR.numops)
        return HR, eR, hR, infoR
    end
end

function environments!(H::LocalHamiltonian, Ψ::InfiniteCMPS; kwargs...)
    ρL, ρR, infoρL, infoρR = environments!(Ψ; kwargs...)
    HL, HR, hL, hR, e, infoHL, infoHR = environments(H,(Ψ,ρL,ρR); kwargs...)
    return HL, HR, ρL, ρR, hL, hR, e, infoHL, infoHR, infoρL, infoρR
end

function environments(H::LocalHamiltonian, Ψρs::InfiniteCMPSData,
                        HL₀ = zero(Ψρs[1].Q), HR₀ = zero(Ψρs[1].Q); kwargs...)
    HL, eL, hL, infoHL = leftenv(H, Ψρs, HL₀; kwargs...)
    HR, eR, hR, infoHR = rightenv(H, Ψρs, HR₀; kwargs...)
    eL ≈ eR ||
        @warn "non-matching energy from left and right environments"
    e = rmul!(eL + eR, 0.5)
    return HL, HR, hL, hR, e, infoHL, infoHR
end
