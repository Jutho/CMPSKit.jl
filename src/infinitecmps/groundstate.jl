function groundstate(H::LocalHamiltonian, Ψ₀::UniformCMPS;
                        optalg = LBFGS(20; verbosity = 2),
                        eigalg = KrylovKit.Arnoldi(; krylovdim = min(64, virtualdim(Ψ₀))),
                        linalg = KrylovKit.GMRES(; krylovdim = min(256, virtualdim(Ψ₀))),
                        finalize! = OptimKit._finalize!,
                        kwargs...)

    δ = 1
    function retract(x, d, α)
        ΨL, = x
        QL = ΨL.Q
        RLs = ΨL.Rs
        KL = copy(QL)
        for R in RLs
            mul!(KL, R', R, +1/2, 1)
        end

        dRs = d
        RdR = zero(QL)
        for (R, dR) in zip(RLs, dRs)
            mul!(RdR, R', dR, true, true)
        end

        RLs = RLs .+ α .* dRs
        KL = KL - (α/2) * (RdR - RdR')
        QL = KL
        for R in RLs
            mul!(QL, R', R, -1/2, 1)
        end

        ΨL = InfiniteCMPS(QL, RLs; gauge = :left)
        ρR, λ, infoR = rightenv(ΨL; eigalg = eigalg, linalg = linalg, kwargs...)
        rmul!(ρR, 1/tr(ρR[]))
        ρL = one(ρR)
        HL, E, e, hL, infoL =
            leftenv(H, (ΨL,ρL,ρR); eigalg = eigalg, linalg = linalg, kwargs...)

        if infoR.converged == 0 || infoL.converged == 0
            @warn "step $α : not converged, energy = $e"
            @show infoR
            @show infoL
        end

        return (ΨL, ρR, HL, E, e, hL), d
    end

    transport!(v, x, d, α, xnew) = v # simplest possible transport

    function inner(x, d1, d2)
        return 2*real(sum(dot.(d1, d2)))
    end

    function precondition(x, d)
        ΨL, ρR, = x
        dRs = d
        return dRs .* Ref(posreginv(ρR[0], δ))
    end

    function fg(x)
        (ΨL, ρR, HL, E, e, hL) = x

        gradQ, gradRs = gradient(H, (ΨL, one(ρR), ρR), HL, zero(HL); kwargs...)

        Rs = ΨL.Rs

        dRs = .-(Rs) .* Ref(gradQ) .+ gradRs

        return E, dRs
    end

    scale!(d, α) = rmul!.(d, α)
    add!(d1, d2, α) = axpy!.(α, d2, d1)

    function _finalize!(x, E, d, numiter)
        normgrad2 = real(inner(x, d, d))
        δ = max(1e-12, 1e-3*normgrad2)
        return finalize!(x, E, d, numiter)
    end

    ΨL₀, = leftgauge(Ψ₀; kwargs...)
    ρR, λ, infoR = rightenv(ΨL₀; kwargs...)
    ρL = one(ρR)
    rmul!(ρR, 1/tr(ρR[]))
    HL, E, e, hL, infoL = leftenv(H, (ΨL₀,ρL,ρR); kwargs...)
    x = (ΨL₀, ρR, HL, E, e, hL)

    x, E, normgrad, numfg, history =
        optimize(fg, x, optalg; retract = retract,
                                precondition = precondition,
                                finalize! = _finalize!,
                                inner = inner, transport! = transport!,
                                scale! = scale!, add! = add!,
                                isometrictransport = true)
    (ΨL, ρR, HL, E, e, hL) = x
    return ΨL, ρR, E, e, normgrad, numfg, history
end

function groundstate2(H::LocalHamiltonian, Ψ₀::UniformCMPS;
                        optalg = LBFGS(20; verbosity = 2),
                        eigalg = KrylovKit.Arnoldi(; krylovdim = min(64, virtualdim(Ψ₀))),
                        linalg = KrylovKit.GMRES(; krylovdim = min(256, virtualdim(Ψ₀))),
                        finalize! = OptimKit._finalize!,
                        kwargs...)

    δ = 1
    function retract(x, d, α)
        ΨL, = x
        QL = ΨL.Q
        RLs = ΨL.Rs
        KL = copy(QL)
        for R in RLs
            mul!(KL, R', R, +1/2, 1)
        end

        dRs = d
        RdR = zero(QL)
        for (R, dR) in zip(RLs, dRs)
            mul!(RdR, R', dR, true, true)
        end

        RLs = RLs .+ α .* dRs
        KL = KL - (α/2) * (RdR - RdR')
        QL = KL
        for R in RLs
            mul!(QL, R', R, -1/2, 1)
        end

        ΨL = InfiniteCMPS(QL, RLs; gauge = :left)
        ρR, λ, infoR = rightenv(ΨL; eigalg = eigalg, linalg = linalg, kwargs...)
        rmul!(ρR, 1/tr(ρR[]))
        ρL = one(ρR)
        HL, E, e, hL, infoL =
            leftenv(H, (ΨL,ρL,ρR); eigalg = eigalg, linalg = linalg, kwargs...)

        if infoR.converged == 0 || infoL.converged == 0
            @warn "step $α : not converged, energy = $e"
            @show infoR
            @show infoL
        end

        return (ΨL, ρR, HL, E, e, hL), d
    end

    transport!(v, x, d, α, xnew) = v # simplest possible transport

    function inner(x, d1, d2)
        return 2*real(sum(dot.(d1, d2)))
    end

    function precondition(x, d)
        ΨL, ρR, = x
        dRs = d
        return dRs .* Ref(posreginv(ρR[0], δ))
    end

    function fg(x)
        (ΨL, ρR, HL, E, e, hL) = x

        gradQ, gradRs = gradient(H, (ΨL, one(ρR), ρR), HL, zero(HL); kwargs...)

        Rs = ΨL.Rs

        dRs = .-(Rs) .* Ref(gradQ) .+ gradRs

        return E, dRs
    end

    scale!(d, α) = rmul!.(d, α)
    add!(d1, d2, α) = axpy!.(α, d2, d1)

    function _finalize!(x, E, d, numiter)
        normgrad2 = real(inner(x, d, d))
        δ = max(1e-12, 1e-3*normgrad2)
        return finalize!(x, E, d, numiter)
    end

    ΨL₀, = leftgauge(Ψ₀; kwargs...)
    ρR, λ, infoR = rightenv(ΨL₀; kwargs...)
    ρL = one(ρR)
    rmul!(ρR, 1/tr(ρR[]))
    HL, E, e, hL, infoL = leftenv(H, (ΨL₀,ρL,ρR); kwargs...)
    x = (ΨL₀, ρR, HL, E, e, hL)

    x, E, normgrad, numfg, history =
        optimize(fg, x, optalg; retract = retract,
                                precondition = precondition,
                                finalize! = _finalize!,
                                inner = inner, transport! = transport!,
                                scale! = scale!, add! = add!,
                                isometrictransport = true)
    (ΨL, ρR, HL, E, e, hL) = x
    return ΨL, ρR, E, e, normgrad, numfg, history
end
