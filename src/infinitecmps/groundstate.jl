function groundstate(H::LocalHamiltonian, Ψ₀::UniformCMPS;
                            alg::OptimKit.OptimizationAlgorithm,
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
        ρR, λ, infoR = rightenv(ΨL; kwargs...)
        rmul!(ρR, 1/tr(ρR[]))
        ρL = one(ρR)
        HL, E, e, hL, infoL = leftenv(H, (ΨL,ρL,ρR); kwargs...)

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
        optimize(fg, x, alg; retract = retract,
                                precondition = precondition,
                                finalize! = _finalize!,
                                inner = inner, transport! = transport!,
                                scale! = scale!, add! = add!,
                                isometrictransport = true)
    (ΨL, ρR, HL, E, e, hL) = x
    return ΨL, ρR, E, e, normgrad, numfg, history
end

function groundstate2(H::LocalHamiltonian, Ψ₀::UniformCMPS;
                            alg::OptimKit.OptimizationAlgorithm, kwargs...)

    δ = 1
    function retract(x, d, α)
        ΨL, = x
        QL = ΨL.Q
        RLs = ΨL.Rs
        KL = copy(QL)
        for R in RLs
            mul!(KL, R', R, +1/2, 1)
        end

        dK, dRs = d
        RdR = zero(QL)
        for (R, dR) in zip(RLs, dRs)
            mul!(RdR, R', dR, true, true)
        end

        RLs = RLs .+ α .* dRs
        KL = KL + α * dK
        QL = KL
        for R in RLs
            mul!(QL, R', R, -1/2, 1)
        end

        ΨL = InfiniteCMPS(QL, RLs; gauge = :left)
        ρR, λ, infoR = rightenv(ΨL; kwargs...)
        rmul!(ρR, 1/tr(ρR[]))
        ρL = one(ρR)
        HL, E, e, hL, infoL = leftenv(H, (ΨL,ρL,ρR); kwargs...)

        if infoR.converged == 0 || infoL.converged == 0
            @warn "step $α : not converged, energy = $e"
            @show infoR
            @show infoL
        end

        return (ΨL, ρR, HL, E, e, hL), d
    end

    transport!(v, x, d, α, xnew) = v # simplest possible transport

    function inner(x, d1, d2)
        dK1, dR1s = d1
        dK2, dR2s = d2
        return 2*real(dot(dK1,dK2) + sum(dot.(dR1s, dR2s)))
    end

    function precondition(x, d)
        ΨL, ρR, = x
        dK, dRs = d
        P = posreginv(ρR[0], δ)
        return (dK, dRs .* Ref(P))
    end

    function fg(x)
        (ΨL, ρR, HL, E, e, hL) = x

        gradQ, gradRs = gradient(H, (ΨL, one(ρR), ρR), HL, zero(HL); kwargs...)

        Q = ΨL.Q
        Rs = ΨL.Rs

        dK = (gradQ - gradQ')/2
        dRs = Rs .* Ref((-1/2) * (gradQ + gradQ')) .+ gradRs

        return E, (dK, dRs)
    end

    scale!(d, α) = (rmul!(d[1], α), rmul!.(d[2], α))
    add!(d1, d2, α) = (axpy!(α, d2[1], d1[1]), axpy!.(α, d2[2], d1[2]))

    function finalize!(x, E, d, numiter)
        normgrad2 = real(inner(x, d, d))
        δ = max(1e-12, 1e-3*normgrad2)
        return x, E, d
    end

    ΨL₀, = leftgauge(Ψ₀; kwargs...)
    ρR, λ, infoR = rightenv(ΨL₀; kwargs...)
    ρL = one(ρR)
    rmul!(ρR, 1/tr(ρR[]))
    HL, E, e, hL, infoL = leftenv(H, (ΨL₀,ρL,ρR); kwargs...)
    x = (ΨL₀, ρR, HL, E, e, hL)

    x, E, normgrad, numfg, history =
        optimize(fg, x, alg; retract = retract, precondition = precondition,
                                finalize! = finalize!,
                                inner = inner, transport! = transport!,
                                scale! = scale!, add! = add!,
                                isometrictransport = true)
    (ΨL, ρR, HL, E, e, hL) = x
    return ΨL, ρR, E, e, normgrad, numfg, history
end

function groundstate3(H::LocalHamiltonian, Ψ₀::UniformCMPS;
                            alg::OptimKit.OptimizationAlgorithm, δmax = 1e-6, kwargs...)

    δ = δmax
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
        ρR, λ, infoR = rightenv(ΨL; kwargs...)
        rmul!(ρR, 1/tr(ρR[]))
        ρL = one(ρR)
        HL, E, e, hL, infoL = leftenv(H, (ΨL,ρL,ρR); kwargs...)

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
        R = ΨL.Rs[1]
        return dRs .* Ref(posreginv(R[0]*ρR[0]*R[0]', δ))
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

    function finalize!(x, E, d, numiter)
        normgrad2 = real(inner(x, d, d))
        δ = max(1e-12, 1e-3*normgrad2)
        return x, E, d
    end

    ΨL₀, = leftgauge(Ψ₀; kwargs...)
    ρR, λ, infoR = rightenv(ΨL₀; kwargs...)
    ρL = one(ρR)
    rmul!(ρR, 1/tr(ρR[]))
    HL, E, e, hL, infoL = leftenv(H, (ΨL₀,ρL,ρR); kwargs...)
    x = (ΨL₀, ρR, HL, E, e, hL)

    x, E, normgrad, numfg, history =
        optimize(fg, x, alg; retract = retract, precondition = precondition,
                                finalize! = finalize!,
                                inner = inner, transport! = transport!,
                                scale! = scale!, add! = add!,
                                isometrictransport = true)
    (ΨL, ρR, HL, E, e, hL) = x
    return ΨL, ρR, E, e, normgrad, numfg, history
end


function testgroundstate(H::LocalHamiltonian, Ψ₀::UniformCMPS; kwargs...)

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
        ρR, λ, infoR = rightenv(ΨL; kwargs...)
        rmul!(ρR, 1/tr(ρR[]))
        ρL = one(ρR)
        HL, E, e, hL, infoL = leftenv(H, (ΨL,ρL,ρR); kwargs...)

        if infoR.converged == 0 || infoL.converged == 0
            @warn "step $α : not converged, energy = $e"
            @show infoR
            @show infoL
        end

        return (ΨL, ρR, HL, E, e, hL), dRs
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


    ΨL₀, = leftgauge(Ψ₀; kwargs...)
    ρR, λ, infoR = rightenv(ΨL₀; kwargs...)
    rmul!(ρR, 1/tr(ρR[]))
    ρL = one(ρR)
    HL, E, e, hL, infoL = leftenv(H, (ΨL₀,ρL,ρR); kwargs...)
    x = (ΨL₀, ρR, HL, E, e, hL)

    return optimtest(fg, x; retract = retract,
                                inner = inner)
end

function testgroundstate2(H::LocalHamiltonian, Ψ₀::UniformCMPS; kwargs...)

    δ = 1e-6
    function retract(x, d, α)
        ΨL, = x
        QL = ΨL.Q
        RLs = ΨL.Rs
        KL = copy(QL)
        for R in RLs
            mul!(KL, R', R, +1/2, 1)
        end

        dK, dRs = d
        RdR = zero(QL)
        for (R, dR) in zip(RLs, dRs)
            mul!(RdR, R', dR, true, true)
        end

        RLs = RLs .+ α .* dRs
        KL = KL + α * dK
        QL = KL
        for R in RLs
            mul!(QL, R', R, -1/2, 1)
        end

        ΨL = InfiniteCMPS(QL, RLs; gauge = :left)
        ρR, λ, infoR = rightenv(ΨL; kwargs...)
        rmul!(ρR, 1/tr(ρR[]))
        ρL = one(ρR)
        HL, E, e, hL, infoL = leftenv(H, (ΨL,ρL,ρR); kwargs...)

        if infoR.converged == 0 || infoL.converged == 0
            @warn "step $α : not converged, energy = $e"
            @show infoR
            @show infoL
        end

        return (ΨL, ρR, HL, E, e, hL), d
    end

    transport!(v, x, d, α, xnew) = v # simplest possible transport

    function inner(x, d1, d2)
        dK1, dR1s = d1
        dK2, dR2s = d2
        return 2*real(dot(dK1,dK2) + sum(dot.(dR1s, dR2s)))
    end

    function precondition(x, d)
        ΨL, ρR, = x
        dK, dRs = d
        P = posreginv(ρR[0], δ)
        return (dK * P, dRs .* Ref(P))
    end

    function fg(x)
        (ΨL, ρR, HL, E, e, hL) = x

        gradQ, gradRs = gradient(H, (ΨL, one(ρR), ρR), HL, zero(HL); kwargs...)

        Q = ΨL.Q
        Rs = ΨL.Rs

        dK = (gradQ - gradQ')/2
        dRs = Rs .* Ref((-1/2)*(gradQ + gradQ')) .+ gradRs

        return E, (dK, dRs)
    end

    scale!(d, α) = (rmul!(d[1], α), rmul!.(d[2], α))
    add!(d1, d2, α) = (axpy!(α, d1[1], d2[1]), axpy!.(α, d1[2], d2[2]))

    ΨL₀, = leftgauge(Ψ₀; kwargs...)
    ρR, λ, infoR = rightenv(ΨL₀; kwargs...)
    rmul!(ρR, 1/tr(ρR[]))
    ρL = one(ρR)
    HL, E, e, hL, infoL = leftenv(H, (ΨL₀,ρL,ρR); kwargs...)
    x = (ΨL₀, ρR, HL, E, e, hL)

    return optimtest(fg, x; alpha = -1e-3:1e-5:1e-3, retract = retract,
                                inner = inner)
end
