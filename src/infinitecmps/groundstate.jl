# groundstate with UniformCMPS
function groundstate(H::LocalHamiltonian, Ψ₀::UniformCMPS;
                        optalg = ConjugateGradient(; verbosity = 2, gradtol = 1e-7),
                        eigalg = defaulteigalg(Ψ₀),
                        linalg = defaultlinalg(Ψ₀),
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
        ρR, λ, infoR = rightenv(ΨL, ρR; eigalg = eigalg, linalg = linalg, kwargs...)
        rmul!(ρR, 1/tr(ρR[]))
        ρL = one(ρR)
        HL, E, e, hL, infoL =
            leftenv(H, (ΨL, ρL, ρR); eigalg = eigalg, linalg = linalg, kwargs...)

        if infoR.converged == 0 || infoL.converged == 0
            @warn "step $α : not converged, energy = $E"
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

    ΨL, = leftgauge(Ψ₀; kwargs...)
    ρR, λ, infoR = rightenv(ΨL; kwargs...)
    ρL = one(ρR)
    rmul!(ρR, 1/tr(ρR[]))
    HL, E, e, hL, infoL = leftenv(H, (ΨL,ρL,ρR); kwargs...)
    x = (ΨL, ρR, HL, E, e, hL)

    if infoR.converged == 0 || infoL.converged == 0
        @warn "initial point not converged, energy = $E"
        @show infoR
        @show infoL
    end

    x, E, grad, numfg, history =
        optimize(fg, x, optalg; retract = retract,
                                precondition = precondition,
                                finalize! = _finalize!,
                                inner = inner, transport! = transport!,
                                scale! = scale!, add! = add!,
                                isometrictransport = true)
    (ΨL, ρR, HL, E, e, hL) = x
    normgrad = sqrt(inner(x, grad, grad))
    return ΨL, ρR, E, e, normgrad, numfg, history
end

function groundstate(H::LocalHamiltonian, Ψ₀::FourierCMPS;
                        optalg = ConjugateGradient(; verbosity = 2, gradtol = 1e-7),
                        eigalg = defaulteigalg(Ψ₀),
                        linalg = defaultlinalg(Ψ₀),
                        finalize! = OptimKit._finalize!,
                        test = false,
                        kwargs...)

    δ = 1
    function retract(x, d, α)
        ΨL, ρR, HL, = x
        QL = ΨL.Q
        RLs = ΨL.Rs
        dK, dRs = d

        RdR = sum(adjoint.(RLs) .* dRs)
        dRdR = sum(adjoint.(dRs) .* dRs)

        QL = QL + α * dK - α * RdR - (α * α / 2) * dRdR
        RLs = RLs .+ α .* dRs

        ΨL = InfiniteCMPS(QL, RLs; gauge = :left)
        ρR, λ, infoR = rightenv(ΨL, ρR; eigalg = eigalg, linalg = linalg, kwargs...)
        rmul!(ρR, 1/tr(ρR[0]))
        ρL = one(ρR)
        HL, E, e, hL, infoL =
            leftenv(H, (ΨL, ρL, ρR); eigalg = eigalg, linalg = linalg, kwargs...)

        if infoR.converged == 0 || infoL.converged == 0
            @warn "step $α : not converged, energy = $E"
            @show infoR
            @show infoL
        end

        return (ΨL, ρR, HL, E, e, hL), d
    end

    transport!(v, x, d, α, xnew) = v # simplest possible transport

    function inner(x, d1, d2)
        dK1, dRs1 = d1
        dK2, dRs2 = d2
        return 2*real(dot(dK1, dK2)) + 2*real(sum(dot.(dRs1, dRs2)))
    end

    function fg(x)
        ΨL, ρR, HL, E, e, hL = x

        gradQ, gradRs = gradient(H, (ΨL, one(ρR), ρR), HL, zero(HL); kwargs...)

        Q = ΨL.Q
        RLs = ΨL.Rs

        dK = truncate!((gradQ - gradQ')/2; Kmax = nummodes(Q))
        dRs = truncate!.((.-(RLs)) .* (gradQ,) .+ gradRs; Kmax = nummodes(RLs[1]))

        return E, (dK, dRs)
    end

    function scale!(d, α)
        dK, dRs = d
        dK = rmul!(dK, α)
        dRs = rmul!.(dRs, α)
        return (dK, dRs)
    end

    function add!(d1, d2, α)
        dK1, dR1s = d1
        dK2, dR2s = d2
        axpy!(α, dK2, dK1)
        axpy!.(α, dR2s, dR1s)
        return (dK1, dR1s)
    end

    function precondition(x, d)
        ΨL, ρR, = x
        dK, dRs = d
        ρinv = posreginv(ρR[0], δ)
        dKρinv = sylvester(inv(ρinv), inv(ρinv), dK)
        return (dKρinv, dRs .* Ref(ρinv))
    end

    function _finalize!(x, E, d, numiter)
        normgrad2 = real(inner(x, d, d))
        δ = max(1e-12, 1e-1*normgrad2)
        return finalize!(x, E, d, numiter)
    end

    ΨL = Ψ₀
    ρR, λ, infoR = rightenv(ΨL; kwargs...)
    ρL = one(ρR)
    @assert norm(LeftTransfer(ΨL)(ρL)) < 1e-12
    rmul!(ρR, 1/tr(ρR[0]))
    HL, E, e, hL, infoL = leftenv(H, (ΨL,ρL,ρR); kwargs...)
    x = (ΨL, ρR, HL, E, e, hL)

    if infoR.converged == 0 || infoL.converged == 0
        @warn "initial point not converged, energy = $E"
        @show infoR
        @show infoL
    end

    if test
        return optimtest(fg, x; alpha = -0.1:0.01:0.1, retract = retract, inner = inner)
    end

    x, E, grad, numfg, history =
        optimize(fg, x, optalg; retract = retract,
                                precondition = precondition,
                                finalize! = _finalize!,
                                inner = inner, transport! = transport!,
                                scale! = scale!, add! = add!,
                                isometrictransport = true)
    (ΨL, ρR, HL, E, e, hL) = x
    normgrad = sqrt(inner(x, grad, grad))
    return ΨL, ρR, E, e, normgrad, numfg, history
end
