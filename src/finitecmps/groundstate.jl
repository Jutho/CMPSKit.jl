function groundstate(H::LocalHamiltonian,
                        Ψ₀::FiniteCMPS,
                        alg::OptimKit.OptimizationAlgorithm;
                        boundaries = :fixed,
                        left_boundary = boundaries,
                        right_boundary = boundaries,
                        optimnodes = nodes(Ψ₀.Q),
                        Kmax = 60,
                        tol = eps(),
                        plot = (Ψ, ρL, ρR, e, E) -> nothing)

    if !(left_boundary ∈ (:free, :fixed) && right_boundary ∈ (:free, :fixed))
         throw(ArgumentError("unknown boundary conditions, expected values are `:free` or `:fixed`"))
    end

    function fg(x)
        Ψ, HL, HR, ρL, ρR, hL, hR, E, e = x
        gradQ, gradRs, gradvL, gradvR =
            gradient(H, (Ψ, ρL, ρR), HL, HR; gradindices = optimindices)
        return E, (gradQ, gradRs, gradvL, gradvR)
    end
    function inner(x, g1, g2)
        gradQ1, gradRs1, gradvL1, gradvR1 = g1
        gradQ2, gradRs2, gradvL2, gradvR2 = g2
        s = 2*real(dot(gradQ1, gradQ2))
        for (gradR1, gradR2) in zip(gradRs1, gradRs2)
            s += 2*real(dot(gradR1, gradR2))
        end
        if left_boundary == :free
            s += 2*real(dot(gradvL1, gradvL2))
        end
        if right_boundary == :free
            s += 2*real(dot(gradvR1, gradvR2))
        end
        return s
    end
    function retract(x, g, α)
        Ψ, = x
        Qold, Rolds, vLold, vRold = Ψ
        dQ, dRs, dvL, dvR = g
        Q = update(Qold, dQ, α)
        Rs = ntuple(length(dRs)) do n
            return update(Rolds[n], dRs[n], α)
        end
        if left_boundary == :free
            ndvL = norm(dvL)
            sα, cα = sincos(α*ndvL)
            vL = normalize!(axpby!(sα/ndvL, dvL, cα, copy(vLold)))
            dvL = axpby!(-sα*ndvL, vLold, cα, copy(dvL))
        else
            vL = vLold
        end
        if right_boundary == :free
            ndvR = norm(dvR)
            sα, cα = sincos(α*ndvR)
            vR = normalize!(axpby!(sα/ndvR, dvR, cα, copy(vRold)))
            dvR = axpby!(-sα*ndvR, vRold, cα, copy(dvR))
        else
            vR = vRold
        end
        Ψ = FiniteCMPS(Q, Rs, vL, vR)
        HL, HR, ρL, ρR, hL, hR, E, e, infoHL, infoHR, infoρL, infoρR =
            environments!(H, Ψ; Kmax = Kmax, tol = tol)
        infoHL || @warn "HL not converged"
        infoHR || @warn "HR not converged"
        infoρL || @warn "ρL not converged"
        infoρR || @warn "ρR not converged"
        g = (dQ, dRs, dvL, dvR)
        return (Ψ, HL, HR, ρL, ρR, hL, hR, E, e), g
    end
    function update(p::PiecewiseLinear, dp, α)
        pvnew = p.values[optimindices]
        pvnew .+= α .* dp
        pnew = PiecewiseLinear(optimnodes, pvnew)
        if nodes(p) === optimnodes
            return pnew
        else
            return PiecewiseLinear(nodes(p), pnew.(nodes(p)))
        end
    end
    function scale!(g, α)
        dQ, dRs, dvL, dvR = g
        rmul!(dQ, α)
        for dR in dRs
            rmul!(dR, α)
        end
        rmul!(dvL, α)
        rmul!(dvR, α)
        return (dQ, dRs, dvL, dvR)
    end
    function add!(g1, g2, β)
        dQ1, dR1s, dvL1, dvR1 = g1
        dQ2, dR2s, dvL2, dvR2 = g2
        axpy!(β, dQ2, dQ1)
        for (dR1, dR2) in zip(dR1s, dR2s)
            axpy!(β, dR2, dR1)
        end
        axpy!(β, dvL2, dvL1)
        axpy!(β, dvR2, dvR1)
        return (dQ1, dR1s, dvL1, dvR1)
    end
    function transport!(ξ, x, η, α, x′)
        Ψ, = x
        Qold, Rolds, vLold, vRold = Ψ
        dQ, dRs, dvL, dvR = ξ
        _, _, dvL′, dvR′ = η
        if left_boundary == :free
            ndvL′ = norm(dvL′)
            sα, cα = sincos(α*ndvL′)
            d1 = dot(vLold, dvL)
            d2 = dot(dvL′, dvL)/ndvL′
            dvL = axpy!((cα-1)*d1 - sα*d2, vLold, dvL)
            dvL = axpy!(((cα-1)*d2 + sα*d1)/ndvL′, dvL′, dvL)
        end
        if right_boundary == :free
            ndvR′ = norm(dvR′)
            sα, cα = sincos(α*ndvR′)
            d1 = dot(vRold, dvR)
            d2 = dot(dvR′, dvR)/ndvR′
            dvR = axpy!((cα-1)*d1 - sα*d2, vRold, dvR)
            dvR = axpy!(((cα-1)*d2 + sα*d1)/ndvR′, dvR′, dvR)
        end
        return (dQ, dRs, dvL, dvR)
    end
    function finalize!(x, f, g, numiter)
        Ψ, HL, HR, ρL, ρR, hL, hR, E, e = x
        plot(Ψ, ρL, ρR, e, E)
        # # make some plots
        # plotgrid = 0:1e-4:1
        # e = energydensity(H, ψ)
        # R = Rs[1]
        # density = localdot(ρL, R*ρR*R')
        # subplot(1,2,1)
        # plot(plotgrid, e.(plotgrid))
        # subplot(1,2,2)
        # plot(plotgrid, density.(plotgrid))
        # display(ylim((0,5)))
        return x, f, g
    end

    envnodes = nodes(Ψ₀.Q)
    optimindices = indexin(optimnodes, envnodes)
    @assert !any(isnothing, optimindices)
    @assert optimindices[1] == 1
    @assert optimindices[end] == length(envnodes)

    Q₀, R₀s, vL₀, vR₀ = Ψ₀
    Q = copy(Q₀)
    Rs = R₀s
    vL = normalize(vL₀)
    vR = normalize(vR₀)
    Ψ = FiniteCMPS(Q, Rs, vL, vR)
    HL, HR, ρL, ρR, hL, hR, E, e, infoHL, infoHR, infoρL, infoρR =
        environments!(H, Ψ; Kmax = Kmax, tol = tol)
    x = (Ψ, HL, HR, ρL, ρR, hL, hR, E, e)

    x, E, grad, numfg, history = optimize(fg, x, alg; retract = retract,
                                            inner = inner,
                                            transport! = transport!,
                                            scale! = scale!,
                                            add! = add!,
                                            finalize! = finalize!)

    (Ψ, HL, HR, ρL, ρR, hL, hR, E, e) = x
    normgrad = sqrt(inner(x, grad, grad))
    return Ψ, ρL, ρR, E, e, normgrad, numfg, history
end

function groundstate2(H::LocalHamiltonian,
                        Ψ₀::FiniteCMPS,
                        alg::OptimKit.OptimizationAlgorithm;
                        boundaries = :fixed,
                        left_boundary = boundaries,
                        right_boundary = boundaries,
                        optimnodes = nodes(Ψ₀.Q),
                        Kmax = 60,
                        tol = eps(),
                        plot = (Ψ, ρL, ρR, e, E) -> nothing)

    if !(left_boundary ∈ (:free, :fixed) && right_boundary ∈ (:free, :fixed))
         throw(ArgumentError("unknown boundary conditions, expected values are `:free` or `:fixed`"))
    end

    function fg(x)
        Ψ, HL, HR, ρL, ρR, hL, hR, E, e = x
        gradQ, gradRs, gradvL, gradvR =
            gradient2(H, (Ψ, ρL, ρR), HL, HR; gradindices = optimindices)
        return E, (gradQ, gradRs, gradvL, gradvR)
    end
    function inner(x, g1, g2)
        gradQ1, gradRs1, gradvL1, gradvR1 = g1
        gradQ2, gradRs2, gradvL2, gradvR2 = g2
        s = 2*real(dot(gradQ1, gradQ2))
        for (gradR1, gradR2) in zip(gradRs1, gradRs2)
            s += 2*real(dot(gradR1, gradR2))
        end
        if left_boundary == :free
            s += 2*real(dot(gradvL1, gradvL2))
        end
        if right_boundary == :free
            s += 2*real(dot(gradvR1, gradvR2))
        end
        return s
    end
    function retract(x, g, α)
        Ψ, = x
        Qold, Rolds, vLold, vRold = Ψ
        dQ, dRs, dvL, dvR = g
        Q = Qold + α * dQ
        Rs = ntuple(length(dRs)) do n
            Rolds[n] + α * dRs[n]
        end
        if left_boundary == :free
            ndvL = norm(dvL)
            sα, cα = sincos(α*ndvL)
            vL = normalize!(axpby!(sα/ndvL, dvL, cα, copy(vLold)))
            dvL = axpby!(-sα*ndvL, vLold, cα, copy(dvL))
        else
            vL = vLold
        end
        if right_boundary == :free
            ndvR = norm(dvR)
            sα, cα = sincos(α*ndvR)
            vR = normalize!(axpby!(sα/ndvR, dvR, cα, copy(vRold)))
            dvR = axpby!(-sα*ndvR, vRold, cα, copy(dvR))
        else
            vR = vRold
        end
        Ψ = FiniteCMPS(Q, Rs, vL, vR)
        HL, HR, ρL, ρR, hL, hR, E, e, infoHL, infoHR, infoρL, infoρR =
            environments!(H, Ψ; Kmax = Kmax, tol = tol)
        infoHL || @warn "HL not converged"
        infoHR || @warn "HR not converged"
        infoρL || @warn "ρL not converged"
        infoρR || @warn "ρR not converged"
        g = (dQ, dRs, dvL, dvR)
        return (Ψ, HL, HR, ρL, ρR, hL, hR, E, e), g
    end
    function scale!(g, α)
        dQ, dRs, dvL, dvR = g
        rmul!(dQ, α)
        for dR in dRs
            rmul!(dR, α)
        end
        rmul!(dvL, α)
        rmul!(dvR, α)
        return (dQ, dRs, dvL, dvR)
    end
    function add!(g1, g2, β)
        dQ1, dR1s, dvL1, dvR1 = g1
        dQ2, dR2s, dvL2, dvR2 = g2
        axpy!(β, dQ2, dQ1)
        for (dR1, dR2) in zip(dR1s, dR2s)
            axpy!(β, dR2, dR1)
        end
        axpy!(β, dvL2, dvL1)
        axpy!(β, dvR2, dvR1)
        return (dQ1, dR1s, dvL1, dvR1)
    end
    function transport!(ξ, x, η, α, x′)
        Ψ, = x
        Qold, Rolds, vLold, vRold = Ψ
        dQ, dRs, dvL, dvR = ξ
        _, _, dvL′, dvR′ = η
        if left_boundary == :free
            ndvL′ = norm(dvL′)
            sα, cα = sincos(α*ndvL′)
            d1 = dot(vLold, dvL)
            d2 = dot(dvL′, dvL)/ndvL′
            dvL = axpy!((cα-1)*d1 - sα*d2, vLold, dvL)
            dvL = axpy!(((cα-1)*d2 + sα*d1)/ndvL′, dvL′, dvL)
        end
        if right_boundary == :free
            ndvR′ = norm(dvR′)
            sα, cα = sincos(α*ndvR′)
            d1 = dot(vRold, dvR)
            d2 = dot(dvR′, dvR)/ndvR′
            dvR = axpy!((cα-1)*d1 - sα*d2, vRold, dvR)
            dvR = axpy!(((cα-1)*d2 + sα*d1)/ndvR′, dvR′, dvR)
        end
        return (dQ, dRs, dvL, dvR)
    end
    function finalize!(x, f, g, numiter)
        Ψ, HL, HR, ρL, ρR, hL, hR, E, e = x
        plot(Ψ, ρL, ρR, e, E)
        # # make some plots
        # plotgrid = 0:1e-4:1
        # e = energydensity(H, ψ)
        # R = Rs[1]
        # density = localdot(ρL, R*ρR*R')
        # subplot(1,2,1)
        # plot(plotgrid, e.(plotgrid))
        # subplot(1,2,2)
        # plot(plotgrid, density.(plotgrid))
        # display(ylim((0,5)))
        return x, f, g
    end

    envnodes = nodes(Ψ₀.Q)
    optimindices = indexin(optimnodes, envnodes)
    @assert !any(isnothing, optimindices)
    @assert optimindices[1] == 1
    @assert optimindices[end] == length(envnodes)

    Q₀, R₀s, vL₀, vR₀ = Ψ₀
    Q = copy(Q₀)
    Rs = R₀s
    vL = normalize(vL₀)
    vR = normalize(vR₀)
    Ψ = FiniteCMPS(Q, Rs, vL, vR)
    HL, HR, ρL, ρR, hL, hR, E, e, infoHL, infoHR, infoρL, infoρR =
        environments!(H, Ψ; Kmax = Kmax, tol = tol)
    x = (Ψ, HL, HR, ρL, ρR, hL, hR, E, e)

    x, E, grad, numfg, history = optimize(fg, x, alg; retract = retract,
                                            inner = inner,
                                            transport! = transport!,
                                            scale! = scale!,
                                            add! = add!,
                                            finalize! = finalize!)

    (Ψ, HL, HR, ρL, ρR, hL, hR, E, e) = x
    normgrad = sqrt(inner(x, grad, grad))
    return Ψ, ρL, ρR, E, e, normgrad, numfg, history
end
