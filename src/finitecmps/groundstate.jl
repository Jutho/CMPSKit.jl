function groundstate(H::LocalHamiltonian, Ψ₀::FiniteCMPS,
                                    alg::OptimKit.OptimizationAlgorithm;
                                    boundaries = :fixed,
                                    envnodes = nodes(Ψ₀.Q),
                                    Kmax = 60, tol = eps())
    # if boundaries == :fixed
    #     _groundstate_fixed(H, Ψ₀, alg; envnodes = envnodes, Kmax = Kmax, tol = tol)
    # elseif boundaries == :free
    #     _groundstate_free(H, Ψ₀, alg; envnodes = envnodes, Kmax = Kmax, tol = tol)
    # else
    #     throw(ArgumentError("unknown value for keyword `boundaries`, expected values are `:free` or `:fxied`"))
    # end

    function fg(x)
        Ψ, HL, HR, ρL, ρR, hL, hR, E, e = x
        gradQ, gradRs, = gradient(H, (Ψ, ρL, ρR), HL, HR; gradindices = optimindices)
        return E, (gradQ, gradRs)
    end
    function inner(x, g1, g2)
        gradQ1, gradRs1 = g1
        gradQ2, gradRs2 = g2
        s = 2*real(dot(gradQ1, gradQ2))
        for (gradR1, gradR2) in zip(gradRs1, gradRs2)
            s += 2*real(dot(gradR1, gradR2))
        end
        return s
    end
    function retract(x, g, α)
        Ψ, = x
        Qold, Rolds, vL, vR = Ψ
        dQ, dRs = g
        Q = update(Qold, dQ, α)
        Rs = ntuple(length(dRs)) do n
            return update(Rolds[n], dRs[n], α)
        end
        Ψ = FiniteCMPS(Q, Rs, vL, vR)
        HL, HR, ρL, ρR, hL, hR, E, e, infoHL, infoHR, infoρL, infoρR =
            environments!(H, Ψ; Kmax = Kmax, tol = tol)
        infoHL || @warn "HL not converged"
        infoHR || @warn "HR not converged"
        infoρL || @warn "ρL not converged"
        infoρR || @warn "ρR not converged"
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
        dQ, dRs = g
        rmul!(dQ, α)
        for dR in dRs
            rmul!(dR, α)
        end
        return g
    end
    function add!(g1, g2, β)
        dQ1, dR1s = g1
        dQ2, dR2s = g2
        axpy!(β, dQ2, dQ1)
        for (dR1, dR2) in zip(dR1s, dR2s)
            axpy!(β, dR2, dR1)
        end
        return g1
    end
    function finalize!(x, f, g, numiter)
        Ψ, HL, HR, ρL, ρR, hL, hR, E, e = x
        Q, Rs, vL, vR = Ψ
        (a, b) = domain(ρL)
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

    optimnodes = nodes(Ψ₀.Q)
    optimindices = indexin(optimnodes, envnodes)
    @assert !any(isnothing, optimindices)
    @assert optimindices[1] == 1
    @assert optimindices[end] == length(envnodes)

    Q₀, R₀s, vL, vR = Ψ₀
    Q = PiecewiseLinear(envnodes, Q₀.(envnodes))
    Rs = map(R₀s) do R₀
        PiecewiseLinear(envnodes, R₀.(envnodes))
    end
    Ψ = FiniteCMPS(Q, Rs, vL, vR)
    HL, HR, ρL, ρR, hL, hR, E, e, infoHL, infoHR, infoρL, infoρR =
        environments!(H, Ψ; Kmax = Kmax, tol = tol)
    x = (Ψ, HL, HR, ρL, ρR, hL, hR, E, e)

    return optimize(fg, x, alg; retract = retract, inner = inner,
                                scale! = scale!, add! = add!, finalize! = finalize!)
end
