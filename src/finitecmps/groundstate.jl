function groundstate(H::LocalHamiltonian, Ψ₀::FiniteCMPS;
                        optalg::OptimKit.OptimizationAlgorithm =
                            LBFGS(30; verbosity = 2, gradtol = 1e-3),
                        boundaries = :fixed,
                        left_boundary = boundaries,
                        right_boundary = boundaries,
                        optimnodes = nodes(Ψ₀.Q),
                        Kmax = 60,
                        tol = eps(),
                        callback = (numiter, E, normgrad, Ψ, ρL, ρR, e) -> nothing)

    if !(left_boundary ∈ (:free, :fixed) && right_boundary ∈ (:free, :fixed))
         throw(ArgumentError("unknown boundary conditions, expected values are `:free` or `:fixed`"))
    end

    envnodes = nodes(Ψ₀.Q)
    optimindices::Vector{Int} = indexin(optimnodes, envnodes)
    @assert !any(isnothing, optimindices)
    @assert optimindices[1] == 1
    @assert optimindices[end] == length(envnodes)

    function fg(x)
        Ψ, HL, HR, ρL, ρR, hL, hR, E, e = x
        g = gradient(H, (Ψ, ρL, ρR), HL, HR; gradindices = optimindices,
                    left_boundary = left_boundary, right_boundary = right_boundary)
        return E, g
    end
    inner(x, g1, g2) = 2*real(dot(g1, g2))
    function retract(x, g, α)
        Ψ, = x
        Qold, Rsold, vLold, vRold = Ψ
        dQ, dRs, dvL, dvR = g
        Q = Qold + α * dQ
        Rs = Rsold .+ α .* dRs
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
        g = FiniteCMPSTangent(Ψ, dQ, dRs, dvL, dvR, optimindices)
        return (Ψ, HL, HR, ρL, ρR, hL, hR, E, e), g
    end
    function transport!(ξ, x, η, α, x′)
        Ψ, = x
        Ψ′, = x′
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
        return FiniteCMPSTangent(Ψ′, dQ, dRs, dvL, dvR, ξ.indices)
    end
    function finalize!(x, f, g, numiter)
        Ψ, HL, HR, ρL, ρR, hL, hR, E, e = x
        normgrad = sqrt(inner(x, g, g))
        callback(numiter, E, normgrad, Ψ, ρL, ρR, e)
        return x, f, g
    end

    x = let (Q₀, R₀s, vL₀, vR₀) = Ψ₀
        Q = PiecewiseLinear(collect(Q₀.nodes), copy.(Q₀.values))
        Rs = map(R₀s) do R₀
            PiecewiseLinear(collect(R₀.nodes), copy.(R₀.values))
        end
        vL = normalize(vL₀)
        vR = normalize(vR₀)
        Ψ = FiniteCMPS(Q, Rs, vL, vR)
        HL, HR, ρL, ρR, hL, hR, E, e, infoHL, infoHR, infoρL, infoρR =
            environments!(H, Ψ; Kmax = Kmax, tol = tol)
        (Ψ, HL, HR, ρL, ρR, hL, hR, E, e)
    end
    x, _, grad, numfg, history = optimize(fg, x, optalg; retract = retract, inner = inner,
                                            transport! = transport!, finalize! = finalize!)

    return let (Ψ, HL, HR, ρL, ρR, hL, hR, E, e) = x
        normgrad = sqrt(inner(x, grad, grad))
        Ψ, ρL, ρR, E, e, normgrad, numfg, history
    end
end
