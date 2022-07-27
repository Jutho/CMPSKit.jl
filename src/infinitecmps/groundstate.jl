using Printf

# groundstate with UniformCMPS
groundstate(Ĥ::LocalHamiltonian, Ψ₀::UniformCMPS; kwargs...) =
    groundstate_unconstrained(Ĥ, Ψ₀; kwargs...)

groundstate(Ĥ::LocalHamiltonian, Ψ₀::UniformCMPS, n₀::Number; kwargs...) =
    groundstate_constrained(Ĥ, Ψ₀, ntuple(k->n₀, Val(length(Ψ₀.Rs))); kwargs...)

groundstate(Ĥ::LocalHamiltonian,
            Ψ₀::UniformCMPS{<:AbstractMatrix,N},
            n₀s::NTuple{N,<:Number}; kwargs...) where {N} =
    groundstate_constrained(Ĥ, Ψ₀, n₀s; kwargs...)

function groundstate_unconstrained(Ĥ::LocalHamiltonian, Ψ₀::UniformCMPS;
                                    gradtol = 1e-7,
                                    verbosity = 2,
                                    optalg = LBFGS(; gradtol = gradtol, verbosity = verbosity - 2),
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
        ρR, λ, info_ρR = rightenv(ΨL, ρR; eigalg = eigalg, linalg = linalg, kwargs...)
        rmul!(ρR, 1/tr(ρR[]))
        HL, E, e, hL, info_HL =
            leftenv(Ĥ, (ΨL, ρL, ρR); eigalg = eigalg, linalg = linalg, kwargs...)

        if info_ρR.converged == 0 || info_HL.converged == 0
            @warn "step $α : not converged, e = $E"
            @show info_ρR
            @show info_HL
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

        gradQ, gradRs = gradient(Ĥ, (ΨL, ρL, ρR), HL, zero(HL); kwargs...)

        Rs = ΨL.Rs

        dRs = .-(Rs) .* Ref(gradQ) .+ gradRs

        return E, dRs
    end

    scale!(d, α) = rmul!.(d, α)
    add!(d1, d2, α) = axpy!.(α, d2, d1)

    function _finalize!(x, E, d, numiter)
        normgrad2 = inner(x, d, d)
        δ = max(1e-12, 1e-3*normgrad2)
        normgrad = sqrt(normgrad2)
        verbosity > 1 &&
            @info @sprintf("UniformCMPS ground state: iter %4d: e = %.12f, ‖∇e‖ = %.4e",
                                numiter, E, normgrad)
        return finalize!(x, E, d, numiter)
    end

    ΨL, = leftgauge(Ψ₀; kwargs...)
    ρR, λ, info_ρR = rightenv(ΨL; kwargs...)
    ρL = one(ρR)
    rmul!(ρR, 1/tr(ρR[]))
    HL, E, e, hL, info_HL = leftenv(Ĥ, (ΨL,ρL,ρR); kwargs...)
    x = (ΨL, ρR, HL, E, e, hL)

    if info_ρR.converged == 0 || info_HL.converged == 0
        @warn "initial point not converged, energy = $E"
        @show info_ρR
        @show info_HL
    end

    verbosity > 0 &&
        @info @sprintf("UniformCMPS ground state: initialization with e = %.12f", E)

    x, E, grad, numfg, history =
        optimize(fg, x, optalg; retract = retract,
                                precondition = precondition,
                                finalize! = _finalize!,
                                inner = inner, transport! = transport!,
                                scale! = scale!, add! = add!,
                                isometrictransport = true)
    (ΨL, ρR, HL, E, e, hL) = x
    normgrad = sqrt(inner(x, grad, grad))
    if verbosity > 0
        if normgrad <= gradtol
            @info @sprintf("UniformCMPS ground state: converged after %d iterations: e = %.12f, ‖∇e‖ = %.4e",
                            size(history, 1), E, normgrad)
        else
            @warn @sprintf("UniformCMPS ground state: not converged to requested tol: e = %.12f, ‖∇e‖ = %.4e",
                            E, normgrad)
        end
    end
    return ΨL, ρL, ρR, E, e, normgrad, numfg, history
end

# groundstate with UniformCMPS
function groundstate_constrained(Ĥ::LocalHamiltonian,
                        Ψ₀::UniformCMPS{<:AbstractMatrix, N},
                        n₀s::NTuple{N,Number};
                        gradtol = 1e-7,
                        verbosity = 2,
                        optalg = LBFGS(; gradtol = gradtol, verbosity = verbosity - 2),
                        eigalg = defaulteigalg(Ψ₀),
                        linalg = defaultlinalg(Ψ₀),
                        finalize! = OptimKit._finalize!,
                        chemical_potential_relaxation = 1.,
                        kwargs...) where {N}

    δ = 1
    μs = ntuple(k -> one(scalartype(Ψ₀)), N)
    n̂s = ntuple(k -> ψ̂[k]' * ψ̂[k], N)
    N̂s = ntuple(k -> ∫(n̂s[k], (-Inf,+Inf)), N)
    Ω̂ = Ĥ - sum( μs .* N̂s )
    function retract(x, d, α)
        ΨL, = x
        QL = ΨL.Q
        RLs = ΨL.Rs
        KL = copy(QL)
        for R in RLs
            mul!(KL, R', R, +1/2, 1)
        end

        dRs, dμs = d
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
        ρR, λ, info_ρR = rightenv(ΨL, ρR; eigalg = eigalg, linalg = linalg, kwargs...)
        rmul!(ρR, 1/tr(ρR[]))
        ns = ntuple(k->expval(n̂s[k], ΨL, ρL, ρR)[], N)
        ΩL, Ω, ω, ωL, info_ΩL =
            leftenv(Ω̂, (ΨL, ρL, ρR); eigalg = eigalg, linalg = linalg, kwargs...)

        if info_ρR.converged == 0 || info_ΩL.converged == 0
            @warn "step $α : not converged, ω = $Ω"
            @show info_ρR
            @show info_ΩL
        end

        return (ΨL, ρR, ΩL, Ω, ω, ns, ωL), d
    end

    transport!(v, x, d, α, xnew) = v # simplest possible transport

    function inner(x, d1, d2)
        dRs1, dμs1 = d1
        dRs2, dμs2 = d2
        return 2*real(sum(dot.(dRs1, dRs2))) + sum(dμs1 .* dμs2)
    end

    function precondition(x, d)
        ΨL, ρR, = x
        dRs, dμs = d
        return (dRs .* Ref(posreginv(ρR[0], δ)), zero.(dμs)) # no updates of μ
    end

    function fg(x)
        (ΨL, ρR, ΩL, Ω, ω, ns, ωL) = x

        gradQ, gradRs = gradient(Ω̂, (ΨL, ρL, ρR), ΩL, zero(ΩL); kwargs...)

        Rs = ΨL.Rs

        dRs = .-(Rs) .* Ref(gradQ) .+ gradRs

        dμs = n₀s .- ns

        return Ω, (dRs, dμs)
    end

    scale!(d, α) = (rmul!.(d[1], α), d[2] .* α)
    add!(d1, d2, α) = (axpy!.(α, d2[1], d1[1]), α .* d2[2] .+ d1[2])

    function _finalize!(x, Ω, d, numiter)
        (ΨL, ρR, ΩL, Ω, ω, ns, ωL) = x
        normgrad2 = real(inner(x, d, d))
        normgrad = sqrt(normgrad2)
        E = expval(density(Ĥ), ΨL, ρL, ρR)[]
        dμs = d[2]
        if verbosity > 1
            s = @sprintf("UniformCMPS ground state: iter %4d: ", numiter)
            s *= _groundstate_constraint_infostring(Ω, E, ns, μs, normgrad)
            @info s
        end
        μs = μs .+ chemical_potential_relaxation .* dμs
        Ω̂ = Ĥ - sum(μs .* N̂s)
        δ = max(1e-12, 1e-3*normgrad2)
        # recompute energy and gradient:
        ΩL, Ω, ω, ωL, info_ΩL =
            leftenv(Ω̂, (ΨL, ρL, ρR); eigalg = eigalg, linalg = linalg, kwargs...)

        if info_ρR.converged == 0 || info_ΩL.converged == 0
            @warn "finalizing step with new chemical potential : not converged, ω = $Ω"
            @show info_ρR
            @show info_ΩL
        end

        x = (ΨL, ρR, ΩL, Ω, ω, ns, ωL)
        gradQ, gradRs = gradient(Ω̂, (ΨL, ρL, ρR), ΩL, zero(ΩL); kwargs...)
        Rs = ΨL.Rs
        dRs = .-(Rs) .* Ref(gradQ) .+ gradRs
        d = (dRs, dμs)
        return finalize!(x, Ω, d, numiter)
    end

    ΨL, = leftgauge(Ψ₀; kwargs...)

    ρR, λ, info_ρR = rightenv(ΨL; kwargs...)
    ρL = one(ρR)
    rmul!(ρR, 1/tr(ρR[]))
    ns = ntuple(k->expval(n̂s[k], ΨL, ρL, ρR)[], N)
    # rescale initial cMPS to better approximate target densities, using geometric mean
    # this does not change the environments ρL and ρR
    scale_factor = prod( n₀s ./ ns)^(1/N)
    rmul!(ΨL.Q, scale_factor)
    rmul!.(ΨL.Rs, sqrt(scale_factor))
    ns = ns .* scale_factor
    ΩL, Ω, ω, ωL, info_ΩL =
        leftenv(Ω̂, (ΨL, ρL, ρR); eigalg = eigalg, linalg = linalg, kwargs...)

    if info_ρR.converged == 0 || info_ΩL.converged == 0
        @warn "initial point not converged, ω = $Ω"
        @show info_ρR
        @show info_ΩL
    end
    x = (ΨL, ρR, ΩL, Ω, ω, ns, ωL)

    if verbosity > 0
        E = expval(density(Ĥ), ΨL, ρL, ρR)[]
        s = "UniformCMPS ground state: initalization with "
        s *= _groundstate_constraint_infostring(Ω, E, ns)
        @info s
    end

    x, Ω, grad, numfg, history =
        optimize(fg, x, optalg; retract = retract,
                                precondition = precondition,
                                finalize! = _finalize!,
                                inner = inner, transport! = transport!,
                                scale! = scale!, add! = add!,
                                isometrictransport = true)
    (ΨL, ρR) = x
    normgrad = sqrt(inner(x, grad, grad))
    e = expval(density(Ĥ), ΨL, ρL, ρR)
    E = e[]
    if verbosity > 0
        if normgrad <= gradtol
            s = @sprintf("UniformCMPS ground state: converged after %d iterations: ", size(history, 1))
        else
            s = "UniformCMPS ground state: not converged to requested tol: "
        end
        s *= _groundstate_constraint_infostring(Ω, E, ns, μs, normgrad)
        @info s
    end
    return ΨL, ρL, ρR, E, e, ns, μs, Ω, normgrad, numfg, history
end

function _groundstate_constraint_infostring(ω, e, ns, μs = nothing, normgrad = nothing)
    s = @sprintf("ω = %.12f, e = %.12f", ω, e)
    N = length(ns)
    if N == 1
        s *= @sprintf(", n = %.6f", ns[1])
        if !isnothing(μs)
            s *= @sprintf(", μ = %.6f", μs[1])
        end
        if !isnothing(μs)
            s *= @sprintf(", ‖∇ω‖ = %.4e", normgrad)
        end
    else
        s *= ", ns = ("
        for k = 1:N
            s *= @sprintf("%.3f", ns[k])
            if k < N
                s *= ", "
            else
                s *= ")"
            end
        end
        if !isnothing(μs)
            s *= ", μs = ("
            for k = 1:N
                s *= @sprintf("%.3f", μs[k])
                if k < N
                    s *= ", "
                else
                    s *= ")"
                end
            end
        end
        if !isnothing(normgrad)
            s *= @sprintf("), ‖∇ω‖ = %.4e", normgrad)
        end
    end
    return s
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

    # TODO: make this work and test this
    # function precondition(x, d)
    #     ΨL, ρR, = x
    #     dK, dRs = d
    #     ρinv = posreginv(ρR[0], δ)
    #     dKρinv = sylvester(inv(ρinv), inv(ρinv), dK)
    #     return (dKρinv, dRs .* Ref(ρinv))
    # end

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
                                # precondition = precondition, # TODO
                                finalize! = _finalize!,
                                inner = inner, transport! = transport!,
                                scale! = scale!, add! = add!,
                                isometrictransport = true)
    (ΨL, ρR, HL, E, e, hL) = x
    normgrad = sqrt(inner(x, grad, grad))
    return ΨL, one(ρR), ρR, E, e, normgrad, numfg, history
end
