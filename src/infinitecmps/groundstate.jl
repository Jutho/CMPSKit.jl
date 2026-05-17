using Printf

# groundstate with UniformCMPS
# function groundstate(Ĥ::LocalHamiltonian, Ψ₀::UniformCMPS; kwargs...)
#     return groundstate_unconstrained(Ĥ, Ψ₀; kwargs...)
# end

function groundstate(Ĥ::LocalHamiltonian, Ψ₀::UniformCMPS, n₀::Number; kwargs...)
    return groundstate_constrained(Ĥ, Ψ₀, ntuple(k -> n₀, Val(length(Ψ₀.Rs))); kwargs...)
end

function groundstate(Ĥ::LocalHamiltonian,
                     Ψ₀::UniformCMPS{<:AbstractMatrix,N},
                     n₀s::NTuple{N,<:Number}; kwargs...) where {N}
    return groundstate_constrained(Ĥ, Ψ₀, n₀s; kwargs...)
end

function groundstate_unconstrained(Ĥ::LocalHamiltonian, Ψ₀::UniformCMPS;
                                   gradtol=1e-7,
                                   verbosity=2,
                                   optalg=LBFGS(; gradtol=gradtol, verbosity=verbosity - 2),
                                   eigalg=defaulteigalg(Ψ₀),
                                   linalg=defaultlinalg(Ψ₀),
                                   (finalize!)=OptimKit._finalize!,
                                   kwargs...)
    δ = 1
    function retract(x, d, α)
        ΨL, = x
        QL = ΨL.Q
        RLs = ΨL.Rs
        KL = copy(QL)
        for R in RLs
            mul!(KL, R', R, +1 / 2, 1)
        end

        dRs = d
        RdR = zero(QL)
        for (R, dR) in zip(RLs, dRs)
            mul!(RdR, R', dR, true, true)
        end

        RLs = RLs .+ α .* dRs
        KL = KL - (α / 2) * (RdR - RdR')
        QL = KL
        for R in RLs
            mul!(QL, R', R, -1 / 2, 1)
        end

        ΨL = InfiniteCMPS(QL, RLs; gauge=:l)
        ρR, λ, info_ρR = rightenv(ΨL, ρR; eigalg=eigalg, linalg=linalg, kwargs...)
        rmul!(ρR, 1 / tr(ρR[]))
        HL, E, e, hL, info_HL = leftenv(Ĥ, (ΨL, ρL, ρR); eigalg=eigalg, linalg=linalg,
                                        kwargs...)

        if info_ρR.converged == 0 || info_HL.converged == 0
            @warn "step $α : not converged, e = $E"
            @show info_ρR
            @show info_HL
        end

        return (ΨL, ρR, HL, E, e, hL), d
    end

    transport!(v, x, d, α, xnew) = v # simplest possible transport

    function inner(x, d1, d2)
        return 2 * real(sum(dot.(d1, d2)))
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
        δ = max(1e-12, 1e-3 * normgrad2)
        normgrad = sqrt(normgrad2)
        verbosity > 1 &&
            @info @sprintf("UniformCMPS ground state: iter %4d: e = %.12f, ‖∇e‖ = %.4e",
                           numiter, E, normgrad)
        return finalize!(x, E, d, numiter)
    end

    ΨL, = leftgauge(Ψ₀; kwargs...)
    ρR, λ, info_ρR = rightenv(ΨL; kwargs...)
    ρL = one(ρR)
    rmul!(ρR, 1 / tr(ρR[]))
    HL, E, e, hL, info_HL = leftenv(Ĥ, (ΨL, ρL, ρR); kwargs...)
    x = (ΨL, ρR, HL, E, e, hL)

    if info_ρR.converged == 0 || info_HL.converged == 0
        @warn "initial point not converged, energy = $E"
        @show info_ρR
        @show info_HL
    end

    verbosity > 0 &&
        @info @sprintf("UniformCMPS ground state: initialization with e = %.12f", E)

    x, E, grad, numfg, history = optimize(fg, x, optalg; retract=retract,
                                          precondition=precondition,
                                          (finalize!)=_finalize!,
                                          inner=inner, (transport!)=transport!,
                                          (scale!)=scale!, (add!)=add!,
                                          isometrictransport=true)
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

function groundstate_unconstrained2(Ĥ::LocalHamiltonian, Ψ₀::UniformCMPS;
                                    gradtol=1e-7,
                                    verbosity=2,
                                    optalg=LBFGS(; gradtol=gradtol,
                                                 verbosity=verbosity - 2),
                                    eigalg=defaulteigalg(Ψ₀),
                                    linalg=defaultlinalg(Ψ₀),
                                    (finalize!)=OptimKit._finalize!,
                                    kwargs...)
    δ = 1
    function retract(x, d, α)
        ΨL, ΨR, C = x
        QL, RLs = ΨL
        QR, RRs = ΨR

        QC = QL * C # == C * QR
        RCs = RLs .* (C,) # == (C,) .* RRs

        dC, dQC, dRCs = d

        C = C + α * dC
        QC = QC + α * dQC
        RCs = RCs .+ α .* dRCs

        RLs = RCs ./ (C,)
        QL = QC / C
        KL = QL + sum(RL' * RL for RL in RLs) / 2
        KL = (KL - KL') / 2
        QL = KL - sum(RL' * RL for RL in RLs) / 2

        ΨL = InfiniteCMPS(QL, RLs; gauge=:l)
        ρR, λ, info_ρR = rightenv(ΨL, C * C'; eigalg=eigalg, linalg=linalg, kwargs...)
        ρR = rmul!(ρR, 1 / tr(ρR[]))
        C = sqrt(ρR)
        QR = C \ QL * C
        RRs = (C,) .\ RLs .* (C,)
        KR = QR + sum(RR * RR' for RR in RRs) / 2
        KR = (KR - KR') / 2
        QR = KR - sum(RR * RR' for RR in RRs) / 2
        ΨR = InfiniteCMPS(QR, RRs; gauge=:r)

        HL, E, e, hL, info_HL = leftenv(Ĥ, (ΨL, one(ρR), C * C'); eigalg=eigalg,
                                        linalg=linalg, kwargs...)
        HR, E, e, hR, info_HR = rightenv(Ĥ, (ΨR, C' * C, one(ρR)); eigalg=eigalg,
                                         linalg=linalg, kwargs...)
        if info_ρR.converged == 0 || info_HL.converged == 0 || info_HR.converged == 0
            @warn "step $α : not converged, e = $E"
            @show info_ρR
            @show info_HL
            @show info_HR
        end
        return (ΨL, ΨR, C, HL, HR, E, e, hL, hR), d
    end

    transport!(v, x, d, α, xnew) = v # simplest possible transport

    function inner(x, d1, d2)
        ΨL, ΨR, C, = x
        dC1, dQC1, dRCs1 = d1
        dC2, dQC2, dRCs2 = d2
        RLs = ΨL.Rs
        dWCs1 = dRCs1 .- RLs .* (dC1,)
        dWCs2 = dRCs2 .- RLs .* (dC2,)
        return 2 * real(sum(dot.(dWCs1, dWCs2)))
    end

    function precondition(x, d)
        return d
    end

    function fg(x)
        (ΨL, ΨR, C, HL, HR, E, e, hL, hR) = x
        gradC, gradQC, gradRCs = centergradient(Ĥ, (ΨL, ΨR, C), HL, HR)
        return E, (gradC, gradQC, gradRCs)
    end

    scale!((dC, dQC, dRCs), α) = (rmul!(dC, α), rmul!(dQC, α), rmul!.(dRCs, α))
    function add!((dC1, dQC1, dRCs1), (dC2, dQC2, dRCs2), α)
        return (axpy!(α, dC2, dC1), axpy!(α, dQC2, dQC1), axpy!.(α, dRCs2, dRCs1))
    end

    function _finalize!(x, E, d, numiter)
        normgrad2 = inner(x, d, d)
        δ = max(1e-12, 1e-3 * normgrad2)
        normgrad = sqrt(normgrad2)
        verbosity > 1 &&
            @info @sprintf("UniformCMPS ground state: iter %4d: e = %.12f, ‖∇e‖ = %.4e",
                           numiter, E, normgrad)
        return finalize!(x, E, d, numiter)
    end

    ΨL, = leftgauge(Ψ₀; kwargs...)
    ρR, λ, info_ρR = rightenv(ΨL; eigalg=eigalg, linalg=linalg, kwargs...)
    ρR = rmul!(ρR, 1 / tr(ρR[]))
    C = sqrt(ρR)
    QL, RLs = ΨL
    QR = C \ QL * C
    RRs = (C,) .\ RLs .* (C,)
    KR = QR + sum(RR * RR' for RR in RRs) / 2
    KR = (KR - KR') / 2
    QR = KR - sum(RR * RR' for RR in RRs) / 2
    ΨR = InfiniteCMPS(QR, RRs; gauge=:r)
    HL, E, e, hL, info_HL = leftenv(Ĥ, (ΨL, one(ρR), C * C'); eigalg=eigalg, linalg=linalg,
                                    kwargs...)
    HR, E, e, hR, info_HR = rightenv(Ĥ, (ΨR, C' * C, one(ρR)); eigalg=eigalg,
                                     linalg=linalg, kwargs...)
    x = (ΨL, ΨR, C, HL, HR, E, e, hL, hR)

    if info_ρR.converged == 0 || info_HL.converged == 0 || info_HR.converged == 0
        @warn "initial point not converged, energy = $E"
        @show info_ρR
        @show info_HL
        @show info_HR
    end

    verbosity > 0 &&
        @info @sprintf("UniformCMPS ground state: initialization with e = %.12f", E)

    x, E, grad, numfg, history = optimize(fg, x, optalg; retract=retract,
                                          precondition=precondition,
                                          (finalize!)=_finalize!,
                                          inner=inner, (transport!)=transport!,
                                          (scale!)=scale!, (add!)=add!,
                                          isometrictransport=true)
    (ΨL, ΨR, C, HL, HR, E, e, hL, hR) = x
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
    return ΨL, ΨR, C, HL, HR, E, e, normgrad, numfg, history
end

# groundstate with UniformCMPS
function groundstate_constrained(Ĥ::LocalHamiltonian,
                                 Ψ₀::UniformCMPS{<:AbstractMatrix,N},
                                 n₀s::NTuple{N,Number};
                                 gradtol=1e-7,
                                 verbosity=2,
                                 optalg=LBFGS(; gradtol=gradtol, verbosity=verbosity - 2),
                                 eigalg=defaulteigalg(Ψ₀),
                                 linalg=defaultlinalg(Ψ₀),
                                 (finalize!)=OptimKit._finalize!,
                                 chemical_potential_relaxation=1.0,
                                 kwargs...) where {N}
    δ = 1
    μs = ntuple(k -> one(scalartype(Ψ₀)), N)
    n̂s = ntuple(k -> ψ̂[k]' * ψ̂[k], N)
    N̂s = ntuple(k -> ∫(n̂s[k], (-Inf, +Inf)), N)
    Ω̂ = Ĥ - sum(μs .* N̂s)
    function retract(x, d, α)
        ΨL, = x
        QL = ΨL.Q
        RLs = ΨL.Rs
        KL = copy(QL)
        for R in RLs
            mul!(KL, R', R, +1 / 2, 1)
        end

        dRs, dμs = d
        RdR = zero(QL)
        for (R, dR) in zip(RLs, dRs)
            mul!(RdR, R', dR, true, true)
        end

        RLs = RLs .+ α .* dRs
        KL = KL - (α / 2) * (RdR - RdR')
        QL = KL
        for R in RLs
            mul!(QL, R', R, -1 / 2, 1)
        end

        ΨL = InfiniteCMPS(QL, RLs; gauge=:l)
        ρR, λ, info_ρR = rightenv(ΨL, ρR; eigalg=eigalg, linalg=linalg, kwargs...)
        rmul!(ρR, 1 / tr(ρR[]))
        ns = ntuple(k -> expval(n̂s[k], ΨL, ρL, ρR)[], N)
        ΩL, Ω, ω, ωL, info_ΩL = leftenv(Ω̂, (ΨL, ρL, ρR); eigalg=eigalg, linalg=linalg,
                                        kwargs...)

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
        return 2 * real(sum(dot.(dRs1, dRs2))) + sum(dμs1 .* dμs2)
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
        δ = max(1e-12, 1e-3 * normgrad2)
        # recompute energy and gradient:
        ΩL, Ω, ω, ωL, info_ΩL = leftenv(Ω̂, (ΨL, ρL, ρR); eigalg=eigalg, linalg=linalg,
                                        kwargs...)

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
    rmul!(ρR, 1 / tr(ρR[]))
    ns = ntuple(k -> expval(n̂s[k], ΨL, ρL, ρR)[], N)
    # rescale initial cMPS to better approximate target densities, using geometric mean
    # this does not change the environments ρL and ρR
    scale_factor = prod(n₀s ./ ns)^(1 / N)
    rmul!(ΨL.Q, scale_factor)
    rmul!.(ΨL.Rs, sqrt(scale_factor))
    ns = ns .* scale_factor
    ΩL, Ω, ω, ωL, info_ΩL = leftenv(Ω̂, (ΨL, ρL, ρR); eigalg=eigalg, linalg=linalg,
                                    kwargs...)

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

    x, Ω, grad, numfg, history = optimize(fg, x, optalg; retract=retract,
                                          precondition=precondition,
                                          (finalize!)=_finalize!,
                                          inner=inner, (transport!)=transport!,
                                          (scale!)=scale!, (add!)=add!,
                                          isometrictransport=true)
    (ΨL, ρR) = x
    normgrad = sqrt(inner(x, grad, grad))
    e = expval(density(Ĥ), ΨL, ρL, ρR)
    E = e[]
    if verbosity > 0
        if normgrad <= gradtol
            s = @sprintf("UniformCMPS ground state: converged after %d iterations: ",
                         size(history, 1))
        else
            s = "UniformCMPS ground state: not converged to requested tol: "
        end
        s *= _groundstate_constraint_infostring(Ω, E, ns, μs, normgrad)
        @info s
    end
    return ΨL, ρL, ρR, E, e, ns, μs, Ω, normgrad, numfg, history
end

function _groundstate_constraint_infostring(ω, e, ns, μs=nothing, normgrad=nothing)
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
        for k in 1:N
            s *= @sprintf("%.3f", ns[k])
            if k < N
                s *= ", "
            else
                s *= ")"
            end
        end
        if !isnothing(μs)
            s *= ", μs = ("
            for k in 1:N
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

# function groundstate(H::LocalHamiltonian, Ψ₀::FourierCMPS;
#                      optalg=ConjugateGradient(; verbosity=2, gradtol=1e-7),
#                      eigalg=defaulteigalg(Ψ₀),
#                      linalg=defaultlinalg(Ψ₀),
#                      (finalize!)=OptimKit._finalize!,
#                      test=false,
#                      kwargs...)
#     δ = 1
#     function retract(x, d, α)
#         ΨL, ρR, HL, = x
#         QL = ΨL.Q
#         RLs = ΨL.Rs
#         dK, dRs = d

#         RdR = sum(adjoint.(RLs) .* dRs)
#         dRdR = sum(adjoint.(dRs) .* dRs)

#         QL = QL + α * dK - α * RdR - (α * α / 2) * dRdR
#         RLs = RLs .+ α .* dRs

#         ΨL = InfiniteCMPS(QL, RLs; gauge=:l)
#         ρR, λ, infoR = rightenv(ΨL, ρR; eigalg=eigalg, linalg=linalg, kwargs...)
#         rmul!(ρR, 1 / tr(ρR[0]))
#         ρL = one(ρR)
#         HL, E, e, hL, infoL = leftenv(H, (ΨL, ρL, ρR); eigalg=eigalg, linalg=linalg,
#                                       kwargs...)

#         if infoR.converged == 0 || infoL.converged == 0
#             @warn "step $α : not converged, energy = $E"
#             @show infoR
#             @show infoL
#         end

#         return (ΨL, ρR, HL, E, e, hL), d
#     end

#     transport!(v, x, d, α, xnew) = v # simplest possible transport

#     function inner(x, d1, d2)
#         dK1, dRs1 = d1
#         dK2, dRs2 = d2
#         return 2 * real(dot(dK1, dK2)) + 2 * real(sum(dot.(dRs1, dRs2)))
#     end

#     function fg(x)
#         ΨL, ρR, HL, E, e, hL = x

#         gradQ, gradRs = gradient(H, (ΨL, one(ρR), ρR), HL, zero(HL); kwargs...)

#         Q = ΨL.Q
#         RLs = ΨL.Rs

#         dK = truncate!((gradQ - gradQ') / 2; Kmax=nummodes(Q))
#         dRs = truncate!.((.-(RLs)) .* (gradQ,) .+ gradRs; Kmax=nummodes(RLs[1]))

#         return E, (dK, dRs)
#     end

#     function scale!(d, α)
#         dK, dRs = d
#         dK = rmul!(dK, α)
#         dRs = rmul!.(dRs, α)
#         return (dK, dRs)
#     end

#     function add!(d1, d2, α)
#         dK1, dR1s = d1
#         dK2, dR2s = d2
#         axpy!(α, dK2, dK1)
#         axpy!.(α, dR2s, dR1s)
#         return (dK1, dR1s)
#     end

#     # TODO: make this work and test this
#     function precondition(x, d)
#         ΨL, ρR, = x
#         dK, dRs = d
#         ρinv = posreginv(ρR[0], δ)
#         dKρinv = sylvester(inv(ρinv), inv(ρinv), dK)
#         return (dKρinv, dRs .* Ref(ρinv))
#     end

#     function _finalize!(x, E, d, numiter)
#         normgrad2 = real(inner(x, d, d))
#         δ = max(1e-12, 1e-1 * normgrad2)
#         return finalize!(x, E, d, numiter)
#     end

#     ΨL = Ψ₀
#     ρR, λ, infoR = rightenv(ΨL; kwargs...)
#     ρL = one(ρR)
#     @assert norm(LeftTransfer(ΨL)(ρL)) < 1e-12
#     rmul!(ρR, 1 / tr(ρR[0]))
#     HL, E, e, hL, infoL = leftenv(H, (ΨL, ρL, ρR); kwargs...)
#     x = (ΨL, ρR, HL, E, e, hL)

#     if infoR.converged == 0 || infoL.converged == 0
#         @warn "initial point not converged, energy = $E"
#         @show infoR
#         @show infoL
#     end

#     if test
#         return optimtest(fg, x; alpha=-0.1:0.01:0.1, retract=retract, inner=inner)
#     end

#     x, E, grad, numfg, history = optimize(fg, x, optalg; retract=retract,
#                                           precondition = precondition, # TODO
#                                           (finalize!)=_finalize!,
#                                           inner=inner, (transport!)=transport!,
#                                           (scale!)=scale!, (add!)=add!,
#                                           isometrictransport=true)
#     (ΨL, ρR, HL, E, e, hL) = x
#     normgrad = sqrt(inner(x, grad, grad))
#     return ΨL, one(ρR), ρR, E, e, normgrad, numfg, history
# end

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

function groundstate_MDMinv(H::LocalHamiltonian, Ψ₀::UniformCMPS;
                        optalg = ConjugateGradient(; verbosity = 2, gradtol = 1e-7),
                        eigalg = defaulteigalg(Ψ₀),
                        linalg = defaultlinalg(Ψ₀),
                        kwargs...)

    δ = 1e-3                    # used for preconditioner regularization
    αcache = 1.0                # used for δ*α regularization
    P_qr = nothing              # cached qr(P) factorization
    x0_vec = nothing            # cached Krylov initial guess in packed-vector form
    function retract(x, d, α)
        ΨL, M, Minv, Ds, = x
        dX, dDs = d

        M =   M * Constant(exp(α * dX[]))
        Minv = Constant(exp(-α * dX[])) * Minv
        Ds = Ds .+ α .* dDs

        Rs_new = map(D -> M * D * Minv, Ds) 
        ΔRs = Rs_new .- ΨL.Rs

        QL_new = ΨL.Q - sum([R' * ΔR + 0.5 * ΔR' * ΔR for (R, ΔR) in zip(ΨL.Rs, ΔRs)])

        ΨL = InfiniteCMPS(QL_new, Rs_new; gauge = :left)
        ρR, _, infoR = rightenv(ΨL; eigalg = eigalg, linalg = linalg, kwargs...)
        rmul!(ρR, 1/tr(ρR[]))
        ρL = one(ρR)
        HL, E, e, hL, infoL =
            leftenv(H, (ΨL,ρL,ρR); eigalg = eigalg, linalg = linalg, kwargs...)

        if infoR.converged == 0 || infoL.converged == 0
            @warn "step $α : not converged, energy = $e"
            @show infoR
            @show infoL
        end

        return (ΨL, M, Minv, Ds, ρR, HL, E, e, hL), d
    end

    transport!(v, x, d, α, xnew) = v # simplest possible transport

    function inner(x, d1, d2)
        dX1, dDs1 = d1
        dX2, dDs2 = d2
        s = dX1 === dX2 ? norm(dX1)^2 : real(dot(dX1, dX2))
        for (dD1,dD2) in zip(dDs1, dDs2)
            if dD1 === dD2
                s += norm(dD1)^2
            else
                s += real(dot(dD1, dD2))
            end
        end
        return s
    end

    function precondition!(x, d; full::Bool=false)
        # x = (ΨL, M, Minv, Ds, ρR, HL, E, e, hL)
        _, M, Minv, Ds, ρR, _, _, _, _ = x
        dX, dDs = d

        χ = size(M[], 1)
        dcomp = length(Ds)
        n = χ^2 + dcomp*χ

        # PSD-stabilized rhoR
        U, S, _ = svd(ρR[])
        ρRmat = U * Diagonal(S) * U'

        EL = M'[] * M[]
        ER = Minv[] * ρRmat * (Minv)'[]

        # tangent mapping
        function _f(rv)
            _dX = rv[1]
            _dDs = rv[2:end]

            prec = [EL * (_dX[] * D[] - D[] * _dX[] + dD[]) * ER for (dD, D) in zip(_dDs, Ds)]

            dX_mapped = Constant(sum([S * D'[] - D'[] * S for (S, D) in zip(prec, Ds)]))
            dX_mapped[][diagind(dX_mapped[])] .= 0

            dDs_mapped = Constant.(diagm.(diag.(prec)))
            return RecursiveVec(dX_mapped, dDs_mapped...)
        end

        # (A + reg)(v) = _f(v) + reg(v), reg: δ on dDs, δ*α on dX
        Aplusreg = function(v)
            _dX = copy(reshape(view(v, 1:χ^2), χ, χ))
            off = χ^2
            _dDs = ntuple(k -> begin
                dD = v[off+1:off+χ]
                off += χ
                Constant(diagm(dD))
            end, dcomp)

            rv = RecursiveVec(Constant(_dX), _dDs...)
            
            y = _f(rv)

            dXreg = Constant(y[1][] + (δ * αcache) .* rv[1][])
            dXreg[][diagind(dXreg[])] .= 0
            dDsreg = ntuple(k -> Constant(y[1+k][] + δ .* rv[1+k][]), dcomp)

            yrv = RecursiveVec(dXreg, dDsreg...)

            w = similar(v)
            w[1:χ^2] .= vec(yrv[1][])
            off = χ^2
            for k in 1:dcomp
                w[off+1:off+χ] .= diag(yrv[1+k][])
                off += χ
            end
            return w
        end

        # RHS is the current gradient
        b = zeros(ComplexF64, n)
        b[1:χ^2] .= vec(dX[])
        off = χ^2
        for k in 1:dcomp
            b[off+1:off+χ] .= diag(dDs[k][])
            off += χ
        end

        # build cached P_qr once
        if P_qr === nothing
            P = zeros(ComplexF64, n, n)
            for i in 1:n
                ei = zeros(ComplexF64, n)
                ei[i] = 1.0
                P[:, i] = Aplusreg(ei)
            end
            ϵ = 1e-10 * opnorm(P, 1)
            @inbounds for i in 1:n
                P[i,i] += ϵ
            end
            P_qr = qr(P)
            x0_vec = nothing
        end

        bhat = P_qr \ b

        if full
            dnew = bhat
        else
            Ahat = vec -> (P_qr \ Aplusreg(vec))

            # better starting vector
            x0 = x0_vec === nothing ? bhat : x0_vec

            # tolerance heuristic (preconditioner solves can be loose)
            tol = norm(bhat) * min(sqrt(max(δ, 1e-12)), 1e-3)

            # one/two-step Krylov
            dnew, info = KrylovKit.linsolve(
                Ahat,
                bhat,
                x0;
                maxiter=2,
                tol=tol,
                ishermitian=true,
                isposdef=true,
                verbosity=0,
            )

            # if cache stale, rebuild once
            if norm(info.residual) > tol
                @info "preconditioner: residual too large → rebuilding cached P"
                P_qr = nothing
                return precondition!(x, d; full=true)
            end

            x0_vec = dnew
        end
        
        dXsol = copy(reshape(view(dnew, 1:χ^2), χ, χ))

        off = χ^2
        dDs_sol = ntuple(k -> begin
            dD = dnew[off+1:off+χ]
            off += χ
            Constant(diagm(dD))
        end, dcomp)

        preconditioned_gradient = (Constant(dXsol), dDs_sol)

        return preconditioned_gradient
    end

    function fg(x)
        ΨL, M, Minv, _, ρR, HL, E,  = x

        gradQ, gradRs = gradient(H, (ΨL, one(ρR), ρR), HL, zero(HL); kwargs...)

        Rs = ΨL.Rs
        dRs = map((R, gradR) -> gradR - R * gradQ, Rs, gradRs)

        dDs = map(dR -> M'[] * dR[] * Minv'[], dRs)
        dDs = 2 .* Constant.(diagm.((diag.(dDs))))

        dX = 2 * sum(map((dR, R) -> M' * (dR * R' - R' * dR) * Minv', dRs, Rs))

        dX[][diagind(dX[])] .= 0 

        return E, (dX, dDs)
    end

    function scale!(d, α)
        dX, dDs = d
        rmul!(dX, α)
        for dD in dDs
            rmul!(dD, α)
        end
        return d
    end
    function add!(d1, d2, α)
        dX1, dD1s = d1
        dX2, dD2s = d2
        axpy!(α, dX2, dX1)
        for (dD1, dD2) in zip(dD1s, dD2s)
            axpy!(α, dD2, dD1)
        end
        return d1
    end

    function _finalize!(x, E, d, numiter)
        αcache = abs(E)^(1/3)
        dX, dDs = d
        dX_scaled = αcache^(-3) * dX
        dDs_scaled = αcache^(-2.5) .* dDs
        d_scaled = (dX_scaled, dDs_scaled...)
        δ = max(1e-12, norm(d_scaled)^2)

        _, _, _, _, ρR, _, E, = x
        println("finalize: δ = $(δ), α = $(αcache), cond number = $(cond(ρR[]))")
        return x, E, d, numiter
    end

    ΨL₀ = Ψ₀
    M = Constant(eigvecs(ΨL₀.Rs[1][]))
    Minv = Constant(inv(M[]))
    D1 = diagm(diag(Minv[] * ΨL₀.Rs[1][] * M[]))
    D2 = diagm(diag(Minv[] * ΨL₀.Rs[2][] * M[]))
    Ds = Constant.((D1,D2))
    ρR, _, infoR = rightenv(ΨL₀; kwargs...)
    ρL = one(ρR)
    rmul!(ρR, 1/tr(ρR[]))
    HL, E, e, hL, infoL = leftenv(H, (ΨL₀,ρL,ρR); kwargs...)
    x = (ΨL₀, M, Minv, Ds, ρR, HL, E, e, hL)

    x, E, normgrad, numfg, history =
    optimize(fg, x, optalg; retract = retract,
                            finalize! = _finalize!,
                            precondition = precondition!,
                            inner = inner, transport! = transport!,
                            scale! = scale!, add! = add!,
                            isometrictransport = true)

    (ΨL, M, Minv, Ds, ρR, HL, E, e, hL) = x
    return ΨL, ρR, E, e, normgrad, numfg, history
end

function groundstate_diagonal(H::LocalHamiltonian, Ψ₀::UniformCMPS;
                        optalg = ConjugateGradient(; verbosity = 2, gradtol = 1e-7),
                        eigalg = defaulteigalg(Ψ₀),
                        linalg = defaultlinalg(Ψ₀),
                        finalize! = OptimKit._finalize!,
                        kwargs...)

    function retract(x, d, α)
        Ψ, ρL, ρR, = x
        Q = Ψ.Q
        Rs = Ψ.Rs

        dQ, dRs = d

        Rs = Rs .+ α .* dRs
        Q = Q + α * dQ

        Ψ = InfiniteCMPS(Q, Rs)
        ρR, λ, infoR = rightenv(Ψ; eigalg = eigalg, linalg = linalg, kwargs...)

        ρL, ρR, = environments!(Ψ; eigalg = eigalg, linalg = linalg, kwargs...)
        HL, E, e, hL, infoL = leftenv(H, (Ψ,ρL,ρR); eigalg = eigalg, linalg = linalg, kwargs...)
        HR, E, e, hR, infoR = rightenv(H, (Ψ,ρL,ρR); eigalg = eigalg, linalg = linalg, kwargs...)

        dQ = dQ - (λ/α)*one(Q)
        d = dQ, dRs

        if infoR.converged == 0 || infoL.converged == 0
            @warn "step $α : not converged, energy = $e"
            @show infoR
            @show infoL
        end

        return (Ψ, ρL, ρR, HL, HR, E, e, hL, hR), d
    end

    transport!(v, x, d, α, xnew) = v # simplest possible transport

    function inner(x, d1, d2)
        dQ1, dR1 = d1
        dQ2, dR2 = d2
        s = dQ1 === dQ2 ? 2*norm(dQ1)^2 : 2*real(dot(dQ1, dQ2))
        for (dR1,dR2) in zip(dR1, dR2)
            if dR1 === dR2
                s += 2*norm(dR1)^2
            else
                s += 2*real(dot(dR1, dR2))
            end
        end
        return s
    end

    function fg(x)
        Ψ, ρL, ρR, HL, HR, E, _, _, _ = x

        gradQ, gradRs = gradient(H, (Ψ, ρL, ρR), HL, HR; kwargs...)

        gradRs = (Constant(diagm(diag(gradRs[1][]))), Constant(diagm(diag(gradRs[2][]))))

        return E, (gradQ, gradRs)
    end

    function scale!(d, α)
        dQ, dRs = d
        rmul!(dQ, α)
        for dR in dRs
            rmul!(dR, α)
        end
        return d
    end
    function add!(d1, d2, α)
        dQ1, dR1s = d1
        dQ2, dR2s = d2
        axpy!(α, dQ2, dQ1)
        for (dR1, dR2) in zip(dR1s, dR2s)
            axpy!(α, dR2, dR1)
        end
        return d1
    end

    ρL, ρR, λ, infoR = environments!(Ψ₀; kwargs...)

    HL, E, e, hL, infoL = leftenv(H, (Ψ₀,ρL,ρR); kwargs...)
    HR, E, e, hR, infoR = rightenv(H, (Ψ₀,ρL,ρR); kwargs...)
    x = (Ψ₀, ρL, ρR, HL, HR, E, e, hL, hR)

    x, E, normgrad, numfg, history = optimize(fg, x, optalg; retract = retract,
                                inner = inner, transport! = transport!,
                                scale! = scale!, add! = add!,
                                isometrictransport = true)

    (Ψ, ρL, ρR, HL, HR, E, e, hL, hR) = x
    return Ψ, ρL, ρR, E, e, normgrad, numfg, history
end