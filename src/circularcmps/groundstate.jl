using Printf

# groundstate with CircularCMPS{<:Constant}
groundstate(Ĥ::LocalHamiltonian, Ψ₀::UniformCircularCMPS; kwargs...) =
    groundstate_unconstrained(Ĥ, Ψ₀; kwargs...)

groundstate(Ĥ::LocalHamiltonian, Ψ₀::UniformCircularCMPS, n₀::Number; kwargs...) =
    groundstate_constrained(Ĥ, Ψ₀, ntuple(k->n₀, Val(length(Ψ₀.Rs))); kwargs...)

groundstate(Ĥ::LocalHamiltonian,
            Ψ₀::UniformCircularCMPS{<:AbstractMatrix, N},
            n₀s::NTuple{N, <:Number}; kwargs...) where {N} =
    groundstate_constrained(Ĥ, Ψ₀, n₀s; kwargs...)

function groundstate_unconstrained(Ĥ::LocalHamiltonian, Ψ₀::UniformCircularCMPS;
                                    gradtol = 1e-7,
                                    verbosity = 2,
                                    optalg = LBGFS(20; gradtol = gradtol, verbosity = verbosity - 2),
                                    # eigalg = defaulteigalg(Ψ₀),
                                    # linalg = defaultlinalg(Ψ₀),
                                    finalize! = OptimKit._finalize!,
                                    kwargs...)

    δ = 1
    function retract(x, d, α)
        ΨL, ρR = x
        QL = ΨL.Q
        RLs = ΨL.Rs

        dRs = d
        RdR = zero(QL)
        dRdR = zero(QL)
        for (R, dR) in zip(RLs, dRs)
            mul!(RdR, R', dR, true, true)
            mul!(dRdR, dR', dR, true, true)
        end

        RLs = RLs .+ α .* dRs
        QL = QL - α * RdR - α^2/2 * dRdR

        ΨL = normalize!(CircularCMPS(QL, RLs, period(ΨL)))
        ρR, = rightenv(InfiniteCMPS(ΨL.Q, ΨL.Rs; gauge=:l), ρR)
        x = (ΨL, ρR)
        return x, d
    end

    transport!(v, x, d, α, xnew) = v # simplest possible transport

    function inner(x, d1, d2)
        return 2*real(sum(dot.(d1, d2)))
    end

    function precondition(x, d)
        ΨL, ρR = x
        dRs = d
        return dRs .* Ref(posreginv(ρR[0], δ))
    end

    function fg(x)
        ΨL, = x
        𝔼H, 𝔼 = environment(Ĥ, ΨL)
        ℰ = real(expval(Ĥ, ΨL, 𝔼))
        gradQ, gradRs = gradient(Ĥ, ΨL, 𝔼, 𝔼H)

        Rs = ΨL.Rs
        dRs = .-(Rs) .* Ref(gradQ) .+ gradRs

        return ℰ, dRs
    end

    scale!(d, α) = rmul!.(d, α)
    add!(d1, d2, α) = axpy!.(α, d2, d1)

    function _finalize!(x, E, d, numiter)
        normgrad2 = inner(x, d, d)
        δ = max(1e-12, 1e-3*normgrad2)
        normgrad = sqrt(normgrad2)
        verbosity > 1 &&
            @info @sprintf("CircularCMPS ground state: iter %4d: E = %.12f, ‖∇E‖ = %.4e",
                                numiter, E, normgrad)
        return finalize!(x, E, d, numiter)
    end

    ΨL, = leftgauge(Ψ₀; kwargs...)
    ΨL = normalize!(ΨL)
    ℰ = real(expval(Ĥ, ΨL))
    ρR, = rightenv(InfiniteCMPS(ΨL.Q, ΨL.Rs; gauge=:l))
    x = (ΨL, ρR)

    verbosity > 0 &&
        @info @sprintf("CircularCMPS ground state: initialization with ℰ = %.12f", ℰ)

    x, ℰ, grad, numfg, history =
        optimize(fg, x, optalg; retract = retract,
                                precondition = precondition,
                                finalize! = _finalize!,
                                inner = inner, transport! = transport!,
                                scale! = scale!, add! = add!,
                                isometrictransport = true)
    ΨL, = x
    normgrad = sqrt(inner(x, grad, grad))
    if verbosity > 0
        if normgrad <= gradtol
            @info @sprintf("CircularCMPS ground state: converged after %d iterations: e = %.12f, ‖∇e‖ = %.4e",
                            size(history, 1), ℰ, normgrad)
        else
            @warn @sprintf("CircularCMPS ground state: not converged to requested tol: e = %.12f, ‖∇e‖ = %.4e",
                            ℰ, normgrad)
        end
    end
    return ΨL, ℰ, normgrad, numfg, history
end


# EXPERIMENTAL STUFF
#--------------------

# groundstate2: no preconditioning
function groundstate2(Ĥ::LocalHamiltonian, Ψ₀::UniformCircularCMPS;
                                    gradtol = 1e-7,
                                    verbosity = 2,
                                    optalg = LBGFS(20; gradtol = gradtol, verbosity = verbosity - 2),
                                    # eigalg = defaulteigalg(Ψ₀),
                                    # linalg = defaultlinalg(Ψ₀),
                                    finalize! = OptimKit._finalize!,
                                    kwargs...)

    function retract(x, d, α)
        ΨL = x
        QL = ΨL.Q
        RLs = ΨL.Rs

        dRs = d
        RdR = zero(QL)
        dRdR = zero(QL)
        for (R, dR) in zip(RLs, dRs)
            mul!(RdR, R', dR, true, true)
            mul!(dRdR, dR', dR, true, true)
        end

        RLs = RLs .+ α .* dRs
        QL = QL - α * RdR - α^2/2 * dRdR

        ΨL = normalize!(CircularCMPS(QL, RLs, period(ΨL)))
        return ΨL, d
    end

    transport!(v, x, d, α, xnew) = v # simplest possible transport

    function inner(x, d1, d2)
        return 2*real(sum(dot.(d1, d2)))
    end

    # function precondition(x, d)
    #     ΨL, ρR, = x
    #     dRs = d
    #     return dRs .* Ref(posreginv(ρR[0], δ))
    # end

    function fg(x)
        ΨL = x
        𝔼H, 𝔼 = environment(Ĥ, ΨL)
        ℰ = real(expval(Ĥ, ΨL, 𝔼))
        gradQ, gradRs = gradient(Ĥ, ΨL, 𝔼, 𝔼H)

        Rs = ΨL.Rs
        dRs = .-(Rs) .* Ref(gradQ) .+ gradRs

        return ℰ, dRs
    end

    scale!(d, α) = rmul!.(d, α)
    add!(d1, d2, α) = axpy!.(α, d2, d1)

    function _finalize!(x, E, d, numiter)
        normgrad2 = inner(x, d, d)
        normgrad = sqrt(normgrad2)
        verbosity > 1 &&
            @info @sprintf("CircularCMPS ground state: iter %4d: E = %.12f, ‖∇E‖ = %.4e",
                                numiter, E, normgrad)
        return finalize!(x, E, d, numiter)
    end


    ΨL, = leftgauge(Ψ₀; kwargs...)
    ΨL = normalize!(ΨL)
    ℰ = real(expval(Ĥ, ΨL))
    x = ΨL
    _, d = fg(ΨL)

    verbosity > 0 &&
        @info @sprintf("CircularCMPS ground state: initialization with ℰ = %.12f", ℰ)

    x, ℰ, grad, numfg, history =
        optimize(fg, x, optalg; retract = retract,
                                finalize! = _finalize!,
                                inner = inner, transport! = transport!,
                                scale! = scale!, add! = add!,
                                isometrictransport = true)
    ΨL = x
    normgrad = sqrt(inner(x, grad, grad))
    if verbosity > 0
        if normgrad <= gradtol
            @info @sprintf("CircularCMPS ground state: converged after %d iterations: e = %.12f, ‖∇e‖ = %.4e",
                            size(history, 1), ℰ, normgrad)
        else
            @warn @sprintf("CircularCMPS ground state: not converged to requested tol: e = %.12f, ‖∇e‖ = %.4e",
                            ℰ, normgrad)
        end
    end
    return ΨL, ℰ, normgrad, numfg, history
end

# groundstate3: preconditioning with tangent_pace_metric
function groundstate3(Ĥ::LocalHamiltonian, Ψ₀::UniformCircularCMPS;
                                    gradtol = 1e-7,
                                    verbosity = 2,
                                    optalg = LBFGS(20; gradtol = gradtol, verbosity = verbosity - 2),
                                    # eigalg = defaulteigalg(Ψ₀),
                                    # linalg = defaultlinalg(Ψ₀),
                                    finalize! = OptimKit._finalize!,
                                    kwargs...)

    δ = 1
    function retract(x, d, α)
        ΨL = x
        QL = ΨL.Q
        RLs = ΨL.Rs

        dRs = d
        RdR = zero(QL)
        dRdR = zero(QL)
        for (R, dR) in zip(RLs, dRs)
            mul!(RdR, R', dR, true, true)
            mul!(dRdR, dR', dR, true, true)
        end

        RLs = RLs .+ α .* dRs
        QL = QL - α * RdR - α^2/2 * dRdR

        ΨL = normalize!(CircularCMPS(QL, RLs, period(ΨL)))
        x = ΨL
        return x, d
    end

    transport!(v, x, d, α, xnew) = v # simplest possible transport

    function inner(x, d1, d2)
        return 2*real(sum(dot.(d1, d2)))
    end

    function precondition(x, d)
        ΨL = x
        dRs = d
        Rs = ΨL.Rs
        metric = tangent_space_metric(ΨL)
        rhs = KrylovKit.RecursiveVec(dRs)
        v₀ = KrylovKit.RecursiveVec(zero.(dRs))
        # zero initialisation is important to make CG iterates descent directions
        η = min(0.1, sqrt(norm(rhs)))
        pdRs, info = linsolve(rhs, v₀, KrylovKit.CG(; maxiter = 500, tol = η*norm(rhs), verbosity = 0)) do v
            Ws = (v...,)
            V = -sum(adjoint.(Rs) .* Ws)
            GV, GWs = metric(V, Ws)
            GWs = mul!.(GWs, Rs, Ref(GV), -1, +1)
            GWs = axpy!.(δ, Ws, GWs)
            return KrylovKit.RecursiveVec(GWs)
        end
        @show real(dot(pdRs, rhs)/norm(pdRs)/norm(rhs))
        if info.converged == 0
            @warn "Not converged"
        end
        return (pdRs...,)
    end

    function fg(x)
        ΨL = x
        𝔼H, 𝔼 = environment(Ĥ, ΨL)
        ℰ = real(expval(Ĥ, ΨL, 𝔼))
        gradQ, gradRs = gradient(Ĥ, ΨL, 𝔼, 𝔼H)

        Rs = ΨL.Rs
        dRs = .-(Rs) .* Ref(gradQ) .+ gradRs

        return ℰ, dRs
    end

    scale!(d, α) = rmul!.(d, α)
    add!(d1, d2, α) = axpy!.(α, d2, d1)

    function _finalize!(x, E, d, numiter)
        normgrad2 = inner(x, d, d)
        δ = max(1e-12, 1e-3*normgrad2)
        normgrad = sqrt(normgrad2)
        verbosity > 1 &&
            @info @sprintf("CircularCMPS ground state: iter %4d: E = %.12f, ‖∇E‖ = %.4e",
                                numiter, E, normgrad)
        return finalize!(x, E, d, numiter)
    end

    ΨL, = leftgauge(Ψ₀; kwargs...)
    ΨL = normalize!(ΨL)
    ℰ = real(expval(Ĥ, ΨL))
    x = ΨL

    verbosity > 0 &&
        @info @sprintf("CircularCMPS ground state: initialization with ℰ = %.12f", ℰ)

    x, ℰ, grad, numfg, history =
        optimize(fg, x, optalg; retract = retract,
                                precondition = precondition,
                                finalize! = _finalize!,
                                inner = inner, transport! = transport!,
                                scale! = scale!, add! = add!,
                                isometrictransport = true)
    ΨL = x
    normgrad = sqrt(inner(x, grad, grad))
    if verbosity > 0
        if normgrad <= gradtol
            @info @sprintf("CircularCMPS ground state: converged after %d iterations: e = %.12f, ‖∇e‖ = %.4e",
                            size(history, 1), ℰ, normgrad)
        else
            @warn @sprintf("CircularCMPS ground state: not converged to requested tol: e = %.12f, ‖∇e‖ = %.4e",
                            ℰ, normgrad)
        end
    end
    return ΨL, ℰ, normgrad, numfg, history
end
