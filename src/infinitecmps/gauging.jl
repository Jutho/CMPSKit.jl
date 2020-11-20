leftgauge(Ψ::UniformCMPS, args...; kwargs...) = leftgauge!(copy(Ψ), args...; kwargs...)
rightgauge(Ψ::UniformCMPS, args...; kwargs...) = rightgauge!(copy(Ψ), args...; kwargs...)

function leftgauge!(Ψ::UniformCMPS, C₀ = one(Ψ.Q);
                    maxreorth = 10,
                    eigalg = Arnoldi(; krylovdim = min(64, length(Ψ.Q[0]))),
                    kwargs...)

    U = one(Ψ.Q)
    if Ψ.gauge == :l
        η = norm(LeftTransfer(Ψ)(U))
        info = ConvergenceInfo(1, nothing, η, 0, 1)
        return Ψ, zero(scalartype(Ψ)), U, info
    end
    tol = eigalg.tol
    ρL, λ, info = leftenv(Ψ, C₀'*C₀; eigalg = eigalg, kwargs...)
    D, V = eigen!(Hermitian(ρL[]))
    Dsqrt = sqrt.(max.(D, defaulttol(D)))
    _, C = qr!(Diagonal(Dsqrt)*V')
    C ./= norm(C)
    CL = UpperTriangular(C)

    Ψ.Q = Constant(rdiv!(CL*Ψ.Q[], CL))
    Qdiag = view(Ψ.Q[], diagind(Ψ.Q[]))
    Qdiag .-= λ
    Ψ.Rs = map(R->Constant(rdiv!(CL*R[], CL)), Ψ.Rs)
    numreorth = 0
    η = norm(LeftTransfer(Ψ)(U))
    numiter = info.numiter
    numops = info.numops
    while η > tol
        ρL, dλ, info = leftenv(Ψ, U; eigalg = eigalg, kwargs...)
        numiter += info.numiter
        numops += info.numops
        D, V = eigen!(Hermitian(ρL[]))
        Dsqrt = sqrt.(max.(D, defaulttol(D)))
        _, dC = qr!(Diagonal(Dsqrt)*V')
        dCL = UpperTriangular(dC)

        Ψ.Q = Constant(rdiv!(dCL*Ψ.Q[], dCL))
        Qdiag = view(Ψ.Q[], diagind(Ψ.Q[]))
        Qdiag .-= dλ
        Ψ.Rs = map(R->Constant(rdiv!(dCL*R[], dCL)), Ψ.Rs)
        λ += dλ
        C = lmul!(dCL, C)
        C ./= norm(C)
        η = norm(LeftTransfer(Ψ)(U))
        numreorth += 1
        if numreorth == maxreorth
            break
        end
    end
    converged = Int(η < tol)
    if converged > 0
        Ψ.gauge = :l
    end
    return Ψ, λ, Constant(C), ConvergenceInfo(converged, nothing, η, numiter, numops)
end

function rightgauge!(Ψ::UniformCMPS, C₀ = one(Ψ.Q); kwargs...)
    if Ψ.gauge == :r
        U = one(Ψ.Q)
        η = norm(RightTransfer(Ψ)(U))
        info = ConvergenceInfo(1, nothing, η, 0, 1)
        return Ψ, zero(scalartype(Ψ)), U, info
    end
    _adjoint!(Ψ.Q[])
    foreach(Ψ.Rs) do R
        _adjoint!(R[])
    end
    Ψ.gauge = :n
    _, λ, C, info = leftgauge!(Ψ, C₀'; kwargs...)
    _adjoint!(Ψ.Q[])
    foreach(Ψ.Rs) do R
        _adjoint!(R[])
    end
    _adjoint!(C[])
    if Ψ.gauge == :l
        Ψ.gauge = :r
    end
    return Ψ, λ, C, info
end
#
# function leftcanonical!(Ψ::UniformCMPS, C₀ = one(Ψ.Q); maxreorth = 10, kwargs...)
#     Ψ, λ, C, infoL = leftgauge!(Ψ, C₀; maxreorth = maxreorth, kwargs...)
#     ρR, infoR = rightenv(Ψ; kwargs...)
#
