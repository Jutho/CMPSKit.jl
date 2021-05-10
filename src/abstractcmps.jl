abstract type AbstractCMPS{T,N} end

scalartype(ψ::AbstractCMPS{T}) where T = scalartype(T)

function leftreducedoperator(op::FieldOperator, Ψ::AbstractCMPS, ρL = nothing; kwargs...)
    if isnothing(ρL)
        ρL = leftenv(Ψ; kwargs...)
    end
    _leftreducedoperator(op, Ψ.Q, Ψ.Rs, ρL)
end
function rightreducedoperator(op::FieldOperator, Ψ::AbstractCMPS, ρR = nothing; kwargs...)
    if isnothing(ρR)
        ρR = rightenv(Ψ; kwargs...)
    end
    _rightreducedoperator(op, Ψ.Q, Ψ.Rs, ρR)
end

function _leftreducedoperator(op::FieldOperator, Q, Rs, ρL)
    A, B = _ketbrafactors(op, Q, Rs)
    return B'*(ρL*A)
end

function _rightreducedoperator(op::FieldOperator, Q, Rs, ρR)
    A, B = _ketbrafactors(op, Q, Rs)
    return (A*ρR)*B'
end

function _expval(op::FieldOperator, Q, Rs, ρL, ρR)
    A, B = _ketbrafactors(op, Q, Rs)
    return localdot(ρL*B, A*ρR)
end

function leftreducedoperator(ops::LocalOperator, Ψ::AbstractCMPS, ρL = nothing; kwargs...)
    if isnothing(ρL)
        ρL, = leftenv(Ψ; kwargs...)
    end
    hL = zero(ρL)
    for (coeff, op) in zip(coefficients(ops), operators(ops))
        if coeff isa Number
            axpy!(coeff, leftreducedoperator(op, Ψ, ρL), hL)
        else
            mul!(hL, coeff, leftreducedoperator(op, Ψ, ρL), true, true)
        end
    end
    return hL
end

function rightreducedoperator(ops::LocalOperator, Ψ::AbstractCMPS, ρR = nothing; kwargs...)
    if isnothing(ρR)
        ρR, = rightenv(Ψ; kwargs...)
    end
    hR = zero(ρR)
    for (coeff, op) in zip(coefficients(ops), operators(ops))
        if coeff isa Number
            axpy!(coeff, rightreducedoperator(op, Ψ, ρR), hR)
        else
            mul!(hR, coeff, rightreducedoperator(op, Ψ, ρR), true, true)
        end
    end
    return hR
end

function expval(ops::LocalOperator, Ψ::AbstractCMPS, ρL = nothing, ρR = nothing; kwargs...)
    if isnothing(ρL)
        ρL, = leftenv(Ψ; kwargs...)
    end
    if isnothing(ρR)
        ρR, = rightenv(Ψ; kwargs...)
    end
    (a, b) = domain(Ψ)
    if !isfinite(a)
        a = zero(a)
    end
    if !isfinite(b)
        b = oneunit(b)
    end
    ZL = localdot(ρL, ρR)
    Z = real(ZL(a))
    Zb = ZL(b)
    Z ≈ Zb || @warn "error in computing normalisation: Za = $Z, Zb = $Zb"
    evs = _expval.(operators(ops), Ref(Ψ.Q), Ref(Ψ.Rs), Ref(ρL), Ref(ρR))
    ev = sum(coefficients(ops) .* evs)
    return ev/Z
end

function expval(H::LocalHamiltonian, Ψ::AbstractCMPS, ρL = nothing, ρR = nothing; kwargs...)
    @assert domain(H) == domain(Ψ)
    return integrate(expval(H.h, Ψ, ρL, ρR; kwargs...), domain(H))
end
