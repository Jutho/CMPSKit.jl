abstract type AbstractCMPS{T,N} end

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
            hL += coeff * leftreducedoperator(op, Ψ, ρL)
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
            axpy!(coeff, rightreducedoperator(op, Ψ, ρR), hL)
        else
            hL += coeff * rightreducedoperator(op, Ψ, ρR)
        end
    end
    return hL
end

function expval(ops::LocalOperator, Ψ::AbstractCMPS, ρL = nothing, ρR = nothing; kwargs...)
    if isnothing(ρL)
        ρL, = leftenv(Ψ; kwargs...)
    end
    if isnothing(ρR)
        ρR, = rightenv(Ψ; kwargs...)
    end
    Z = real(dot(ρL, ρR))
    ev = sum(coeff*_expval(op, Ψ.Q, Ψ.Rs, ρL, ρR)
                for (coeff, op) in zip(coefficients(ops), operators(ops)))
    return ev/Z
end
