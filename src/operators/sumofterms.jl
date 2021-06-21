##############
# SumOfTerms #
##############

struct SumOfLocalTerms{N, T1<:NTuple{N,ScalarFunction},
                            T2<:NTuple{N,LocalOperator}} <: LocalOperator
    coefficients::T1
    operators::T2
end

operators(op::SumOfLocalTerms) = op.operators
coefficients(op::SumOfLocalTerms) = op.coefficients

Base.:*(α::ScalarFunction, op::LocalOperator) = SumOfLocalTerms((α,), (op,))
Base.:*(op::LocalOperator, α::ScalarFunction) = *(α, op)

Base.:*(α::ScalarFunction, op::SumOfLocalTerms) =
    SumOfLocalTerms(α .* op.coefficients, op.operators)
Base.:*(op::SumOfLocalTerms, α::ScalarFunction) = *(α, op)

Base.:-(op::LocalOperator) = SumOfLocalTerms((-1,), (op,))
Base.:+(op::LocalOperator) = SumOfLocalTerms((+1,), (op,))
Base.:-(op::SumOfLocalTerms) = SumOfLocalTerms(.-(op.coefficients), op.operators)
Base.:+(op::SumOfLocalTerms) = SumOfLocalTerms(.+(op.coefficients), op.operators)

Base.:+(op1::LocalOperator, op2::LocalOperator) = SumOfLocalTerms((1,1), (op1, op2))
Base.:-(op1::LocalOperator, op2::LocalOperator) = SumOfLocalTerms((1,-1), (op1, op2))
Base.:+(op1::SumOfLocalTerms, op2::LocalOperator) =
    SumOfLocalTerms((op1.coefficients..., 1), (op1.operators..., op2))
Base.:-(op1::SumOfLocalTerms, op2::LocalOperator) =
    SumOfLocalTerms((op1.coefficients..., -1), (op1.operators..., op2))
Base.:+(op1::LocalOperator, op2::SumOfLocalTerms) =
    SumOfLocalTerms((1, op2.coefficients...), (op1, op2.operators...))
Base.:-(op1::LocalOperator, op2::SumOfLocalTerms) = +(op1, -(op2))
Base.:+(op1::SumOfLocalTerms, op2::SumOfLocalTerms) =
    SumOfLocalTerms((op1.coefficients..., op2.coefficients...),
                    (op1.operators..., op2.operators...))
Base.:-(op1::SumOfLocalTerms, op2::SumOfLocalTerms) = +(op1, -(op2))

Base.:*(op1::SumOfLocalTerms, op2::LocalOperator) =
    SumOfLocalTerms(op1.coefficients, op1.operators .* (op2,))
Base.:*(op1::LocalOperator, op2::SumOfLocalTerms) =
    SumOfLocalTerms(op2.coefficients, (op1,) .* op2.operators)

Base.:*(op1::SumOfLocalTerms{1}, op2::SumOfLocalTerms) =
    op1.coefficients[1] * (op1.operators[1] * op2)

function Base.:*(op1::SumOfLocalTerms, op2::SumOfLocalTerms)
    op1.coefficients[1] * (op1.operators[1] * op2) +
    SumOfLocalTerms(Base.tail(op1.coefficients), Base.tail(op1.operators)) * op2
end

Base.adjoint(op::SumOfLocalTerms) =
    SumOfLocalTerms(adjoint.(op.coefficients), adjoint.(op.operators))
