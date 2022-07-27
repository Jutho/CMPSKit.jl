abstract type LocalOperator end
abstract type FieldOperator <: LocalOperator end

coefficients(op::FieldOperator) = (1,)
operators(op::FieldOperator) = (op,)

###################
# AdjointOperator #
###################
struct AdjointOperator{O<:FieldOperator} <: FieldOperator
    op::O
end
AdjointOperator{O}() where {O<:FieldOperator} = adjoint(O())

Base.adjoint(o::FieldOperator) = AdjointOperator(o)
Base.adjoint(o::AdjointOperator) = o.op
Base.:*(o1::AdjointOperator, o2::AdjointOperator) = (o2'*o1')'

################
# Single terms #
################
struct Annihilation{i} <: FieldOperator end
const ψ̂ = Annihilation{1}()
const ψ̂₁ = Annihilation{1}()
const ψ̂₂ = Annihilation{2}()
const ψ̂₃ = Annihilation{3}()
Base.@pure Base.getindex(::Annihilation{1}, i::Int) = Annihilation{i}()

const Creation{i} = AdjointOperator{Annihilation{i}}

struct DifferentiatedAnnihilation{i} <: FieldOperator end
const ∂ψ̂ = DifferentiatedAnnihilation{1}()
const ∂ψ̂₁ = DifferentiatedAnnihilation{1}()
const ∂ψ̂₂ = DifferentiatedAnnihilation{2}()
const ∂ψ̂₃ = DifferentiatedAnnihilation{3}()
Base.@pure Base.getindex(::DifferentiatedAnnihilation{1}, i::Int) =
    DifferentiatedAnnihilation{i}()

const DifferentiatedCreation{i} = AdjointOperator{DifferentiatedAnnihilation{i}}

∂(::Annihilation{i}) where i = DifferentiatedAnnihilation{i}()
∂(::Creation{i}) where i = DifferentiatedCreation{i}()

struct Pairing{i,j} <: FieldOperator end
Base.:*(::Annihilation{i}, ::Annihilation{j}) where {i,j} = Pairing{i,j}()

Base.:literal_pow(::typeof(^), ::Annihilation{i}, ::Val{2}) where {i} = Pairing{i,i}()
Base.:literal_pow(::typeof(^), ::Creation{i}, ::Val{2}) where {i} = Pairing{i,i}()'

#####################
# NormalOrderedTerm #
#####################
const OnlyAnnihilators = Union{Annihilation, DifferentiatedAnnihilation, Pairing}
struct NormalOrderedTerm{C<:OnlyAnnihilators, A<:OnlyAnnihilators} <: FieldOperator
    creators::C
    annihilators::A
end

const Density{i} = NormalOrderedTerm{ Annihilation{i}, Annihilation{i} }
const Kinetic{i} = NormalOrderedTerm{ DifferentiatedAnnihilation{i},
                                        DifferentiatedAnnihilation{i} }
const ContactInteraction{i,j,k,l} = NormalOrderedTerm{ Pairing{i,j}, Pairing{k,l} }
# ψ[l]'*ψ[k]'*ψ[i]*ψ[j]

Base.:*(c::AdjointOperator{<:OnlyAnnihilators}, a::OnlyAnnihilators) =
    NormalOrderedTerm(c', a)

Base.:*(op1::NormalOrderedTerm, op2::OnlyAnnihilators) =
    NormalOrderedTerm(op1.creators, op1.annihilators*op2)

Base.:*(op1::AdjointOperator{<:OnlyAnnihilators}, op2::NormalOrderedTerm) =
    NormalOrderedTerm(op2.creators*op1', op2.annihilators)

Base.adjoint(o::NormalOrderedTerm) = NormalOrderedTerm(o.annihilators, o.creators)

# the factors that this operator brings down in a cMPS ket or bra
_ketfactor(op::Annihilation{i}, Q, Rs) where {i} = Rs[i]
function _ketfactor(op::DifferentiatedAnnihilation{i}, Q, Rs) where {i}
    R = Rs[i]
    𝒟R = ∂(R)
    mul!(𝒟R, Q, R, 1, 1)
    mul!(𝒟R, R, Q, -1, 1)
    return 𝒟R
end
_ketfactor(op::Pairing{i,j}, Q, Rs) where {i,j} = Rs[i]*Rs[j]
_brafactor(op::OnlyAnnihilators, Q, Rs) = one(Q)

_ketfactor(op::AdjointOperator, Q, Rs) = _brafactor(op', Q, Rs)
_brafactor(op::AdjointOperator, Q, Rs) = _ketfactor(op', Q, Rs)

_ketfactor(op::NormalOrderedTerm, Q, Rs) = _ketfactor(op.annihilators, Q, Rs)
_brafactor(op::NormalOrderedTerm, Q, Rs) = _ketfactor(op.creators, Q, Rs)

_ketbrafactors(op::FieldOperator, Q, Rs) = (_ketfactor(op, Q, Rs), _brafactor(op, Q, Rs))
function _ketbrafactors(op::NormalOrderedTerm{A,A}, Q, Rs) where A
    x = _ketfactor(op, Q, Rs)
    return (x, x)
end

# Local gradients: contribution to the gradient of the partial derivatives with respect
# to Q, R, and ∂R (the last two are treated as being independent)

# If the energy takes the form `e = tr(brafactor(op)' * y), then `_localgradientX` returns
# a function such that `δe = tr(δX' * _localgradientX(y))` for `X` equal to `R`, `Q` or `∂R`.

# For finite and infinite cMPS, `y` will always be `ρL*ketfactors(op)*ρR`.
_localgradientRs(op::OnlyAnnihilators, Q, Rs) = y->zero.(Rs)
function _localgradientRs(op::Creation{i}, Q, Rs) where i
    return function (y)
        R̄s = ntuple(length(Rs)) do n
            R̄ = n == i ? y : zero(y)
        end
        return R̄s
    end
end
function _localgradientRs(op::AdjointOperator{Pairing{i,j}}, Q, Rs) where {i,j}
    return function (y)
        R̄s = ntuple(length(Rs)) do n
            R̄ = zero(y)
            if n == i
                R̄ += y*Rs[j]'
            end
            if n == j
                R̄ += Rs[i]'*y
            end
            R̄
        end
        return R̄s
    end
end
function _localgradientRs(op::NormalOrderedTerm{Annihilation{i}, <:Any}, Q, Rs) where {i}
    return function (y)
        R̄s = ntuple(length(Rs)) do n
            R̄ = n == i ? y : zero(y)
        end
        return R̄s
    end
end

const ContainsDifferentiatedCreation{i} =
    Union{DifferentiatedCreation{i}, NormalOrderedTerm{DifferentiatedAnnihilation{i},<:Any}}

# the following assume ∂R is independent, and only computes the gradient to R
function _localgradientRs(op::ContainsDifferentiatedCreation{i}, Q, Rs) where {i}
    return function (y)
        R̄s = ntuple(length(Rs)) do n
            R̄ = n == i ? Q'*y - y*Q' : zero(y)
        end
        return R̄s
    end
end

function _localgradientRs(op::NormalOrderedTerm{Pairing{i,j}, <:Any}, Q, Rs) where {i,j}
    return function (y)
        R̄s = ntuple(length(Rs)) do n
            R̄ = zero(Rs[n])
            if n == i
                R̄ += y*Rs[j]'
            end
            if n == j
                R̄ += Rs[i]'*y
            end
            return R̄
        end
        return R̄s
    end
end

_localgradientQ(op::FieldOperator, Q, Rs) = y -> zero(Q)
function _localgradientQ(op::ContainsDifferentiatedCreation{i}, Q, Rs) where {i}
    return function (y)
        -Rs[i]'*y + y*Rs[i]'
    end
end

_localgradient∂Rs(op::FieldOperator, Q, Rs) = y -> zero.(Rs)
function _localgradient∂Rs(op::ContainsDifferentiatedCreation{i}, Q, Rs) where i
    return function (y)
        ∂R̄s = ntuple(length(Rs)) do n
            R̄ = n == i ? y : zero(y)
        end
        return ∂R̄s
    end
end
