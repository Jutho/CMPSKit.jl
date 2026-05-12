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
Base.:*(o1::AdjointOperator, o2::AdjointOperator) = (o2' * o1')'

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
Base.@pure Base.getindex(::DifferentiatedAnnihilation{1}, i::Int) = DifferentiatedAnnihilation{i}()

const DifferentiatedCreation{i} = AdjointOperator{DifferentiatedAnnihilation{i}}

∂(::Annihilation{i}) where {i} = DifferentiatedAnnihilation{i}()
∂(::Creation{i}) where {i} = DifferentiatedCreation{i}()

struct Pairing{i,j} <: FieldOperator end
Base.:*(::Annihilation{i}, ::Annihilation{j}) where {i,j} = Pairing{i,j}()

Base.:literal_pow(::typeof(^), ::Annihilation{i}, ::Val{2}) where {i} = Pairing{i,i}()
Base.:literal_pow(::typeof(^), ::Creation{i}, ::Val{2}) where {i} = Pairing{i,i}()'

#####################
# NormalOrderedTerm #
#####################
const OnlyAnnihilators = Union{Annihilation,DifferentiatedAnnihilation,Pairing}
struct NormalOrderedTerm{C<:OnlyAnnihilators,A<:OnlyAnnihilators} <: FieldOperator
    creators::C
    annihilators::A
end

const Density{i} = NormalOrderedTerm{Annihilation{i},Annihilation{i}}
const Kinetic{i} = NormalOrderedTerm{DifferentiatedAnnihilation{i},
                                     DifferentiatedAnnihilation{i}}
const ContactInteraction{i,j,k,l} = NormalOrderedTerm{Pairing{i,j},Pairing{k,l}}
# ψ[l]'*ψ[k]'*ψ[i]*ψ[j]

function Base.:*(c::AdjointOperator{<:OnlyAnnihilators}, a::OnlyAnnihilators)
    return NormalOrderedTerm(c', a)
end

function Base.:*(op1::NormalOrderedTerm, op2::OnlyAnnihilators)
    return NormalOrderedTerm(op1.creators, op1.annihilators * op2)
end

function Base.:*(op1::AdjointOperator{<:OnlyAnnihilators}, op2::NormalOrderedTerm)
    return NormalOrderedTerm(op2.creators * op1', op2.annihilators)
end

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
_ketfactor(op::Pairing{i,j}, Q, Rs) where {i,j} = Rs[i] * Rs[j]
_ketfactor(op::AdjointOperator{<:OnlyAnnihilators}, Q, Rs) = one(Q)
_ketfactor(op::NormalOrderedTerm, Q, Rs) = _ketfactor(op.annihilators, Q, Rs)

_brafactor(op::FieldOperator, Q, Rs) = _ketfactor(op', Q, Rs)

_ketbrafactors(op::FieldOperator, Q, Rs) = (_ketfactor(op, Q, Rs), _brafactor(op, Q, Rs))
function _ketbrafactors(op::NormalOrderedTerm{A,A}, Q, Rs) where {A}
    x = _ketfactor(op, Q, Rs)
    return (x, x)
end

# Tangents: contribution to ket factors, used for excitations. To support topologically
# non-trivial excitations and centrally gauged excitations, we can have different cMPS
# matrices to the left and right
function _ketfactor_tangent(op::Annihilation{i}, QL, RLs, V, Ws, ∂Ws, QR=QL,
                            RRs=RLs) where {i}
    return Ws[i]
end
function _ketfactor_tangent(op::DifferentiatedAnnihilation{i}, QL, RLs, V, Ws, ∂Ws, QR=QL,
                            RRs=RLs) where {i}
    RL = RLs[i]
    RR = RRs[i]
    W = Ws[i]
    𝒟W = copy(∂Ws[i])
    mul!(𝒟W, QL, W, 1, 1)
    mul!(𝒟W, V, RR, 1, 1)
    mul!(𝒟W, W, QR, -1, 1)
    mul!(𝒟W, RL, V, -1, 1)
    return 𝒟W
end
function _ketfactor_tangent(op::Pairing{i,j}, QL, RLs, V, Ws, ∂Ws, QR=QL,
                            RRs=RLs) where {i,j}
    return mul!(Ws[i] * RRs[j], RLs[i], Ws[j], true, true)
end
function _ketfactor_tangent(op::AdjointOperator{<:OnlyAnnihilators}, QL, RLs, V, Ws, ∂Ws,
                            QR=QL, RRs=RLs)
    return zero(QL)
end
function _ketfactor_tangent(op::NormalOrderedTerm, QL, RLs, V, Ws, ∂Ws, QR=QL, RRs=RLs)
    return _ketfactor_tangent(op.annihilators, QL, RLs, V, Ws, ∂Ws, QR, RRs)
end

# Cotangents: contribution to the gradient of the partial derivatives with respect
# to Q, R, and ∂R (the last two are treated as being independent). Also used for excitations,
# with the possibility of having different cMPS matrices left and right.

# If the energy takes the form `e = tr(brafactor(op)' * y), then `_brafactor_cotangentX`
# returns a function such that `δe = tr(δX' * _brafactor_cotangentX(y))` for `X` equal to
# `Q`, `Rs` or `∂Rs`.

# For finite and infinite cMPS, `y` will always be `ρL*ketfactors(op)*ρR`.
_brafactor_cotangentRs(op::OnlyAnnihilators, QL, RLs, QR=QL, RRs=RLs) = y -> zero.(RLs)
function _brafactor_cotangentRs(op::Creation{i}, QL, RLs, QR=QL, RRs=RLs) where {i}
    return function (y)
        R̄s = ntuple(length(RLs)) do n
            return R̄ = n == i ? y : zero(y)
        end
        return R̄s
    end
end
function _brafactor_cotangentRs(op::AdjointOperator{Pairing{i,j}}, QL, RLs, QR=QL,
                                RRs=RLs) where {i,j}
    return function (y)
        R̄s = ntuple(length(RLs)) do n
            R̄ = zero(y)
            if n == i
                R̄ += y * RRs[j]'
            end
            if n == j
                R̄ += RLs[i]' * y
            end
            return R̄
        end
        return R̄s
    end
end
function _brafactor_cotangentRs(op::NormalOrderedTerm{Annihilation{i},<:Any}, QL, RLs,
                                QR=QL, RRs=RLs) where {i}
    return function (y)
        R̄s = ntuple(length(RLs)) do n
            return R̄ = n == i ? y : zero(y)
        end
        return R̄s
    end
end

const ContainsDifferentiatedCreation{i} = Union{DifferentiatedCreation{i},
                                                NormalOrderedTerm{DifferentiatedAnnihilation{i},
                                                                  <:Any}}

# the following assume ∂R is independent, and only computes the gradient to R
function _brafactor_cotangentRs(op::ContainsDifferentiatedCreation{i}, QL, RLs, QR=QL,
                                RRs=RLs) where {i}
    return function (y)
        R̄s = ntuple(length(RLs)) do n
            return R̄ = n == i ? QL' * y - y * QR' : zero(y)
        end
        return R̄s
    end
end

function _brafactor_cotangentRs(op::NormalOrderedTerm{Pairing{i,j},<:Any}, QL, RLs, QR=QL,
                                RRs=RLs) where {i,j}
    return function (y)
        R̄s = ntuple(length(RLs)) do n
            R̄ = zero(RLs[n])
            if n == i
                R̄ += y * RRs[j]'
            end
            if n == j
                R̄ += RLs[i]' * y
            end
            return R̄
        end
        return R̄s
    end
end

_brafactor_cotangentQ(op::FieldOperator, QL, RLs, QR=QL, RRs=RLs) = y -> zero(QL)
function _brafactor_cotangentQ(op::ContainsDifferentiatedCreation{i}, QL, RLs, QR=QL,
                               RRs=RLs) where {i}
    return function (y)
        return -RLs[i]' * y + y * RRs[i]'
    end
end

_brafactor_cotangent∂Rs(op::FieldOperator, QL, RLs, QR=QL, RRs=RLs) = y -> zero.(RLs)
function _brafactor_cotangent∂Rs(op::ContainsDifferentiatedCreation{i}, QL, RLs, QR=QL,
                                 RRs=RLs) where {i}
    return function (y)
        ∂R̄s = ntuple(length(RLs)) do n
            return R̄ = n == i ? y : zero(y)
        end
        return ∂R̄s
    end
end
