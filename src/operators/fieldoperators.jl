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
const ψ = Annihilation{1}()
Base.@pure Base.getindex(::Annihilation{1}, i::Int) = Annihilation{i}()

const Creation{i} = AdjointOperator{Annihilation{i}}

struct DifferentiatedAnnihilation{i} <: FieldOperator end
const ∂ψ = DifferentiatedAnnihilation{1}()
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

localgradientQ(op::FieldOperator, Q, Rs, ρL, ρR) = zero(Q)
localgradientQ(op::DifferentiatedCreation{i}, Q, Rs, ρL, ρR) where {i} =
    -Rs[i]'*(ρL*ρR) + (ρL*ρR)*Rs[i]'
function localgradientQ(op::NormalOrderedTerm{DifferentiatedAnnihilation{i}, <:Any},
                            Q, Rs, ρL, ρR) where {i}
    y = ρL*_ketfactor(op, Q, Rs)*ρR
    return y * Rs[i]' - Rs[i]' * y
end

localgradientRs(op::OnlyAnnihilators, Q, Rs, ρL, ρR) = zero.(Rs)
function localgradientRs(op::Creation{i}, Q, Rs, ρL, ρR) where i
    R̄s = ntuple(length(Rs)) do n
        R̄ = zero(ρL)
        if n == i
            R̄ += ρL*ρR
        end
        R̄
    end
    return R̄s
end
function localgradientRs(op::DifferentiatedCreation{i}, Q, Rs, ρL, ρR) where {i}
    R̄s = ntuple(length(Rs)) do n
        R̄ = zero(ρL)
        if n == i
            f = ρL*ρR
            R̄ += Q'*f - f*Q' - ∂(f)
        end
        R̄
    end
    return R̄s
end
function localgradientRs(op::AdjointOperator{Pairing{i,j}}, Q, Rs, ρL, ρR) where {i,j}
    R̄s = ntuple(length(Rs)) do n
        R̄ = zero(ρL)
        if n == i
            R̄ += ρL*ρR*Rs[j]'
        end
        if n == j
            R̄ += Rs[i]'*ρL*ρR
        end
        R̄
    end
    return R̄s
end

function localgradientRs(op::NormalOrderedTerm{Annihilation{i}, <:Any},
                            Q, Rs, ρL, ρR) where {i}
    f = ρL*_ketfactor(op, Q, Rs)*ρR
    R̄s = ntuple(length(Rs)) do n
        R̄ = zero(f)
        if n == i
            R̄ += f
        end
        R̄
    end
    return R̄s
end
function localgradientRs(op::NormalOrderedTerm{<:DifferentiatedAnnihilation{i}, <:Any},
                            Q, Rs, ρL, ρR) where {i}
    f = ρL*_ketfactor(op, Q, Rs)*ρR
    R̄s = ntuple(length(Rs)) do n
        R̄ = zero(ρL)
        if n == i
            R̄ += Q'*f - f*Q' - ∂(f)
        end
        return R̄
    end
    return R̄s
end
function localgradientRs(op::NormalOrderedTerm{Pairing{i,j}, <:Any},
                            Q, Rs, ρL, ρR) where {i,j}
    f = ρL*_ketfactor(op, Q, Rs)*ρR
    R̄s = ntuple(length(Rs)) do n
        R̄ = zero(ρL)
        if n == i
            R̄ += f*Rs[j]'
        end
        if n == j
            R̄ += Rs[i]'*f
        end
        return R̄
    end
    return R̄s
end
