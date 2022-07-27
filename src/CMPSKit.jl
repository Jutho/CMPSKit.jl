module CMPSKit

export virtualdim
export leftenv, rightenv, leftenv!, rightenv!, environments!
export leftgauge, rightgauge, leftgauge!, rightgauge!,
        leftcanonical, rightcanonical, leftcanonical!, rightcanonical!
export InfiniteCMPS, FiniteCMPS, CircularCMPS, LeftTransfer, RightTransfer
export Constant, FourierSeries, TaylorSeries, PiecewiseLinear
export fit, differentiate, integrate, localdot, domain, period, nummodes, density
export leftreducedoperator, rightreducedoperator, expval, gradient, groundstate
export groundstate2

export norm, normalize, normalize!, dot, isapprox, tr, partialtrace1, partialtrace2
export ⊗

export ∂, ∫, ψ̂, ∂ψ̂, ψ̂₁, ψ̂₂, ψ̂₃, ∂ψ̂₁, ∂ψ̂₂, ∂ψ̂₃

using LinearAlgebra
using KrylovKit
using KrylovKit: ConvergenceInfo
using OptimKit

scalartype(x::Any) = scalartype(typeof(x))
scalartype(T::Type{<:Number}) = T
scalartype(::Type{<:AbstractArray{T}}) where T = scalartype(T)

defaulttol(x::Any) = eps(real(float(one(scalartype(x)))))^(2/3)

isscalar(x) = false
isscalar(x::Number) = true

include("common/regularisedinverse.jl")
include("common/adjoint.jl")
include("common/addkronecker.jl")
include("common/partialtrace.jl")
include("common/exp.jl")
include("common/exp_lazy.jl")

include("functionspaces/generic.jl")
include("functionspaces/constant.jl")
include("functionspaces/fourierseries.jl")
include("functionspaces/taylorseries.jl")
include("functionspaces/piecewise.jl")
include("functionspaces/piecewiselinear.jl")

const MatrixFunction{T} = FunctionSpace{<:AbstractMatrix{T}}
const ScalarFunction = Union{<:Number, FunctionSpace{<:Number}}

include("operators/fieldoperators.jl")
include("operators/sumofterms.jl")
include("operators/hamiltonians.jl")

include("abstractcmps.jl")
include("transfer.jl")

include("finitecmps/finitecmps.jl")
include("finitecmps/transfer.jl")
include("finitecmps/environments.jl")
include("finitecmps/tangent.jl")
include("finitecmps/gradients.jl")
include("finitecmps/groundstate.jl")

include("infinitecmps/infinitecmps.jl")
include("infinitecmps/environments.jl")
include("infinitecmps/gauging.jl")
include("infinitecmps/gradients.jl")
include("infinitecmps/groundstate.jl")

include("circularcmps/circularcmps.jl")
include("circularcmps/gauge.jl")
include("circularcmps/environments.jl")
include("circularcmps/gradients.jl")
include("circularcmps/groundstate.jl")

# Deprecations
Base.@deprecate_binding ψ ψ̂
Base.@deprecate_binding ∂ψ ∂ψ̂

end
