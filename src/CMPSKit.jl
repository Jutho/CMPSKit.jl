module CMPSKit

export expval, average_energy, energy_density

export virtualdim
export leftenv, rightenv, leftenv!, rightenv!, environments!
export leftgauge, rightgauge, leftgauge!, rightgauge!,
        leftcanonical, rightcanonical, leftcanonical!, rightcanonical!
export InfiniteCMPS, FiniteCMPS, LeftTransfer, RightTransfer
export Constant, FourierSeries, TaylorSeries, PiecewiseLinear
export fit, differentiate, integrate, localdot, domain, period, nummodes, density
export leftreducedoperator, rightreducedoperator, expval, gradient, groundstate

export ψ, ∂ψ, ∂, ∫

using LinearAlgebra
using KrylovKit
using KrylovKit: ConvergenceInfo
using OptimKit

scalartype(x::Any) = scalartype(typeof(x))
scalartype(T::Type{<:Number}) = T
scalartype(::Type{<:AbstractArray{T}}) where T = T

isscalar(x) = false
isscalar(x::Number) = true

include("common/regularisedinverse.jl")
include("common/adjoint.jl")

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

include("finitecmps/finitecmps.jl")
include("finitecmps/transfer.jl")
include("finitecmps/environments.jl")

include("infinitecmps/infinitecmps.jl")
include("infinitecmps/transfer.jl")
include("infinitecmps/environments.jl")
include("infinitecmps/gauging.jl")
include("infinitecmps/gradients.jl")
include("infinitecmps/groundstate.jl")

end
