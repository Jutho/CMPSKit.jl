module CMPSKit

export expval, average_energy, energy_density

export virtualdim
export leftenv, rightenv, leftenv!, rightenv!, environments!
export leftgauge, rightgauge, leftgauge!, rightgauge!,
        leftcanonical, rightcanonical, leftcanonical!, rightcanonical!
export InfiniteCMPS, LeftTransfer, RightTransfer
export Constant, FourierSeries
export fit, differentiate, integrate, ∂, ∫, localdot, domain, period, nummodes, density
export leftreducedoperator, rightreducedoperator, expval, gradient, groundstate

export ψ, ∂ψ

using LinearAlgebra
using KrylovKit
using KrylovKit: ConvergenceInfo
using OptimKit

scalartype(x::Any) = scalartype(typeof(x))
scalartype(T::Type{<:Number}) = T
scalartype(::Type{<:AbstractMatrix{T}}) where T = T

isscalar(x) = false
isscalar(x::Number) = true

include("common/regularisedinverse.jl")
include("common/adjoint.jl")

include("functionspaces/functionspace.jl")
include("functionspaces/fourierseries.jl")
include("functionspaces/constant.jl")

const MatrixFunction{T} = FunctionSpace{<:AbstractMatrix{T}}
const ScalarFunction = Union{<:Number, FunctionSpace{<:Number}}

include("operators/fieldoperators.jl")
include("operators/sumofterms.jl")
include("operators/hamiltonians.jl")

include("abstractcmps.jl")

include("infinitecmps/infinitecmps.jl")
include("infinitecmps/transfer.jl")
include("infinitecmps/environments.jl")
include("infinitecmps/gauging.jl")
include("infinitecmps/gradients.jl")
include("infinitecmps/groundstate.jl")


end
