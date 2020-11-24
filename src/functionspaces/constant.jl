# Type definition
struct Constant{T} <: FunctionSpace{T}
    coeffs::Base.RefValue{T}
    Constant(x::T) where T = new{T}(Ref(x))
end

# Basic properties
domain(f::Constant) = (-Inf,+Inf)
period(f::Constant) = 0

coefficients(F::Constant) = F.coeffs

truncate!(F::Constant; kwargs...) = F

Base.:(==)(F1::Constant, F2::Constant) = (F1[] == F2[])

# Indexing, getting and setting coefficients
Base.eachindex(F::Constant) = 0:0
Base.getindex(F::Constant) = F.coeffs[]
Base.getindex(F::Constant, i::Integer) = iszero(i) ? getindex(F) : throw(BoundsError(F, i))
Base.setindex!(F::Constant, v) = F.coeffs[] = v
Base.getindex(F::Constant, v, i::Integer) =
    iszero(i) ? setindex!(F, v) : throw(BoundsError(F, i))

# Use as function
(F::Constant)(x) = F[]

# special purpose constructor
Base.similar(F::Constant, ::Type{T} = scalartype(F)) where {T} =
    Constant(zero(T)*F[])

Base.zero(F::Constant) = Constant(zero(F[]))
Base.one(F::Constant) = Constant(one(F[]))

# Arithmetic (out of place)
Base.copy(F::Constant) = Constant(copy(F[]))

Base.:-(F::Constant) = Constant(-F[])

Base.:*(F::Constant, a) = Constant(F[]*a)
Base.:*(a, F::Constant) = Constant(a*F[])
Base.:/(F::Constant, a) = Constant(F[]/a)
Base.:\(a, F::Constant) = Constant(a\F[])

Base.:+(F1::Constant, F2::Constant) = Constant(F1[]+F2[])
Base.:-(F1::Constant, F2::Constant) = Constant(F1[]-F2[])
Base.:*(F1::Constant, F2::Constant) = Constant(F1[]*F2[])
Base.:/(F1::Constant, F2::Constant) = Constant(F1[]/F2[])
Base.:\(F1::Constant, F2::Constant) = Constant(F1[]\F2[])

Base.conj(F::Constant) = Constant(conj(F[]))
Base.adjoint(F::Constant) = Constant(adjoint(F[]))
Base.transpose(F::Constant) = Constant(transpose(F[]))

LinearAlgebra.tr(F::Constant) = Constant(tr(F[]))

Base.real(F::Constant) = Constant(real(F[]))
Base.imag(F::Constant) = Constant(imag(F[]))

# Arithmetic (in place / mutating methods)
function Base.copy!(Fdst::Constant, Fsrc::Constant)
    if eltype(Fdst) <: Number
        Fdst[] = Fsrc[]
    else
        copy!(Fdst[], Fsrc[])
    end
    return Fdst
end
function LinearAlgebra.rmul!(F::Constant, α)
    if eltype(F) <: Number || !(α isa Number)
        F[] = F[]*α
    else
        rmul!(F[], α)
    end
    return F
end
function LinearAlgebra.lmul!(α, F::Constant)
    if eltype(F) <: Number || !(α isa Number)
        F[] = α*F[]
    else
        lmul!(α, F[])
    end
    return F
end

function LinearAlgebra.mul!(Fdst::Constant, α, Fsrc::Constant)
    if eltype(Fdst) <: Number
        Fdst[] = α*Fsrc[]
    else
        mul!(Fdst[], α, Fsrc[])
    end
    return Fdst
end
function LinearAlgebra.mul!(Fdst::Constant, Fsrc::Constant, α)
    if eltype(Fdst) <: Number
        Fdst[] = Fsrc[]*α
    else
        mul!(Fdst[], Fsrc[], α)
    end
    return Fdst
end
function LinearAlgebra.axpy!(α, Fx::Constant, Fy::Constant)
    if eltype(Fy) <: Number
        Fy[] = α*Fx[] + Fy[]
    else
        axpy!(α, Fx[], Fy[])
    end
    return Fy
end
function LinearAlgebra.axpby!(α, Fx::Constant, β, Fy::Constant)
    if eltype(Fy) <: Number
        Fy[] = α*Fx[] + β*F[y]
    else
        axpby!(α, Fx[], β, Fy[])
    end
    return Fy
end

LinearAlgebra.mul!(F::Constant, F1::Constant, F2::Constant,
                    α = true, β = false) = truncmul!(F, F1, F2, α, β)

function truncmul!(F::Constant, F1::Constant, F2::Constant,
                    α = true, β = false;
                    Kmax::Integer = 1, tol::Real = 0)

    if eltype(F) <: Number || (eltype(F1) <: Number && eltype(F2) <: Number)
        F[] = α * F1[] + β * F[2]
    else
        if eltype(F1) <: Number
            axpby!(α*F1[], F2[], β, F[])
        elseif eltype(F2) <: Number
            axpby!(α*F2[], F1[], β, F[])
        else
            mul!(F[], F1[], F2[], α, β)
        end
    end
    return F
end

# Inner product and norm
localdot(F1::Constant, F2::Constant) = Constant(dot(F1[], F2[]))

LinearAlgebra.dot(F1::Constant, F2::Constant) = dot(F1[], F2[])
LinearAlgebra.norm(F::Constant) = norm(F[])

function LinearAlgebra.isapprox(x::Constant, y::Constant;
                                atol::Real=0,
                                rtol::Real=defaulttol(x[0]))
    return norm(x-y) <= max(atol, rtol*max(norm(x), norm(y)))
end

differentiate(F::Constant) = zero(F)
integrate(F::Constant, (a,b)::Tuple{Real,Real}) = F[]*(b-a)

# Fit constant: take average over N points
fit(f, ::Type{Constant}, (a,b)::Tuple{Real,Real}; N = 5) =
    Constant(sum(f, range(a, b; length = N))/N)

# Inverse and square root
Base.inv(F::Constant) = Constant(inv(F[]))
Base.sqrt(F::Constant) = Constant(sqrt(F[]))
