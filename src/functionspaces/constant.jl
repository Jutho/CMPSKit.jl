# Type definition
struct Constant{T} <: FunctionSeries{T}
    coeffs::Base.RefValue{T}
    Constant(x::T) where T = new{T}(Ref(x))
end

# Basic properties
domain(f::Constant) = (-Inf,+Inf)
period(f::Constant) = 0

coefficients(f::Constant) = f.coeffs

Base.:(==)(f1::Constant, f2::Constant) = (f1[] == f2[])

# Indexing, getting and setting coefficients
Base.eachindex(f::Constant) = 0:0
Base.getindex(f::Constant) = f.coeffs[]
Base.getindex(f::Constant, i::Integer) = iszero(i) ? getindex(f) : throw(BoundsError(f, i))
Base.setindex!(f::Constant, v) = f.coeffs[] = v
Base.getindex(f::Constant, v, i::Integer) =
    iszero(i) ? setindex!(f, v) : throw(BoundsError(f, i))

# Use as function
(f::Constant)(x) = f[]

# Change number of coefficients
truncate!(f::Constant; kwargs...) = f

# special purpose constructor
Base.similar(f::Constant, ::Type{T} = scalartype(f)) where {T} =
    Constant(zero(T)*f[])

Base.zero(f::Constant) = Constant(zero(f[]))
Base.one(f::Constant) = Constant(one(f[]))

# Arithmetic (out of place)
Base.copy(f::Constant) = Constant(copy(f[]))

Base.:-(f::Constant) = Constant(-f[])
Base.:+(f::Constant) = Constant(+f[])

Base.:*(f::Constant, a::Const) = Constant(f[]*a)
Base.:*(a::Const, f::Constant) = Constant(a*f[])
Base.:/(f::Constant, a::Const) = Constant(f[]/a)
Base.:\(a::Const, f::Constant) = Constant(a\f[])

Base.:+(f1::Constant, f2::Constant) = Constant(f1[]+f2[])
Base.:-(f1::Constant, f2::Constant) = Constant(f1[]-f2[])
Base.:*(f1::Constant, f2::Constant) = Constant(f1[]*f2[])
Base.:/(f1::Constant, f2::Constant) = Constant(f1[]/f2[])
Base.:\(f1::Constant, f2::Constant) = Constant(f1[]\f2[])

truncmul(f1::Constant, f2::Constant) = Constant(f1[]*f2[])

Base.conj(f::Constant) = Constant(conj(f[]))
Base.adjoint(f::Constant) = Constant(adjoint(f[]))
Base.transpose(f::Constant) = Constant(transpose(f[]))

LinearAlgebra.tr(f::Constant) = Constant(tr(f[]))

Base.real(f::Constant) = Constant(real(f[]))
Base.imag(f::Constant) = Constant(imag(f[]))

# Arithmetic (in place / mutating methods)
function Base.copy!(fdst::Constant, fsrc::Constant)
    if eltype(fdst) <: Number
        fdst[] = fsrc[]
    else
        copy!(fdst[], fsrc[])
    end
    return fdst
end
function LinearAlgebra.rmul!(f::Constant, α::Number)
    if eltype(f) <: Number || !(α isa Number)
        f[] = f[]*α
    else
        rmul!(f[], α)
    end
    return f
end
function LinearAlgebra.lmul!(α::Number, f::Constant)
    if eltype(f) <: Number || !(α isa Number)
        f[] = α*f[]
    else
        lmul!(α, f[])
    end
    return f
end

LinearAlgebra.axpy!(α::Number, fx::Constant, fy::Constant) =
    truncadd!(fy, fx, α)
LinearAlgebra.axpby!(α::Number, fx::Constant, β::Number, fy::Constant) =
    truncadd!(fy, fx, α, β)
LinearAlgebra.mul!(fy::Constant, s::Number, fx::Constant, α = true, β = false) =
    truncmul!(fy, s, fx, α, β)
LinearAlgebra.mul!(fy::Constant, fx::Constant, s::Number, α = true, β = false) =
    truncmul!(fy, fx, s, α, β)
LinearAlgebra.mul!(f::Constant, f1::Constant, f2::Constant,
                    α = true, β = false) = truncmul!(f, f1, f2, α, β)

function truncadd!(fy::Constant, fx::Constant, α = true, β = true; kwargs...)
    if eltype(fy) <: Number
        fy[] = α*fx[] + β*fy[]
    else
        if iszero(β)
            mul!(fy[], α, fx[])
        elseif isone(β)
            axpy!(α, fx[], fy[])
        else
            axpby!(α, fx[], β, fy[])
        end
    end
    return fy
end

truncmul!(fdst::Constant, α₁::Number, fsrc::Constant, α₂ = true, β = false;
            kwargs...) = truncadd!(fdst, fsrc, α₁ * α₂, β; kwargs...)
truncmul!(fdst::Constant, fsrc::Constant, α₁::Number, α₂ = true, β = false;
            kwargs...) = truncadd!(fdst, fsrc, α₁ * α₂, β; kwargs...)

function truncmul!(f::Constant, f1::Constant, f2::Constant,
                    α = true, β = false;
                    Kmax::Integer = 0, tol::Real = 0)

    if eltype(f) <: Number
        f[] = α * f1[] + β * f2[]
    else
        if eltype(f1) <: Number
            axpby!(α*f1[], f2[], β, f[])
        elseif eltype(f2) <: Number
            axpby!(α*f2[], f1[], β, f[])
        else
            mul!(f[], f1[], f2[], α, β)
        end
    end
    return f
end

# Inner product and norm
localdot(f1::Constant, f2::Constant) = Constant(dot(f1[], f2[]))

LinearAlgebra.dot(f1::Constant, f2::Constant) = dot(f1[], f2[])
LinearAlgebra.norm(f::Constant) = norm(f[])

differentiate(f::Constant) = zero(f)
integrate(f::Constant, (a,b)::Tuple{Real,Real}) = f[]*(b-a)

# Fit constant: take average over N points
fit(f, ::Type{Constant}, (a,b)::Tuple{Real,Real}; numpoints = 5) =
    Constant(sum(f, range(a, b; length = numpoints))/numpoints)

# Inverse and square root
Base.inv(f::Constant) = Constant(inv(f[]))
Base.sqrt(f::Constant) = Constant(sqrt(f[]))
