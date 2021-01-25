# Type definition
mutable struct TaylorSeries{T, S<:Real} <: FunctionSeries{T}
    coeffs::Vector{T}
    offset::S
    function TaylorSeries(coeffs::Vector{T}, offset = 0.) where {T}
        @assert length(coeffs) >= 1
        S = typeof(offset)
        new{T,S}(coeffs, offset)
    end
end

# Basic properties
offset(f::TaylorSeries) = f.offset
domain(f::TaylorSeries) = (-Inf,+Inf)

degree(f::TaylorSeries) = length(f.coeffs) - 1
coefficients(f::TaylorSeries) = f.coeffs

function shift!(f::TaylorSeries{T}, b) where T
    a = offset(f)
    a == b && return f
    for k = 0:degree(f)
        for l = degree(f):-1:k+1
            if T <: AbstractArray
                axpy!((b-a)^(l-k)*binomial(l,k), f[l], f[k])
            else
                f[k] += f[l]*(b-a)^(l-k)*binomial(l,k)
            end
        end
    end
    f.offset = b
    return f
end
shift(f::TaylorSeries, b) = b == offset(f) ? t : shift!(copy(f), b)

function Base.:(==)(f1::TaylorSeries, f2::TaylorSeries)
    offset(f1) == offset(f2) || return (f1 == shift(f2, offset(f1)))
    K = max(degree(f1), degree(f1))
    for k = 0:K
        f1[k] == f2[k] || return false
    end
    return true
end

# Indexing, getting and setting coefficients
Base.eachindex(f::TaylorSeries) = 0:degree(f)
function Base.getindex(f::TaylorSeries, k)
    k < 0 && return BoundsError(f, k)
    @inbounds if k > degree(f)
        return zero(f.coeffs[1])
    else
        return f.coeffs[k+1]
    end
end
function Base.setindex!(f::TaylorSeries, v, k)
    k < 0 && return BoundsError(f, k)
    @inbounds if k > degree(f)
        f0 = f[0]
        while length(f) < k
            push!(f.coeffs, zero(f0))
        end
        push!(f.coeffs, v)
        return v
    else
        setindex!(f.coeffs, v, k+1)
        return v
    end
end

# Use as function
function (f::TaylorSeries)(x::Real)
    k = degree(t)
    xa = x - offset(t)
    (xa == zero(xa) || k == 0) && return copy(f[0])
    v = copy(f[k])
    if v isa AbstractArray
        while k > 0
            v = LinearAlgebra.axpby!(one(xa), f[k-1], xa, v)
            k -= 1
        end
    else
        while k > 0
            v = muladd(xa, v, f[k-1])
            k -= 1
        end
    end
    return v
end

# Change number of coefficients
function truncate!(f::TaylorSeries; Kmax::Integer = degree(f), tol::Real = 0, dx::Real = 1.)
    ktol = findlast(k->(norm(f[k])*abs(dx)^k >= tol), eachindex(f))
    Ktol = ktol === nothing ? 0 : ktol-1
    Kmax = min(Kmax, Ktol)
    if Kmax < degree(f)
        resize!(f.coeffs, Kmax + 1)
    end
    return f
end

function setdegree!(f::TaylorSeries, K::Int)
    while degree(f) < K
        push!(f.coeffs, zero(f[0]))
    end
    if degree(f) > K
        resize!(f.coeffs, K+1)
    end
    return f
end

# special purpose constructor
Base.similar(f::TaylorSeries, ::Type{T} = scalartype(f)) where {T} =
    TaylorSeries([zero(T)*f[0]], offset(f))

Base.zero(f::TaylorSeries) = TaylorSeries([zero(f[0])], offset(f))
Base.one(f::TaylorSeries) = TaylorSeries([one(f[0])], offset(f))

# Arithmetic (out of place)
Base.copy(f::TaylorSeries) = TaylorSeries(copy.(coefficients(f)), offset(f))

Base.:-(f::TaylorSeries) = TaylorSeries(.-coefficients(f), offset(f))
Base.:+(f::TaylorSeries) = TaylorSeries(.+coefficients(f), offset(f))

const Const = Union{Number,AbstractArray}
Base.:*(f::TaylorSeries, a::Const) =
    TaylorSeries([c*a for c in coefficients(f)], offset(f))
Base.:*(a::Const, f::TaylorSeries) =
    TaylorSeries([a*c for c in coefficients(f)], offset(f))
Base.:/(f::TaylorSeries, a::Const) =
    TaylorSeries([c/a for c in coefficients(f)], offset(f))
Base.:\(a::Const, f::TaylorSeries) =
    TaylorSeries([a\c for c in coefficients(f)], offset(f))


function Base.:+(f1::TaylorSeries, f2::TaylorSeries)
    offset(f1) == offset(f2) || return (f1 + shift(f2, offset(f1)))
    K = min(degree(f1), degree(f2))
    coeffs = [f1[j]+f2[j] for j = 0:K]
    for j = K+1:degree(f1)
        push!(coeffs, copy(f1[j]))
    end
    for j = K+1:degree(f2)
        push!(coeffs, copy(f2[j]))
    end
    return TaylorSeries(coeffs, offset(f1))
end

function Base.:-(f1::TaylorSeries, f2::TaylorSeries)
    offset(f1) == offset(f2) || return (f1 - shift(f2, offset(f1)))
    K = min(degree(f1), degree(f2))
    coeffs = [f1[j]-f2[j] for j = 0:K]
    for j = K+1:degree(f1)
        push!(coeffs, copy(f1[j]))
    end
    for j = K+1:degree(f2)
        push!(coeffs, -(f2[j]))
    end
    return TaylorSeries(coeffs, offset(f1))
end

Base.:*(f1::TaylorSeries, f2::TaylorSeries) = truncmul(f1, f2)

function truncmul(f1::TaylorSeries, f2::TaylorSeries; kwargs...)
    if offset(f1) == offset(f2)
        _truncmul(f1, shift(f2, offset(f1)); kwargs...)
    else
        _truncmul(f1, f2; kwargs...)
    end
end

function _truncmul(f1::TaylorSeries, f2::TaylorSeries;
                    Kmax = degree(f1) + degree(f2), tol::Real = 0, dx = 1)
    K = min(Kmax, degree(f1) + degree(f2))
    f = TaylorSeries(sizehint!([zero(f1[0])*zero(f2[0])], K+1), offset(f1))
    return truncmul!(f, f1, f2, true, true; Kmax = Kmax, tol = tol, dx = dx)
end

Base.conj(f::TaylorSeries) = TaylorSeries(map(conj, coefficients(f)), offset(f))
Base.adjoint(f::TaylorSeries) = TaylorSeries(map(adjoint, coefficients(f)), offset(f))
Base.transpose(f::TaylorSeries) = TaylorSeries(map(transpose, coefficients(f)), offset(f))

LinearAlgebra.tr(f::TaylorSeries) = TaylorSeries(map(tr, coefficients(f)), offset(f))

Base.real(f::TaylorSeries) = TaylorSeries(map(real, coefficients(f)), offset(f))
Base.imag(f::TaylorSeries) = TaylorSeries(map(imag, coefficients(f)), offset(f))

# Arithmetic (in place / mutating methods)
function Base.copy!(fdst::TaylorSeries, fsrc::TaylorSeries)
    fdst.offset = offset(fsrc)
    setdegree!(fdst, degree(fsrc))
    for j = 0:degree(fsrc)
        fdst[j] = copy(fsrc[j])
    end
    return fdst
end
function LinearAlgebra.rmul!(f::TaylorSeries, α::Number)
    if eltype(f) <: Number
        rmul!(coefficients(f), α)
    else
        for k in eachindex(f)
            rmul!(f[k], α)
        end
    end
    return f
end
function LinearAlgebra.lmul!(α::Number, f::TaylorSeries)
    if eltype(f) <: Number
        lmul!(α, coefficients(f))
    else
        for k in eachindex(f)
            lmul!(α, f[k])
        end
    end
    return f
end
LinearAlgebra.axpy!(α::Number, fx::TaylorSeries, fy::TaylorSeries) =
    truncadd!(fy, fx, α)
LinearAlgebra.axpby!(α::Number, fx::TaylorSeries, β::Number, fy::TaylorSeries) =
    truncadd!(fy, fx, α, β)
LinearAlgebra.mul!(fy::TaylorSeries, s::Number, fx::TaylorSeries, α = true, β = false) =
    truncmul!(fy, s, fx, α, β)
LinearAlgebra.mul!(fy::TaylorSeries, fx::TaylorSeries, s::Number, α = true, β = false) =
    truncmul!(fy, fx, s, α, β)
LinearAlgebra.mul!(f::TaylorSeries, f1::TaylorSeries, f2::TaylorSeries,
                    α = true, β = false) = truncmul!(f, f1, f2, α, β)

function truncadd!(fy::TaylorSeries, fx::TaylorSeries, α = true, β = true;
                    Kmax::Integer = max(iszero(β) ? 0 : degree(fy), degree(fx)),
                    tol::Real = 0, dx = 1.)
    Kx = degree(fx)
    Ky = min(Kmax, iszero(β) ? Kx : max(Kx, degree(fy)))
    setdegree!(fy, Ky)
    @assert offset(fx) == offset(fy)
    # if offset(fx) == offset(fy)
        if eltype(fy) <: Number
            if Kx > Ky
                axpby!(α, view(coefficients(fx, 1:Ky+1)), β, coefficients(fy))
            else
                axpby!(α, coefficients(fx), β, view(coefficients(fy), 1:Kx+1))
                lmul!(β, view(coefficients(fy), (Kx+2):(Ky+1)))
            end
        else
            for k in 0:Kx
                axpby!(α, fx[k], β, fy[k])
            end
            for k in Kx+1:Ky
                lmul!(β, fy[k])
            end
        end
    # else
    #     dx = offset(fy) - offset(fx)
    #     lmul!(β, fy)
    #     for k = 0:Kx
    #         for l = 0:min(k, Ky)
    #             if eltype(fy) <: Number
    #                 fy[l] += α*fx[k]*dx^(l-k)*binomial(l,k)
    #             else
    #                 axpy!(α*dx^(l-k)*binomial(l,k), fx[k], fy[l])
    #             end
    #         end
    #     end
    # end
    return truncate!(fy; tol = tol, dx = dx)
end

function truncmul!(fdst::TaylorSeries, α₁::Number, fsrc::TaylorSeries,
                    α₂ = true, β = false;
                    Kmax::Integer = max(iszero(β) ? 0 : degree(fdst), degree(fsrc)),
                    tol::Real = 0, dx)
    truncadd!(fdst, fsrc, α₁ * α₂, β; )
    α = α₁ * α₂
    K = min(Kmax, iszero(β) ? degree(fsrc) : max(degree(fdst), degree(fsrc)))
    setdegree!(fdst, K)
    @assert offset(fx) == offset(fy)
    if eltype(fdst) <: Number
        axpby!(α, view(coefficients(fsrc), 1:K+1), β, coefficients(Fdst))
    else
        for k in eachindex(fdst)
            axpby!(Fdst[k], α, Fsrc[k])
        end
    end
    return Fdst
end

function truncmul!(Fy::TaylorSeries, Fx::TaylorSeries, α₁::Number,
                    α₂ = true, β = false; kwargs...)
    return
    α = α₁ * α₂
    domain(Fdst) == domain(Fsrc) || throw(DomainMismatch())
    K = min(Kmax, iszero(β) ? nummodes(Fsrc) : max(nummodes(Fdst), nummodes(Fsrc)))
    setnummodes!(Fdst, K)
    if eltype(Fdst) <: Number
        mul!(coefficients(Fdst), coefficients(Fsrc), α)
    else
        for k in eachindex(Fdst)
            mul!(Fdst[k], Fsrc[k], α)
        end
    end
    return Fdst
end
function truncmul!(F::TaylorSeries, F1::TaylorSeries, F2::TaylorSeries,
                    α = true, β = false;
                    Kmax::Integer = max(nummodes(F), nummodes(F1)+nummodes(F2)), tol::Real = 0)
    domain(F) == domain(F1) == domain(F2) || throw(DomainMismatch())
    K1 = nummodes(F1)
    K2 = nummodes(F2)
    K = min(Kmax, max(iszero(β) ? 0 : nummodes(F), K1+K2))
    setnummodes!(F, K)
    Threads.@threads for k = -K:K
        fk = F[k]
        if fk isa AbstractArray
            T = eltype(fk)
            if β == 0
                fill!(fk, T(β))
            elseif β != 1
                rmul!(fk, T(β))
            end
            for k1 = max(-K1, k-K2):min(K1, k+K2)
                k2 = k - k1
                fk1 = F1[k1]
                fk2 = F2[k2]
                if fk1 isa AbstractArray && fk2 isa AbstractArray
                    mul!(fk, fk1, fk2, T(α), T(true))
                elseif fk1 isa AbstractArray
                    axpy!(T(fk2*α), fk1, fk)
                elseif fk2 isa AbstractArray
                    axpy!(T(fk1*α), fk2, fk)
                else
                    @warn "unexpected branch"
                    fk .= fk + (fk1*fk2*α)
                end
            end
        else
            fk *= β
            for k1 = max(-K1, k-K2):min(K1, k+K2)
                k2 = k - k1
                fk1 = F1[k1]
                fk2 = F2[k2]
                fk += fk1*fk2*α
            end
            F[k] = fk
        end
    end
    if tol != 0
        truncate!(F; tol = tol)
    end
    return F
end

# Inner product and norm
function localdot(f1::TaylorSeries, f2::TaylorSeries)
    offset(f1) == offset(f2) || localdot(f1, shift(f2, offset(f1)))
    K1 = degree(f1)
    K2 = degree(f2)
    coeffs = let K = K1 + K2
        [sum(dot(f1[k1], f2[k-k1]) for k1 = max(0, k-K2):min(k, K1)) for k=0:K]
    end
    return TaylorSeries(coeffs, offset(f1))
end

function differentiate(f::TaylorSeries)
    if degree(f) >= 1
        return TaylorSeries([k*f[k] for k = 1:degree(f)], offset(f))
    else
        return zero(f)
    end
end
function integrate(f::TaylorSeries, (a,b)::Tuple{Real,Real})
    s = t[0]*(b-a)
    c = offset(f)
    db = b - c
    da = a - c
    for k = 1:degree(f)
        s += f[k]*(db^(k+1) - da^(k+1))/(k+1)
    end
    return s
end

# Simple Fourier transform: typically not needed for large number of points, but should
# work with matrix valued functions etc
fit(f, ::Type{TaylorSeries}, (a,b)::Tuple{Real,Real}; kwargs...) =
    fit(f, TaylorSeries, b-a; kwargs...)

function fit(f, ::Type{TaylorSeries}, period = 1; Kmax = 10, tol = 1e-12)
    K = Kmax
    x = (0:2*K) * (period / (2K+1))
    fx = map(f, x)
    ω = 2*pi*0/period
    integrand = exp.((-im*ω) .* x) .* fx
    F = TaylorSeries([sum(integrand)/(2K+1)], period)
    isrealf = all(isreal, fx)
    for k = 1:Kmax
        ω = 2*pi*k/period
        integrand .= exp.((-im*ω) .* x) .* fx
        F[k] = sum(integrand)/(2K+1)
        if isrealf
            F[-k] = conj(F[k])
        else
            integrand .= exp.((+im*ω) .* x) .* fx
            F[-k] = sum(integrand)/(2K+1)
        end
    end
    return F
end

# function Base.inv(F::TaylorSeries; tol=1e-12, Kmax = 2*nummodes(F))
#     K = nummodes(F)
#     while true
#         Fx = map(F, (0:2*K)/(2*K+1))
#         Gx = map(inv, Fx)
#         krange = vcat([0],[(-1)^iseven(l)*((l+1)>>1) for l=1:2*K])
#         factor = -im*2*pi/(2*K+1)
#         coeffG = let factor=factor, Gx=Gx, K = K
#             map(krange) do k
#                 sum(exp(factor*k*(l-1))*Gx[l] for l = 1:(2*K+1))/(2*K+1)
#             end
#         end
#         G = TaylorSeries(coeffG)
#         if norm(G*F-one(G))<tol || K == Kmax
#             return G
#         else
#             K = min(2*K, Kmax)
#         end
#     end
# end
#
# function inv2(F::TaylorSeries; tol = 1e-12, Kmax = 2*nummodes(F))
#     Fi = inv(F; tol = tol, Kmax = Kmax)
#     Fi2 = linsolve(x->setnummodes!(x*F, nummodes(x)), one(F), Fi; tol = tol)
#
#     K = nummodes(F)
#     while true
#         Fx = map(F, (0:2*K)/(2*K+1))
#         Gx = map(inv, Fx)
#         krange = vcat([0],[(-1)^iseven(l)*((l+1)>>1) for l=1:2*K])
#         factor = -im*2*pi/(2*K+1)
#         coeffG = let factor=factor, Gx=Gx, K = K
#             map(krange) do k
#                 sum(exp(factor*k*(l-1))*Gx[l] for l = 1:(2*K+1))/(2*K+1)
#             end
#         end
#         G = TaylorSeries(coeffG)
#         if norm(G*F-one(G))<tol || K == Kmax
#             return G
#         else
#             K = min(2*K, Kmax)
#         end
#     end
#     return Fi2
# end
#
# function Base.sqrt(F::TaylorSeries; tol=1e-12, Kmax = 200)
#     K = nummodes(F)
#     while true
#         Fx = map(F, (0:2*K)/(2*K+1))
#         Gx = map(sqrt, Fx)
#         krange = vcat([0],[(-1)^iseven(l)*((l+1)>>1) for l=1:2*K])
#         factor = -im*2*pi/(2*K+1)
#         coeffG = let factor=factor, Gx=Gx
#             map(krange) do k
#                 sum(exp(factor*k*(l-1))*Gx[l] for l = 1:(2*K+1))/(2*K+1)
#             end
#         end
#         G = TaylorSeries(coeffG)
#         if norm(G*G-F)<tol || K == Kmax
#             return G
#         else
#             K = min(2*K, Kmax)
#         end
#     end
# end
