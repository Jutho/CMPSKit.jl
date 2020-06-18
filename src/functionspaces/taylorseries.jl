using LinearAlgebra

mutable struct TaylorSeries{T,S<:Real}
    coeffs::Vector{T}
    offset::S
end
TaylorSeries(coeffs::Vector{T}) where T = TaylorSeries(coeffs, 0)

Base.length(t::TaylorSeries) = length(t.coeffs)
degree(t::TaylorSeries) = length(t) - 1
offset(t::TaylorSeries) = t.offset

function shift!(t::TaylorSeries{T}, b) where T
    a = offset(t)
    a == b && return t
    for k = 0:degree(t)
        for l = degree(t):-1:k+1
            if T <: AbstractArray
                axpy!((b-a)^(l-k)*binomial(l,k), t[l], t[k])
            else
                t[k] += t[l]*(b-a)^(l-k)*binomial(l,k)
            end
        end
    end
    t.offset = b
    return t
end
shift(t::TaylorSeries, b) = b == offset(t) ? t : shift!(copy(t), b)

Base.copy(t::TaylorSeries) = TaylorSeries([copy(t[k]) for k = 0:degree(t)], offset(t))

function Base.getindex(t::TaylorSeries, k)
    k < 0 && return BoundsError(t, k)
    @inbounds if k > degree(t)
        return zero(t.coeffs[1])
    else
        return t.coeffs[k+1]
    end
end
function Base.setindex!(t::TaylorSeries, v, k)
    k < 0 && return BoundsError(t, k)
    @inbounds if k > degree(t)
        t0 = t[0]
        while length(t) < k
            push!(t.coeffs, zero(t0))
        end
        push!(t.coeffs, v)
        return v
    else
        setindex!(t.coeffs, v, k+1)
        return v
    end
end

function (t::TaylorSeries)(x::Real)
    k = degree(t)
    xa = x - offset(t)
    (xa == zero(xa) || k == 0) && return copy(t[0])
    v = copy(t[k])
    if v isa AbstractArray
        while k > 0
            v = LinearAlgebra.axpby!(one(xa), t[k-1], xa, v)
            k -= 1
        end
    else
        while k > 0
            v = muladd(xa, v, t[k-1])
            k -= 1
        end
    end
    return v
end

Base.eltype(t::TaylorSeries) = eltype(typeof(t))
Base.eltype(::Type{TaylorSeries{<:Real,T}}) where {T} = eltype(T)

function trunc!(t::TaylorSeries; kmax::Integer = length(F)-1, tol::Real = 0)
    ktol = findlast(x->norm(x)>=tol, t.coeffs) - 1
    kmax = min(kmax, ktol === nothing ? 0 : ktol)
    if degree(t) > kmax
        resize!(t.coeffs, kmax+1)
    end
    return t
end

function setdegree!(t::TaylorSeries, K::Int)
    while degree(t) < K
        push!(t.coeffs, zero(t[0]))
    end
    if degree(t) > K
        resize!(t.coeffs, K+1)
    end
    return t
end

function Base.copyto!(tdst::TaylorSeries, tsrc::TaylorSeries)
    @assert offset(tdst) == offset(tsrc)
    setdegree!(tdst, degree(tsrc))
    for j = 0:degree(tsrc)
        tdst[j] = copy(tsrc[j])
    end
    return tdst
end

Base.fill!(t::TaylorSeries{<:Number}, α) = (fill!(t.coeffs, α); return t)
Base.fill!(t::TaylorSeries{<:AbstractArray}, α) = (map!(x->fill!(x, α), t.coeffs, t.coeffs); return t)

LinearAlgebra.rmul!(t::TaylorSeries, α) = (rmul!(t.coeffs, α); return t)
LinearAlgebra.lmul!(α, t::TaylorSeries) = (lmul!(α, t.coeffs); return t)

function LinearAlgebra.mul!(tdst::TaylorSeries, α, tsrc::TaylorSeries)
    @assert offset(tdst) == offset(tsrc)
    setdegree!(tdst, degree(tsrc))
    mul!(tdst.coeffs, α, tsrc.coeffs)
    return tdst
end

function LinearAlgebra.mul!(tdst::TaylorSeries, tsrc::TaylorSeries, α)
    @assert offset(tdst) == offset(tsrc)
    setdegree!(tdst, degree(tsrc))
    mul!(tdst.coeffs, tsrc.coeffs, α)
    return tdst
end

function LinearAlgebra.axpy!(α, tx::TaylorSeries, ty::TaylorSeries)
    @assert offset(tx) == offset(ty)
    Kx = degree(tx)
    setdegree!(ty, max(tx, degree(ty)))
    Ky = degree(ty)
    if Ky > Kx
        LinearAlgebra.axpy!(α, tx.coeffs, view(ty.coeffs, 1:(Kx+1)))
    else
        LinearAlgebra.axpy!(α, tx.coeffs, ty.coeffs)
    end
    return ty
end
function LinearAlgebra.axpby!(α, tx::TaylorSeries, β, ty::TaylorSeries)
    @assert offset(tx) == offset(ty)
    Kx = degree(tx)
    setdegree!(ty, max(tx, degree(ty)))
    Ky = degree(ty)
    if Ky > Kx
        LinearAlgebra.axpby!(α, tx.coeffs, β, view(ty.coeffs, 1:(Kx+1)))
        lmul!(β, view(ty.coeffs, (Kx+2):(Ky+1)))
    else
        LinearAlgebra.axpby!(α, tx.coeffs, β, ty.coeffs)
    end
    return ty
end

function LinearAlgebra.axpy!(α, tx::TaylorSeries{<:AbstractArray},
                                ty::TaylorSeries{<:AbstractArray})
    @assert offset(tx) == offset(ty)
    Kx = degree(tx)
    setdegree!(ty, max(Kx, degree(ty)))
    for j = 0:Kx
        LinearAlgebra.axpy!(α, tx[j], ty[j])
    end
    return ty
end
function LinearAlgebra.axpby!(α, tx::TaylorSeries{<:AbstractArray},
                                β, ty::TaylorSeries{<:AbstractArray})
    @assert offset(tx) == offset(ty)
    Kx = degree(tx)
    setdegree!(ty, max(Kx, degree(ty)))
    Ky = degree(ty)
    for j = 0:Kx
        LinearAlgebra.axpby!(α, tx[j], β, ty[j])
    end
    for j = Kx+1:Ky
        LinearAlgebra.lmul!(β, ty[j])
    end
    return ty
end

Base.zero(t::TaylorSeries) = TaylorSeries([zero(t[0])], offset(t))
Base.one(t::TaylorSeries) = TaylorSeries([one(t[0])], offset(t))

function Base.similar(t::TaylorSeries{T}) where T
    if isbitstype(T)
        return TaylorSeries(similar(t.coeffs), offset(t))
    else
        return TaylorSeries(similar.(t.coeffs), offset(t))
    end
end

function Base.:+(t1::TaylorSeries, t2::TaylorSeries)
    @assert offset(t1) == offset(t2)
    K = min(degree(t1), degree(t2))
    coeffs = [t1[j]+t2[j] for j = 0:K]
    for j = K+1:degree(t1)
        push!(coeffs, copy(t1[j]))
    end
    for j = K+1:degree(t2)
        push!(coeffs, copy(t2[j]))
    end
    return TaylorSeries(coeffs, offset(t1))
end

function Base.:-(t1::TaylorSeries, t2::TaylorSeries)
    @assert offset(t1) == offset(t2)
    K = min(degree(t1), degree(t2))
    coeffs = [t1[j]-t2[j] for j = 0:K]
    for j = K+1:degree(t1)
        push!(coeffs, copy(t1[j]))
    end
    for j = K+1:degree(t2)
        push!(coeffs, -t2[j])
    end
    return TaylorSeries(coeffs, offset(t1))
end

Base.:-(t::TaylorSeries) = TaylorSeries(-t.coeffs, offset(t))

Base.:*(t::TaylorSeries, a::Number) = TaylorSeries([c*a for c in t.coeffs], offset(t))
Base.:*(a::Number, t::TaylorSeries) = TaylorSeries([a*c for c in t.coeffs], offset(t))
Base.:/(t::TaylorSeries, a::Number) = TaylorSeries([c/a for c in t.coeffs], offset(t))
Base.:\(a::Number, t::TaylorSeries) = TaylorSeries([a\c for c in t.coeffs], offset(t))

Base.conj(t::TaylorSeries) = TaylorSeries(map(conj, t.coeffs), offset(t))
Base.adjoint(t::TaylorSeries) = TaylorSeries(map(adjoint, t.coeffs), offset(t))
Base.transpose(t::TaylorSeries) = TaylorSeries(map(transpose, t.coeffs), offset(t))

function Base.:*(t1::TaylorSeries, t2::TaylorSeries; Kmax = degree(t1) + degree(t2))
    @assert offset(t1) == offset(t2)
    K = min(Kmax, degree(t1) + degree(t2))
    coeffs = [zero(t1[0])*zero(t2[0]) for k = 0:K]
    t = TaylorSeries(coeffs, offset(t1))
    return mul!(t, t1, t2, true, true)
end

function LinearAlgebra.mul!(t::TaylorSeries, t1::TaylorSeries, t2::TaylorSeries,
                                α = true, β = false)
    @assert offset(t) == offset(t1) == offset(t2)
    K1 = degree(t1)
    K2 = degree(t2)
    K = degree(t)
    for k = 0:K
        tk = t[k]
        if tk isa AbstractArray
            if β != 1
                rmul!(tk, β)
            end
            for k1 = max(0, k-K2):min(k, K1)
                tk1 = t1[k1]
                tk2 = t2[k-k1]
                if tk1 isa AbstractArray && tk2 isa AbstractArray
                    mul!(tk, tk1, tk2, α, true)
                elseif tk1 isa AbstractArray
                    axpy!(tk2*α, tk1, tk)
                elseif tk2 isa AbstractArray
                    axpy!(tk1*α, tk2, tk)
                else
                    t[k] += tk1*tk2*α
                end
            end
        else
            tk *= β
            for k1 = max(0, k-K2):min(k, K1)
                tk += t1[k1]*t2[k-k1]*α
            end
            t[k] = tk
        end
    end
    return t
end

function differentiate(t::TaylorSeries)
    if degree(t) >= 1
        return TaylorSeries([k*t[k] for k = 1:degree(t)], offset(t))
    else
        return zero(t)
    end
end
function integrate(t::TaylorSeries, (a,b)::Tuple{Real,Real})
    s = t[0]*(b-a)
    c = offset(t)
    for k = 1:degree(t)
        s += t[k]*((b-c)^(k+1) - (a-c)^(k+1))/(k+1)
    end
    return s
end

function localdot(t1::TaylorSeries, t2::TaylorSeries)
    @assert offset(t1) == offset(t2)
    K1 = degree(t1)
    K2 = degree(t2)
    coeffs = let K = K1 + K2
        [sum(dot(t1[k1],t2[k-k1]) for k1 = max(0,k-K2):min(k,K1)) for k=0:K]
    end
    return TaylorSeries(coeffs, offset(t1))
end

LinearAlgebra.tr(t::TaylorSeries) = TaylorSeries(map(tr, t.coeffs), offset(t))

Base.real(t::TaylorSeries) = TaylorSeries(map(real, t.coeffs), offset(t))
Base.imag(t::TaylorSeries) = TaylorSeries(map(imag, t.coeffs), offset(t))

function Base.inv(t::TaylorSeries; tol = 1e-10, Kmax = 10*degree(t))
    t0inv = inv(t[0])
    coeffs = [t0inv]
    K = degree(t)
    for k = 1:Kmax
        invtk = -t0inv*sum(coeffs[i+1]*t[k-i] for i = max(0,k-K):(k-1))
        if norm(invtk) < tol && norm(coeffs[end]) < tol
            break
        end
        push!(coeffs, invtk)
    end
    return TaylorSeries(coeffs, offset(t))
end
