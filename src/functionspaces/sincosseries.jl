# Type definition
# linear combination of constant term, sin(k*π*(x-a)/(b-a)) (for k = 1, 2, ...) and cos(k*π*(x-a)/(b-a)) (for k = 1, 2, ...) terms on the domain (a,b)
struct SinCosSeries{T,S<:Real}
    coeff0::Base.RefValue{T}
    sincoeffs::Vector{T}
    coscoeffs::Vector{T}
    domain::Tuple{S,S}
end
function SinCosSeries(coeff0::T,
                        sincoeffs::Vector{T},
                        coscoeffs::Vector{T},
                        domain::Tuple{Real,Real} = (0,1)) where T
    d = promote(domain...)
    SinCosSeries{T,typeof(d[1])}(Ref(coeff0), sincoeffs, coscoeffs, d)
end

# Basic properties
domain(F::SinCosSeries) = F.domain
period(F::SinCosSeries) = -(domain(F)...)

numcosmodes(F::SinCosSeries) = length(F.coscoeffs)
numsinmodes(F::SinCosSeries) = length(F.sincoeffs)

# Indexing, getting and setting coefficients
eachindex(F::SinCosSeries) =
    Iterators.flatten((0:1:0, 2*(1:numsinmodes(F)) .- 1, 2*(1:numcosmodes(F))))

function Base.getindex(F::SinCosSeries, k::Integer)
    if k == 0
        return F.coeff0[]
    elseif isodd(k)
        l = (k+1)>>1
        if l > numsinmodes(F)
            return zero(F.coeff0[])
        else
            return F.sincoeffs[l]
        end
    else
        l = k>>1
        if l > numcosmodes(F)
            return zero(F.coeff0[])
        else
            return F.coscoeffs[l]
        end
    end
end
function Base.setindex!(F::SinCosSeries, v, k::Integer)
    if k == 0
        F.coeff0[] = v
    elseif isodd(k)
        l = (k+1)>>1
        if l > numsinmodes(F)
            setnumsinmodes!(F, l)
        end
        return F.sincoeffs[l] = v
    else
        l = k>>1
        if l > numcosmodes(F)
            setnumcosmodes!(F, l)
        end
        return F.coscoeffs[l] = v
    end
end
function Base.getindex(F::SinCosSeries, l::Integer, s::Symbol)
    if s == :s || s ==:S
        if l > numsinmodes(F)
            return zero(F.coeff0[])
        else
            return F.sincoeffs[l]
        end
    elseif s == :c || s == :C
        if l > numcosmodes(F)
            return zero(F.coeff0[])
        elseif l == 0
            return F.coeff0[]
        else
            return F.coscoeffs[l]
        end
    else
        return BoundsError(F, (k,s))
    end
end
function Base.setindex!(F::SinCosSeries, v, l::Integer, s::Symbol)
    if s == :s || s == :S
        if l > numsinmodes(F)
            setnumsinmodes!(F, l)
        end
        return F.sincoeffs[l] = v
    elseif s == :c || s == :C
        if l > numcosmodes(F)
            setnumcosmodes!(F, l)
        end
        if l == 0
            return F.coeff0[] = v
        else
            return F.coscoeffs[l] = v
        end
    else
        return BoundsError(F, (k,s))
    end
end

# Use as function
function (F::SinCosSeries)(x)
    a, b = domain(F)
    a <= x <= b || throw(DomainError())
    θ_π = (x-a)/(b-a)
    f = F[0]
    for k = 1:numsinmodes(F)
        f += F[k,:s]*sinpi(k*θ_π)
    end
    for k = 1:numcosmodes(F)
        f += F[k,:c]*cospi(k*θ_π)
    end
    return f
end

# Change number of coefficients
function truncate!(F::SinCosSeries;
                    Kmax::Integer = max(numsinmodes(F), numcosmodes(F)), tol::Real = 0)
    Ksintol = findlast(x->norm(x)>=tol, F.sincoeffs)
    Ksinmax = min(Kmax, Ksintol === nothing ? 0 : Ksintol)
    if Ksinmax < numsinmodes(F)
        resize!(F.sincoeffs, Ksinmax)
    end
    Kcostol = findlast(x->norm(x)>=tol, F.coscoeffs)
    Kcosmax = min(Kmax, Kcostol === nothing ? 0 : Kcostol)
    if Kcosmax < numcosmodes(F)
        resize!(F.coscoeffs, Kcosmax)
    end
    return F
end
function setnumsinmodes!(F::SinCosSeries, K::Int)
    while numsinmodes(F) < K
        push!(F.sincoeffs, zero(F[0]))
    end
    if numsinmodes(F) > K
        resize!(F.sincoeffs, K)
    end
    return F
end
function setnumcosmodes!(F::SinCosSeries, K::Int)
    while numcosmodes(F) < K
        push!(F.coscoeffs, zero(F[0]))
    end
    if numcosmodes(F) > K
        resize!(F.coscoeffs, K)
    end
    return F
end

# Special purpose constructor
function Base.similar(F::SinCosSeries, ::Type{T}) where T
    G0 = similar(F[0], T)
    return SinCosSeries(G0, fill(G0, 0), fill(G0, 0))
end
function Base.zero(F::SinCosSeries)
    G0 = zero(F[0])
    return SinCosSeries(G0, fill(G0, 0), fill(G0, 0))
end
function Base.one(F::SinCosSeries)
    G0 = zero(F[0])
    return SinCosSeries(G0, fill(G0, 0), fill(G0, 0))
end

# Arithmitic (out of place)
function Base.:+(F1::SinCosSeries, F2::SinCosSeries)
    domain(F1) == domain(F2) || throw(DomainError())
    Ks = min(numsinmodes(F1), numsinmodes(F2))
    sincoeffs = [F1[k,:s] + F2[k,:s] for k = 1:Ks]
    for k = Ks+1:numsinmodes(F1)
        push!(sincoeffs, (+1)*F1[k, :s])
    end
    for k = Ks+1:numsinmodes(F2)
        push!(sincoeffs, (+1)*F2[k, :s])
    end

    Kc = min(numcosmodes(F1), numcosmodes(F2))
    coscoeffs = [F1[k,:c] + F2[k,:c] for k = 1:Kc]
    for k = Kc+1:numcosmodes(F1)
        push!(coscoeffs, (+1)*F1[k, :c])
    end
    for k = Kc+1:numcosmodes(F2)
        push!(coscoeffs, (+1)*F2[k, :c])
    end

    return SinCosSeries(F1[0]+F2[0], sincoeffs, coscoeffs, domain(F1))
end

function Base.:-(F1::SinCosSeries, F2::SinCosSeries)
    domain(F1) == domain(F2) || throw(DomainError())
    Ks = min(numsinmodes(F1), numsinmodes(F2))
    sincoeffs = [F1[k,:s] - F2[k,:s] for k = 1:Ks]
    for k = Ks+1:numsinmodes(F1)
        push!(sincoeffs, (+1)*F1[k, :s])
    end
    for k = Ks+1:numsinmodes(F2)
        push!(sincoeffs, (-1)*F2[k, :s])
    end

    Kc = min(numcosmodes(F1), numcosmodes(F2))
    coscoeffs = [F1[k,:c] - F2[k,:c] for k = 1:Kc]
    for k = Kc+1:numcosmodes(F1)
        push!(coscoeffs, (+1)*F1[k, :c])
    end
    for k = Kc+1:numcosmodes(F2)
        push!(coscoeffs, (-1)*F1[k, :c])
    end

    return SinCosSeries(F1[0] - F2[0], sincoeffs, coscoeffs, domain(F1))
end

Base.:*(F::SinCosSeries, a::Number) = SinCosSeries(F[0]*a, F.sincoeffs*a, F.coscoeffs*a)
Base.:*(a::Number, F::SinCosSeries) = SinCosSeries(a*F[0], a*F.sincoeffs, a*F.coscoeffs)
Base.:/(F::SinCosSeries, a::Number) = SinCosSeries(F[0]/a, F.sincoeffs/a, F.coscoeffs/a)
Base.:\(a::Number, F::SinCosSeries) = SinCosSeries(a\F[0], a\F.sincoeffs, a\F.coscoeffs)


*(F1::SinCosSeries, F2::SinCosSeries) = truncmul(F1, F2)

function truncmul(F1::SinCosSeries, F2::SinCosSeries; Kmax = -1, kwargs...)
    domain(F1) == domain(F2) || throw(DomainError())
    F0 = zero(F1[0]) * zero(F2[0])
    Ks = max(numsinmodes(F1)+numcosmodes(F2), numcosmodes(F1)+numsinmodes(F2))
    Kc = max(numsinmodes(F1)+numsinmodes(F2), numcosmodes(F1)+numcosmodes(F2))
    if Kmax >= 0
        Ks = min(Ks, Kmax)
        Kc = min(Kc, Kmax)
    end
    sincoeffs = [copy(F0) for k = 1:Ks]
    coscoeffs = [copy(F0) for k = 1:Kc]
    F = SinCosSeries(F0, sincoeffs, coscoeffs, domain(F1))
    return mul!(F, F1, F2, true, true; Kmax = Kmax, kwargs...)
end


# Arithmetic (in place / mutating methods)
function Base.copy!(Fdst::SinCosSeries, Fsrc::SinCosSeries)
    domain(Fdst) == domain(Fsrc) || throw(DomainError())
    setnumsinmodes!(Fdst, numsinmodes(Fsrc))
    setnumcosmodes!(Fdst, numcosmodes(Fsrc))
    @inbounds for i in eachindex(Fsrc)
        Fdst[i] = Fsrc[i]
    end
    return Fdst
end
function LinearAlgebra.rmul!(F::SinCosSeries{<:Number}, α)
    @inbounds for i in eachindex(F)
        F[i] *= α
    end
    return F
end
function LinearAlgebra.lmul!(α, F::SinCosSeries{<:Number})
    @inbounds for i in eachindex(F)
        F[i] *= α
    end
    return F
end
function LinearAlgebra.rmul!(F::SinCosSeries{<:AbstractArray}, α)
    @inbounds for i in eachindex(F)
        F[i] = rmul!(F[i], α)
    end
    return F
end
function LinearAlgebra.lmul!(α, F::SinCosSeries{<:AbstractArray})
    @inbounds for i in eachindex(F)
        F[i] = lmul!(α, F[i])
    end
    return F
end
function LinearAlgebra.mul!(Fdst::SinCosSeries, α, Fsrc::SinCosSeries)
    domain(Fdst) == domain(Fsrc) || throw(DomainError())
    setnumsinmodes!(Fdst, numsinmodes(Fsrc))
    setnumcosmodes!(Fdst, numcosmodes(Fsrc))
    @inbounds for i in eachindex(Fsrc)
        Fdst[i] = α * Fsrc[i]
    end
    return Fdst
end
function LinearAlgebra.mul!(Fdst::SinCosSeries, Fsrc::SinCosSeries, α)
    domain(Fdst) == domain(Fsrc) || throw(DomainError())
    setnumsinmodes!(Fdst, numsinmodes(Fsrc))
    setnumcosmodes!(Fdst, numcosmodes(Fsrc))
    @inbounds for i in eachindex(Fsrc)
        Fdst[i] = Fsrc[i] * α
    end
    return Fdst
end
function LinearAlgebra.axpy!(α, Fx::SinCosSeries, Fy::SinCosSeries)
    domain(Fy) == domain(Fx) || throw(DomainError())
    setnumsinmodes!(Fy, max(numsinmodes(Fx), numsinmodes(Fy)))
    setnumcosmodes!(Fy, max(numcosmodes(Fx), numcosmodes(Fy)))
    for i in eachindex(Fx)
        Fy[i] += α*Fx[i]
    end
    return Fy
end
function LinearAlgebra.axpby!(α, Fx::SinCosSeries, β, Fy::SinCosSeries)
    domain(Fy) == domain(Fx) || throw(DomainError())
    for i in numsinmodes(Fx)+1:numsinmodes(Fy)
        Fy[i, :s] *= β
    end
    for i in numcosmodes(Fx)+1:numcosmodes(Fy)
        Fy[i, :c] *= β
    end
    setnumsinmodes!(Fy, max(numsinmodes(Fx), numsinmodes(Fy)))
    setnumcosmodes!(Fy, max(numcosmodes(Fx), numcosmodes(Fy)))
    for i in eachindex(Fx)
        Fy[i] = α*Fx[i] + β*Fy[i]
    end
    return Fy
end

function LinearAlgebra.mul!(F::SinCosSeries, F1::SinCosSeries, F2::SinCosSeries,
                                α = true, β = false)
    domain(F) == domain(F1) == domain(F2) || throw(DomainError())
    if β != true
        rmul!(F, β)
    end
    for k in eachindex(F1)
        f1k = F1[k]
        for l in eachindex(F2)
            f2l = F2[l]
            for (p,s) in SimpsonIterator(k,l)
                if (iseven(p) && p <= 2*numcosmodes(F)) ||
                    (isodd(p) && p <= 2*numsinmodes(F)-1)
                    fp = F[p]
                    if fp isa AbstractArray
                        if f1k isa AbstractArray && f2l isa AbstractArray
                                mul!(fp, f1k, f2l, s * α, true)
                        elseif f2l isa Number
                            axpy!((s * α) * f2l, f1k, fp)
                        elseif f1k isa Number
                            axpy!((s * α) * f1k, f2l, fp)
                        else
                            fp .+= (f1k * f2l) .* (s * α)
                        end
                    else
                        F[p] += (f1k * f2l) * (s * α)
                    end
                end
            end
        end
    end
    return F
end


# Inner product and norm
function LinearAlgebra.dot(F1::SinCosSeries, F2::SinCosSeries)
    domain(F1) == domain(F2) || throw(DomainError())
    a, b = domain(F1)
    L = (b-a)/1
    Lpi = L/π
    s = L*dot(F1[0], F2[0])
    for k = 1:2:numsinmodes(F1)
        s += (2*Lpi/k)*dot(F1[k,:s], F2[0])
    end
    for l = 1:2:numsinmodes(F2)
        s += (2*Lpi/l)*dot(F1[0], F2[l,:s])
    end
    for k = 1:min(numcosmodes(F1), numcosmodes(F2))
        s += L/2*dot(F1[k,:c], F2[k,:c])
    end
    for k = 1:min(numsinmodes(F1), numsinmodes(F2))
        s += L/2*dot(F1[k,:s], F2[k,:s])
    end
    for k = 1:numcosmodes(F1)
        for l = 1:numsinmodes(F2)
            if isodd(k+l)
                s += Lpi*dot(F1[k,:c], F2[l,:s])*(2*l)/(l^2 - k^2)
            end
        end
    end
    for k = 1:numsinmodes(F1)
        for l = 1:numcosmodes(F2)
            if isodd(k+l)
                s += Lpi*dot(F1[k,:s], F2[l,:c])*(2*k)/(k^2 - l^2)
            end
        end
    end
    return s
end
function LinearAlgebra.norm(F::SinCosSeries)
    a, b = domain(F)
    L = (b-a)/1
    Lpi = L/π
    s = L*norm(F[0])^2
    for k = 1:2:numsinmodes(F)
        s += (4*Lpi/k)*real(dot(F[k,:s], F[0]))
    end
    for k = 1:numcosmodes(F)
        s += L/2*norm(F[k,:c])^2
    end
    for k = 1:numsinmodes(F)
        s += L/2*norm(F[k,:s])^2
    end
    for k = 1:numcosmodes(F)
        for l = 1:numsinmodes(F)
            if isodd(k+l)
                s += Lpi*real(dot(F[k,:c], F[l,:s]))*(4*l)/(l^2 - k^2)
            end
        end
    end
    return sqrt(s)
end


Base.conj(F::SinCosSeries) =
    SinCosSeries(conj(F[0]), conj.(F.sincoeffs), conj.(F.coscoeffs), domain(F))
Base.adjoint(F::SinCosSeries) =
    SinCosSeries(adjoint(F[0]), adjoint.(F.sincoeffs), adjoint.(F.coscoeffs), domain(F))
Base.transpose(F::SinCosSeries) =
    SinCosSeries(transpose(F[0]),
                    transpose.(F.sincoeffs),
                    transpose.(F.coscoeffs),
                    domain(F))


function differentiate(F::SinCosSeries)
    a, b = domain(F)
    πL = π/(b-a)
    sincoeffs = [-πL*k*F[k,:c] for k in 1:numcosmodes(F)]
    coscoeffs = [+πL*k*F[k,:s] for k in 1:numsinmodes(F)]
    return SinCosSeries(zero(F[0]), sincoeffs, coscoeffs, domain(F))
end
function integrate(F::SinCosSeries, interval::Tuple{Real,Real} = domain(F))
    @assert interval == domain(F)
    a, b = interval
    Lπ = (b-a)/π
    s = ((b-a)/1)*F[0]
    for k = 1:2:numsinmodes(F)
        s += (2*Lπ/k)*F[k,:s]
    end
    return s
end

# Note that k == 0 => constant, iseven(k) => cos((k/2)*pi*x/L, isodd(l) => sin((l+1)/2*pi*x/L)
struct SimpsonIterator
    k::Int
    l::Int
end
function Base.iterate(it::SimpsonIterator, state = 0)
    k, l = it.k, it.l
    if k == 0
        if state == 0
            return l=>1//1, 1
        else
            return nothing
        end
    elseif l == 0
        if state == 0
            return k=>1//1, 1
        else
            return nothing
        end
    elseif isodd(k) && isodd(l) # sin * sin
        if state == 0
            return (k+l+2)=>-1//2, 1
        elseif state == 1
            return abs(k-l)=>1//2, 2
        else
            return nothing
        end
    elseif iseven(k) && iseven(l) # cos * cos
        if state == 0
            return k+l=>1//2, 1
        elseif state == 1
            return abs(k-l)=>1//2, 2
        else
            return nothing
        end
    elseif iseven(k) && isodd(l) # cos * sin
        if state == 0
            return k+l=>1//2, 1
        elseif state == 1 && k != l+1
            return (k < l ? (l-k)=>1//2 : (k-l-2)=>-1//2), 2
        else
            return nothing
        end
    else # sin * cos
        if state == 0
            return k+l=>1//2, 1
        elseif state == 1 && k+1 != l
            return (k+1 > l ? (k-l)=>1//2 : (l-k-2)=>-1//2), 2
        else
            return nothing
        end
    end
end
