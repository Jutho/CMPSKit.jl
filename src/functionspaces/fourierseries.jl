# Type definition
struct FourierSeries{T,S<:Real} <: FunctionSeries{T}
    coeffs::Vector{T}
    period::S
    function FourierSeries(coeffs::Vector{T}, period=1) where {T}
        isodd(length(coeffs)) ||
            throw(ArgumentError("expect odd length for coefficient vector"))
        S = typeof(period)
        new{T,S}(coeffs, period)
    end
end

# Basic properties
period(f::FourierSeries) = f.period
domain(f::FourierSeries) = (zero(period(f)), period(f))

nummodes(f::FourierSeries) = (length(f.coeffs)-1) >> 1
coefficients(f::FourierSeries) = f.coeffs

function Base.:(==)(f1::FourierSeries, f2::FourierSeries)
    period(f1) == period(f2) || return false
    K = max(nummodes(f1), nummodes(f2))
    for k = -K:K
        f1[k] == f2[k] || return false
    end
    return true
end

# Indexing, getting and setting coefficients
Base.eachindex(f::FourierSeries) = UnitRange(-nummodes(f), nummodes(f))
function Base.getindex(f::FourierSeries, k)
    @inbounds if abs(k) > nummodes(f)
        return zero(f.coeffs[1])
    else
        j = ifelse(k == 0, 1, ifelse(k < 0, -2*k+1, 2*k))
        return f.coeffs[j]
    end
end
function Base.setindex!(f::FourierSeries, v, k)
    if abs(k) > nummodes(f)
        setnummodes!(f, abs(k))
    end
    j = ifelse(k == 0, 1, ifelse(k < 0, -2*k+1, 2*k))
    @inbounds setindex!(f.coeffs, v, j)
end

# Use as function
function (f::FourierSeries)(x̃)
    xred = x̃/period(f)
    x = mod1(xred, one(xred))
    K = nummodes(f)
    factor = 2pi
    s, c = sincos(factor*0*x)
    v = zero(f[0]) + f[0]*(c+im*s) # for type stability
    for k = 1:K
        s, c = sincos(factor*k*x)
        v += (f[k]+f[-k])*c + im*(f[k]-f[-k])*s
    end
    return v
end

# Change number of coefficients
function truncate!(f::FourierSeries; Kmax::Integer = nummodes(f), tol::Real = 0)
    Ktol = findlast(x->(norm(x) >= tol), f.coeffs)
    Kmax = min(Kmax, Ktol === nothing ? 0 : Ktol>>1)
    if Kmax < nummodes(f)
        resize!(f.coeffs, 2*Kmax+1)
    end
    return f
end

function setnummodes!(f::FourierSeries, K::Int)
    while nummodes(f) < K
        push!(f.coeffs, zero(f.coeffs[1]), zero(f.coeffs[1]))
    end
    if nummodes(f) > K
        resize!(f.coeffs, 2*K+1)
    end
    return f
end

# Special purpose constructor
Base.similar(f::FourierSeries, ::Type{T} = scalartype(f)) where {T} =
    FourierSeries([zero(T)*f[0]], period(f))

Base.zero(f::FourierSeries) = FourierSeries([zero(f[0])], period(f))
Base.one(f::FourierSeries) = FourierSeries([one(f[0])], period(f))

# Arithmetic (out of place)
Base.copy(f::FourierSeries) = FourierSeries(copy.(coefficients(f)), period(f))

Base.:-(f::FourierSeries) = FourierSeries(.-coefficients(f), period(f))
Base.:+(f::FourierSeries) = FourierSeries(.+coefficients(f), period(f))

Base.:*(f::FourierSeries, a::Const) =
    FourierSeries([c*a for c in coefficients(f)], period(f))
Base.:*(a::Const, f::FourierSeries) =
    FourierSeries([a*c for c in coefficients(f)], period(f))
Base.:/(f::FourierSeries, a::Const) =
    FourierSeries([c/a for c in coefficients(f)], period(f))
Base.:\(a::Const, f::FourierSeries) =
    FourierSeries([a\c for c in coefficients(f)], period(f))

function Base.:+(f1::FourierSeries, f2::FourierSeries)
    domain(f1) == domain(f2) || throw(DomainMismatch())
    K1 = nummodes(f1)
    K2 = nummodes(f2)
    K = max(K1, K2)
    f = FourierSeries(sizehint!([f1[0]+f2[0]], 2*K+1), period(f1))
    for k = 1:K
        if k <= K1 && k <= K2
            f[k] = f1[k] + f2[k]
            f[-k] = f1[-k] + f2[-k]
        elseif k <= K1
            f[k] = copy(f1[k])
            f[-k] = copy(f1[-k])
        else
            f[k] = copy(f2[k])
            f[-k] = copy(f2[-k])
        end
    end
    return f
end

function Base.:-(f1::FourierSeries, f2::FourierSeries)
    domain(f1) == domain(f2) || throw(DomainMismatch())
    K1 = nummodes(f1)
    K2 = nummodes(f2)
    K = max(K1, K2)
    f = FourierSeries(sizehint!([f1[0] - f2[0]], 2*K+1), period(f1))
    for k = 1:K
        if k <= K1 && k <= K2
            f[k] = f1[k] - f2[k]
            f[-k] = f1[-k] - f2[-k]
        elseif k <= K1
            f[k] = copy(f1[k])
            f[-k] = copy(f1[-k])
        else
            f[k] = -(f2[k])
            f[-k] = -(f2[-k])
        end
    end
    return f
end

Base.:*(f1::FourierSeries, f2::FourierSeries) = truncmul(f1, f2)

function truncmul(f1::FourierSeries, f2::FourierSeries;
                    Kmax::Integer = nummodes(f1) + nummodes(f2), tol::Real = 0)
    period(f1) == period(f2) || throw(DomainMismatch())
    f = FourierSeries(sizehint!([zero(f1[0])*zero(f2[0])], 2*Kmax+1), period(f1))
    truncmul!(f, f1, f2, true, false; Kmax = Kmax, tol = tol)
    return f
end

function Base.conj(f::FourierSeries)
    fc = FourierSeries(map(conj, coefficients(f)), period(f))
    K = nummodes(f)
    for k = 1:K
        fc[k], fc[-k] = fc[-k], fc[k]
    end
    return fc
end
function Base.adjoint(f::FourierSeries)
    fc = FourierSeries(map(adjoint, coefficients(f)), period(f))
    K = nummodes(f)
    for k = 1:K
        fc[k], fc[-k] = fc[-k], fc[k]
    end
    return fc
end
Base.transpose(f::FourierSeries) = FourierSeries(map(transpose, coefficients(f)), period(f))

LinearAlgebra.tr(f::FourierSeries) = FourierSeries(map(tr, coefficients(f)), period(f))

Base.real(f::FourierSeries) = rmul!(f+conj(f), 1//2)
Base.imag(f::FourierSeries) = rmul!(f-conj(f), 1//(2*im))

# Arithmetic (in place / mutating methods)
function Base.copy!(fdst::FourierSeries, fsrc::FourierSeries)
    period(fdst) == period(fsrc) || throw(DomainMismatch())
    setnummodes!(fdst, nummodes(fsrc))
    if eltype(fdst) <: Number
        copy!(coefficients(fdst), coefficients(fsrc))
    else
        for k in eachindex(fdst)
            copy!(fdst[k], fsrc[k])
        end
    end
    return fdst
end
function LinearAlgebra.rmul!(f::FourierSeries, α::Number)
    if eltype(f) <: Number
        rmul!(coefficients(f), α)
    else
        for k in eachindex(f)
            rmul!(f[k], α)
        end
    end
    return f
end
function LinearAlgebra.lmul!(α::Number, f::FourierSeries)
    if eltype(f) <: Number
        lmul!(α, coefficients(f))
    else
        for k in eachindex(f)
            lmul!(α, f[k])
        end
    end
    return f
end

LinearAlgebra.axpy!(α::Number, fx::FourierSeries, fy::FourierSeries) =
    truncadd!(fy, fx, α)
LinearAlgebra.axpby!(α::Number, fx::FourierSeries, β::Number, fy::FourierSeries) =
    truncadd!(fy, fx, α, β)
LinearAlgebra.mul!(fy::FourierSeries, s::Number, fx::FourierSeries, α = true, β = false) =
    truncmul!(fy, s, fx, α, β)
LinearAlgebra.mul!(fy::FourierSeries, fx::FourierSeries, s::Number, α = true, β = false) =
    truncmul!(fy, fx, s, α, β)
LinearAlgebra.mul!(f::FourierSeries, f1::FourierSeries, f2::FourierSeries,
                    α = true, β = false) = truncmul!(f, f1, f2, α, β)

function truncadd!(fy::FourierSeries, fx::FourierSeries, α = true, β = true;
                    Kmax::Integer = max(iszero(β) ? 0 : nummodes(fy), nummodes(fx)),
                    tol::Real = 0)
    period(fx) == period(fy) || throw(DomainMismatch())
    Kx = nummodes(fx)
    Ky = min(Kmax, iszero(β) ? Kx : max(Kx, nummodes(fy)))
    setnummodes!(fy, Ky)
    if eltype(fy) <: Number
        if Ky > Kx
            LinearAlgebra.axpby!(α, coefficients(fx), β, view(coefficients(fy), 1:2Kx+1))
            lmul!(β, view(coefficients(fy), (2Kx+2):(2Ky+1)))
        else
            LinearAlgebra.axpby!(α, view(coefficients(fx), 1:2Ky+1), β, coefficients(fy))
        end
    else
        Threads.@threads for k in -Ky:Ky
            if abs(k) <= Kx
                axpby!(α, fx[k], β, fy[k])
            elseif !isone(β)
                lmul!(β, fy[k])
            end
        end
    end
    return iszero(tol) ? fy : truncate!(fy; tol = tol)
end

truncmul!(fdst::FourierSeries, α₁::Number, fsrc::FourierSeries, α₂ = true, β = false;
            kwargs...) = truncadd!(fdst, fsrc, α₁ * α₂, β; kwargs...)
truncmul!(fdst::FourierSeries, fsrc::FourierSeries, α₁::Number, α₂ = true, β = false;
            kwargs...) = truncadd!(fdst, fsrc, α₁ * α₂, β; kwargs...)

function truncmul!(f::FourierSeries, f1::FourierSeries, f2::FourierSeries,
                    α = true, β = false;
                    Kmax::Integer = max(iszero(β) ? 0 : nummodes(f),
                                                        nummodes(f1)+nummodes(f2)),
                    tol::Real = 0)
    period(f) == period(f1) == period(f2) || throw(DomainMismatch())
    K1 = nummodes(f1)
    K2 = nummodes(f2)
    K = min(Kmax, max(iszero(β) ? 0 : nummodes(f), K1+K2))
    setnummodes!(f, K)
    Threads.@threads for k = -K:K
        fk = f[k]
        if fk isa AbstractArray
            T = eltype(fk)
            if β == 0
                fill!(fk, T(β))
            elseif β != 1
                rmul!(fk, T(β))
            end
            for k1 = max(-K1, k-K2):min(K1, k+K2)
                k2 = k - k1
                fk1 = f1[k1]
                fk2 = f2[k2]
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
                fk1 = f1[k1]
                fk2 = f2[k2]
                fk += fk1*fk2*α
            end
            f[k] = fk
        end
    end
    return iszero(tol) ? f : truncate!(f; tol = tol)
end

# Inner product and norm
function localdot(f1::FourierSeries, f2::FourierSeries)
    period(f1) == period(f2) || throw(DomainMismatch())
    K1 = nummodes(f1)
    K2 = nummodes(f2)
    K = K1+K2
    f = FourierSeries(sizehint!([zero(dot(f1[0], f2[0]))], 2*K+1), period(f1))
    for k = -K:K
        fk = f[k]
        for k1 = max(-K1, -K2-k):min(K1, K2-k)
            k2 = k + k1
            fk += dot(f1[k1], f2[k2])
        end
        f[k] = fk
    end
    return f
end

function LinearAlgebra.dot(f1::FourierSeries, f2::FourierSeries)
    domain(f1) == domain(f2) || throw(DomainMismatch())
    K = min(nummodes(f1), nummodes(f2))
    return sum(k->dot(f1[k], f2[k]), -K:K)
end
LinearAlgebra.norm(f::FourierSeries) = norm(norm(fk) for fk in coefficients(f))

# Differentiate and integrate
function differentiate(f::FourierSeries)
    df = 0im * f
    ω = (2*pi)/period(f)
    K = nummodes(f)
    for k = 1:K
        ν = im*ω*k
        df[-k] = -ν*f[-k]
        df[+k] = +ν*f[+k]
    end
    return df
end
function integrate(f::FourierSeries, (a,b) = domain(F))
    p = period(f)
    if b-a == p
        return f[0]*p
    else
        error("not yet implemented")
    end
end

fit(f, ::Type{FourierSeries}, (a,b)::Tuple{Real,Real}; kwargs...) =
    fit(f, FourierSeries, b-a; kwargs...)

function fit(f, ::Type{FourierSeries}, period = 1; Kmax = 10, numpoints = 2*Kmax+1)
    K = Kmax
    x = (0:(numpoints-1)) * (period / numpoints)
    eox = exp.((im*2*pi/period) .* x)
    coeffs = similar(eox, (numpoints, 2*Kmax+1))
    coeffs[:, 1] .= 1
    for k = 1:Kmax
        coeffs[:, 2*k] .= view(coeffs, :, k == 1 ? 1 : 2*k-2) .* eox
        coeffs[:, 2*k+1] .= view(coeffs, :, 2*k-1) .* conj.(eox)
    end
    fx = f.(x)
    fs = FourierSeries(pinv(coeffs)*fx, period)
    if all(isreal, fx)
        return real(fs)
    else
        return fs
    end
end

# Inverse and square root
# ?
