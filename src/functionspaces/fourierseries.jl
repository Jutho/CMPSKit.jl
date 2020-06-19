# Type definition
struct FourierSeries{T,S<:Real} <: FunctionSpace{T}
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

nummodes(F::FourierSeries) = (length(F.coeffs)-1) >> 1
coefficients(F::FourierSeries) = F.coeffs

function Base.:(==)(F1::FourierSeries, F2::FourierSeries)
    F1.period == F2.period || return false
    K = max(nummodes(F1), nummodes(F2))
    for k = -K:K
        F1[k] == F2[k] || return false
    end
    return true
end

# Indexing, getting and setting coefficients
Base.eachindex(F::FourierSeries) = UnitRange(-nummodes(F), nummodes(F))
function Base.getindex(F::FourierSeries, k)
    if abs(k) > nummodes(F)
        return zero(F.coeffs[1])
    else
        j = ifelse(k == 0, 1, ifelse(k < 0, -2*k+1, 2*k))
        @inbounds getindex(F.coeffs, j)
    end
end
function Base.setindex!(F::FourierSeries, v, k)
    if abs(k) > nummodes(F)
        setnummodes!(F, abs(k))
    end
    j = ifelse(k == 0, 1, ifelse(k < 0, -2*k+1, 2*k))
    @inbounds setindex!(F.coeffs, v, j)
end

# Use as function
function (F::FourierSeries)(x)
    xred = x/period(F)
    x = mod1(xred, one(xred))
    K = nummodes(F)
    factor = 2pi
    f = zero(F[0]) + F[0]*(cos(0*factor)+im*sin(0*factor)) # for type stability
    for k = 1:K
        f += (F[k]+F[-k])*cos(factor*k*x) + im*(F[k]-F[-k])*sin(factor*k*x)
    end
    return f
end

# Change number of coefficients
function truncate!(F::FourierSeries; Kmax::Integer = nummodes(F), tol::Real = 0)
    Ktol = findlast(x->norm(x)>=tol, F.coeffs)
    Kmax = min(Kmax, Ktol === nothing ? 0 : Ktol>>1)
    if Kmax < nummodes(F)
        resize!(F.coeffs, 2*Kmax+1)
    end
    return F
end

function setnummodes!(F::FourierSeries, K::Int)
    while nummodes(F) < K
        push!(F.coeffs, zero(F.coeffs[1]), zero(F.coeffs[1]))
    end
    if nummodes(F) > K
        resize!(F.coeffs, 2*K+1)
    end
    return F
end

# special purpose constructor
Base.similar(F::FourierSeries, ::Type{T} = scalartype(F)) where {T} =
    FourierSeries([zero(T)*F[0]], period(F))

Base.zero(F::FourierSeries) = FourierSeries([zero(F[0])], period(F))
Base.one(F::FourierSeries) = FourierSeries([one(F[0])], period(F))

# Arithmetic (out of place)
Base.copy(F::FourierSeries) = FourierSeries(copy.(coefficients(F)), period(F))

Base.:-(F::FourierSeries) = FourierSeries(.-coefficients(F), period(F))

const Const = Union{Number,AbstractArray}
Base.:*(F::FourierSeries, a::Const) =
    FourierSeries([f*a for f in coefficients(F)], period(F))
Base.:*(a::Const, F::FourierSeries) =
    FourierSeries([a*f for f in coefficients(F)], period(F))
Base.:/(F::FourierSeries, a::Const) =
    FourierSeries([f/a for f in coefficients(F)], period(F))
Base.:\(a::Const, F::FourierSeries) =
    FourierSeries([a\f for f in coefficients(F)], period(F))

function Base.:+(F1::FourierSeries, F2::FourierSeries)
    domain(F1) == domain(F2) || throw(DomainMismatch())
    K1 = nummodes(F1)
    K2 = nummodes(F2)
    K = max(K1, K2)
    F = FourierSeries(sizehint!([F1[0]+F2[0]], 2*K+1), period(F1))
    for k = 1:K
        if k <= K1 && k <= K2
            F[k] = F1[k] + F2[k]
            F[-k] = F1[-k] + F2[-k]
        elseif k <= K1
            F[k] = copy(F1[k])
            F[-k] = copy(F1[-k])
        else
            F[k] = copy(F2[k])
            F[-k] = copy(F2[-k])
        end
    end
    return F
end

function Base.:-(F1::FourierSeries, F2::FourierSeries)
    domain(F1) == domain(F2) || throw(DomainMismatch())
    K1 = nummodes(F1)
    K2 = nummodes(F2)
    K = max(K1, K2)
    F = FourierSeries(sizehint!([F1[0] - F2[0]], 2*K+1), period(F1))
    for k = 1:K
        if k <= K1 && k <= K2
            F[k] = F1[k] - F2[k]
            F[-k] = F1[-k] - F2[-k]
        elseif k <= K1
            F[k] = copy(F1[k])
            F[-k] = copy(F1[-k])
        else
            F[k] = -(F2[k])
            F[-k] = -(F2[-k])
        end
    end
    return F
end

Base.:*(F1::FourierSeries, F2::FourierSeries) = truncmul(F1, F2)

function truncmul(F1::FourierSeries, F2::FourierSeries;
                    Kmax::Integer = nummodes(F1) + nummodes(F2), tol::Real = 0)
    domain(F1) == domain(F2) || throw(DomainMismatch())
    F = FourierSeries(sizehint!([zero(F1[0])*zero(F2[0])], 2*Kmax+1), period(F1))
    truncmul!(F, F1, F2, true, false; Kmax = Kmax, tol = tol)
    return F
end

function Base.conj(F::FourierSeries)
    Fc = FourierSeries(map(conj, coefficients(F)), period(F))
    K = nummodes(F)
    for k = 1:K
        Fc[k], Fc[-k] = Fc[-k], Fc[k]
    end
    return Fc
end
function Base.adjoint(F::FourierSeries)
    Fc = FourierSeries(map(adjoint, coefficients(F)), period(F))
    K = nummodes(F)
    for k = 1:K
        Fc[k], Fc[-k] = Fc[-k], Fc[k]
    end
    return Fc
end
Base.transpose(F::FourierSeries) = FourierSeries(map(transpose, coefficients(F)), period(F))

LinearAlgebra.tr(F::FourierSeries) = FourierSeries(map(tr, coefficients(F)), period(F))

Base.real(F::FourierSeries) = rmul!(F+conj(F), 1//2)
Base.imag(F::FourierSeries) = rmul!(F-conj(F), 1//(2*im))

# Arithmetic (in place / mutating methods)
function Base.copy!(Fdst::FourierSeries, Fsrc::FourierSeries)
    domain(Fdst) == domain(Fsrc) || throw(DomainMismatch())
    setnummodes!(Fdst, nummodes(Fsrc))
    if eltype(Fdst) <: Number
        copy!(coefficients(Fdst), coefficients(Fsrc))
    else
        for k in eachindex(Fdst)
            copy!(Fdst[k], Fsrc[k])
        end
    end
    return Fdst
end
function LinearAlgebra.rmul!(F::FourierSeries, α::Number)
    if eltype(F) <: Number
        rmul!(coefficients(F), α)
    else
        for k in eachindex(F)
            rmul!(F[k], α)
        end
    end
    return F
end
function LinearAlgebra.lmul!(α::Number, F::FourierSeries)
    if eltype(F) <: Number
        lmul!(α, coefficients(F))
    else
        for k in eachindex(F)
            lmul!(α, F[k])
        end
    end
    return F
end

function LinearAlgebra.mul!(Fdst::FourierSeries, α::Number, Fsrc::FourierSeries)
    domain(Fdst) == domain(Fsrc) || throw(DomainMismatch())
    setnummodes!(Fdst, nummodes(Fsrc))
    if eltype(Fdst) <: Number
        mul!(coefficients(Fdst), α, coefficients(Fsrc))
    else
        for k in eachindex(Fdst)
            mul!(Fdst[k], α, Fsrc[k])
        end
    end
    return Fdst
end
function LinearAlgebra.mul!(Fdst::FourierSeries, Fsrc::FourierSeries, α::Number)
    domain(Fdst) == domain(Fsrc) || throw(DomainMismatch())
    setnummodes!(Fdst, nummodes(Fsrc))
    if eltype(Fdst) <: Number
        mul!(coefficients(Fdst), coefficients(Fsrc), α)
    else
        for k in eachindex(Fdst)
            mul!(Fdst[k], Fsrc[k], α)
        end
    end
    return Fdst
end
function LinearAlgebra.axpy!(α::Number, Fx::FourierSeries, Fy::FourierSeries)
    domain(Fx) == domain(Fy) || throw(DomainMismatch())
    Kx = nummodes(Fx)
    setnummodes!(Fy, max(Kx, nummodes(Fy)))
    Ky = nummodes(Fy)
    if eltype(Fy) <: Number
        if Ky > Kx
            LinearAlgebra.axpy!(α, coefficients(Fx), view(coefficients(Fy), 1:2Kx+1))
        else
            LinearAlgebra.axpy!(α, coefficients(Fx), coefficients(Fy))
        end
    else
        K = nummodes(Fx)
        Threads.@threads for k in -K:K
            axpy!(α, Fx[k], Fy[k])
        end
    end
    return Fy
end
function LinearAlgebra.axpby!(α::Number, Fx::FourierSeries, β::Number, Fy::FourierSeries)
    domain(Fx) == domain(Fy) || throw(DomainMismatch())
    Kx = nummodes(Fx)
    setnummodes!(Fy, max(Kx, nummodes(Fy)))
    Ky = nummodes(Fy)
    if eltype(Fy) <: Number
        if Ky > Kx
            LinearAlgebra.axpby!(α, coefficients(Fx), β, view(coefficients(Fy), 1:2Kx+1))
            lmul!(β, view(coefficients(Fy), (2Kx+2):(2Ky+1)))
        else
            LinearAlgebra.axpby!(α, coefficients(Fx), β, coefficients(Fy))
        end
    else
        K = nummodes(Fy)
        Threads.@threads for k in -K:K
            if abs(k) <= Kx
                axpby!(α, Fx[k], β, Fy[k])
            else
                lmul!(β, Fy[k])
            end
        end
    end
    return Fy
end

LinearAlgebra.mul!(F::FourierSeries, F1::FourierSeries, F2::FourierSeries,
                    α = true, β = false) = truncmul!(F, F1, F2, α, β)

function truncmul!(F::FourierSeries, F1::FourierSeries, F2::FourierSeries,
                    α = true, β = false;
                    Kmax::Integer = nummodes(F1)+nummodes(F2), tol::Real = 0)
    domain(F) == domain(F1) == domain(F2) || throw(DomainMismatch())
    K1 = nummodes(F1)
    K2 = nummodes(F2)
    K = min(Kmax, K1+K2)
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
function localdot(F1::FourierSeries, F2::FourierSeries)
    domain(F1) == domain(F2) || throw(DomainMismatch())
    K1 = nummodes(F1)
    K2 = nummodes(F2)
    K = K1+K2
    F = FourierSeries(sizehint!([zero(dot(F1[0], F2[0]))], 2*K+1), period(F1))
    for k = -K:K
        fk = F[k]
        for k1 = max(-K1, -K2-k):min(K1, K2-k)
            k2 = k + k1
            fk += dot(F1[k1], F2[k2])
        end
        F[k] = fk
    end
    return F
end

function LinearAlgebra.dot(F1::FourierSeries, F2::FourierSeries)
    domain(F1) == domain(F2) || throw(DomainMismatch())
    K = min(nummodes(F1), nummodes(F2))
    return sum(k->dot(F1[k], F2[k]), -K:K)
end
LinearAlgebra.norm(F::FourierSeries) = norm(norm(Fk) for Fk in coefficients(F))

function LinearAlgebra.isapprox(x::FourierSeries, y::FourierSeries;
                                atol::Real=0,
                                rtol::Real=defaulttol(x[0]))
    return norm(x-y) <= max(atol, rtol*max(norm(x), norm(y)))
end

function differentiate(F::FourierSeries)
    dF = 0im * F
    ω = (2*pi)/period(F)
    K = nummodes(F)
    for k = 1:K
        f = im*ω*k
        dF[-k] = -f*F[-k]
        dF[+k] = +f*F[+k]
    end
    return dF
end
function integrate(F::FourierSeries, (a,b) = domain(F))
    p = period(F)
    if b-a == p
        return F[0]*p
    else
        error("not yet implemented")
    end
end

# Simple Fourier transform: typically not needed for large number of points, but should
# work with matrix valued functions etc
function fit(f, ::Type{FourierSeries}, period = 1; Kmax = 10, tol = 1e-12)
    K = Kmax
    x = (0:2*K) * (period / (2K+1))
    fx = map(f, x)
    ω = 2*pi*0/period
    integrand = exp.((-im*ω) .* x) .* fx
    F = FourierSeries([sum(integrand)/(2K+1)], period)
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

# function Base.inv(F::FourierSeries; tol=1e-12, Kmax = 2*nummodes(F))
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
#         G = FourierSeries(coeffG)
#         if norm(G*F-one(G))<tol || K == Kmax
#             return G
#         else
#             K = min(2*K, Kmax)
#         end
#     end
# end
#
# function inv2(F::FourierSeries; tol = 1e-12, Kmax = 2*nummodes(F))
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
#         G = FourierSeries(coeffG)
#         if norm(G*F-one(G))<tol || K == Kmax
#             return G
#         else
#             K = min(2*K, Kmax)
#         end
#     end
#     return Fi2
# end
#
# function Base.sqrt(F::FourierSeries; tol=1e-12, Kmax = 200)
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
#         G = FourierSeries(coeffG)
#         if norm(G*G-F)<tol || K == Kmax
#             return G
#         else
#             K = min(2*K, Kmax)
#         end
#     end
# end
