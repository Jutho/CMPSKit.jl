# In principle, FiniteCMPSTangent has the same fields as FiniteCMPS.
# However, instances of FiniteCMPSTangent constitute a vector space, and we can define the # necessary methods to allow using them as such in KrylovKit.jl.
# Furthermore, we explicitly store the base point and the indices of the original tent
# functions, as we might actually be working on a finer grid than the optimization grid.
mutable struct FiniteCMPSTangent{T<:MatrixFunction, N, V<:AbstractVector, I}
    base::FiniteCMPS{T, N, V}
    dQ::T
    dRs::NTuple{N,T}
    dvL::V
    dvR::V
    indices::I
    function FiniteCMPSTangent(Ψ::FiniteCMPS{T, N, V}, dQ::T, dRs::NTuple{N,T}, dvL::V, dvR::V, indices = 1:length(nodes(dQ))) where {T,N,V}
        grid = nodes(Ψ.Q)
        grid == nodes(dQ) || throw(DomainMismatch())
        for dR in dRs
            grid == nodes(dR) || throw(DomainMismatch())
        end
        a = first(grid)
        Qa = Ψ.Q(a)
        length(dvL) == length(dvR) == size(Qa,1) || throw(DimensionMismatch())
        size(dQ(a)) == size(Qa) || throw(DimensionMismatch())
        for dR in dRs
            size(dR(a)) == size(Qa) || throw(DimensionMismatch())
        end

        first(indices) == 1
        @assert last(indices) == length(grid)
        @assert issorted(indices)

        return new{T,N,V,typeof(indices)}(Ψ, dQ, dRs, dvL, dvR, indices)
    end
end
domain(Φ::FiniteCMPSTangent) = domain(Φ.dQ)
base(Φ::FiniteCMPSTangent) = Φ.base

Base.iterate(Φ::FiniteCMPSTangent, args...) = iterate((Φ.dQ, Φ.dRs, Φ.dvL, Φ.dvR), args...)

for f in (:copy, :zero, :similar)
    @eval Base.$f(Φ::FiniteCMPSTangent) =
        FiniteCMPSTangent(Φ.base, $f(Φ.dQ), $f.(Φ.dRs), $f(Φ.dvL), $f(Φ.dvR), Φ.indices)
end

function Base.copy!(Φ₁::FiniteCMPSTangent, Φ₂::FiniteCMPSTangent)
    (base(Φ₁) === base(Φ₂) && Φ₁.indices === Φ₂.indices) || throw(DomainMismatch())

    copy!(Φ₁.dvL, Φ₂.dvL)
    copy!(Φ₁.dvR, Φ₂.dvR)
    copy!(Φ₁.dQ, Φ₂.dQ)
    copy!.(Φ₁.dRs, Φ₂.dRs)
    return Φ₁
end

# Basic out-of-place arithmitic
Base.:-(Φ::FiniteCMPSTangent) =
    FiniteCMPSTangent(base(Φ), -Φ.dQ, .- Φ.dRs, -Φ.dvL, -Φ.dvR, Φ.indices)
Base.:*(α, Φ::FiniteCMPSTangent) =
        FiniteCMPSTangent(base(Φ), α * Φ.dQ, α .* Φ.dRs, α * Φ.dvL, α * Φ.dvR, Φ.indices)
Base.:\(α, Φ::FiniteCMPSTangent) =
        FiniteCMPSTangent(base(Φ), α \ Φ.dQ, α .\ Φ.dRs, α \ Φ.dvL, α \ Φ.dvR, Φ.indices)
Base.:*(Φ::FiniteCMPSTangent, α) =
        FiniteCMPSTangent(base(Φ), Φ.dQ * α, Φ.dRs .* α, Φ.dvL * α, Φ.dvR * α, Φ.indices)
Base.:/(Φ::FiniteCMPSTangent, α) =
        FiniteCMPSTangent(base(Φ), Φ.dQ / α, Φ.dRs ./ α, Φ.dvL / α, Φ.dvR / α, Φ.indices)
function Base.:+(Φ₁::FiniteCMPSTangent, Φ₂::FiniteCMPSTangent)
    (base(Φ₁) == base(Φ₂) && Φ₁.indices == Φ₂.indices) || throw(DomainMismatch())

    return FiniteCMPSTangent(base(Φ₁),
                                Φ₁.dQ + Φ₂.dQ,
                                Φ₁.dRs .+ Φ₂.dRs,
                                Φ₁.dvL + Φ₂.dvL,
                                Φ₁.dvR + Φ₂.dvR,
                                Φ₁.indices)
end
function Base.:-(Φ₁::FiniteCMPSTangent, Φ₂::FiniteCMPSTangent)
    (base(Φ₁) == base(Φ₂) && Φ₁.indices == Φ₂.indices) || throw(DomainMismatch())

    return FiniteCMPSTangent(base(Φ₁),
                                Φ₁.dQ - Φ₂.dQ,
                                Φ₁.dRs .- Φ₂.dRs,
                                Φ₁.dvL - Φ₂.dvL,
                                Φ₁.dvR - Φ₂.dvR,
                                Φ₁.indices)
end

# In-place arithmitic
function LinearAlgebra.axpy!(α, Φ₁::FiniteCMPSTangent, Φ₂::FiniteCMPSTangent)
    (base(Φ₁) == base(Φ₂) && Φ₁.indices == Φ₂.indices) || throw(DomainMismatch())

    axpy!(α, Φ₁.dvL, Φ₂.dvL)
    axpy!(α, Φ₁.dvR, Φ₂.dvR)
    axpy!(α, Φ₁.dQ, Φ₂.dQ)
    axpy!.(α, Φ₁.dRs, Φ₂.dRs)
    return Φ₂
end
function LinearAlgebra.axpby!(α, Φ₁::FiniteCMPSTangent, β, Φ₂::FiniteCMPSTangent)
    (base(Φ₁) == base(Φ₂) && Φ₁.indices == Φ₂.indices) || throw(DomainMismatch())

    axpby!(α, Φ₁.dvL, β, Φ₂.dvL)
    axpby!(α, Φ₁.dvR, β, Φ₂.dvR)
    axpby!(α, Φ₁.dQ, β, Φ₂.dQ)
    axpby!.(α, Φ₁.dRs, β, Φ₂.dRs)
    return Φ₂
end
function LinearAlgebra.lmul!(α, Φ::FiniteCMPSTangent)
    lmul!(α, Φ.dvL)
    lmul!(α, Φ.dvR)
    lmul!(α, Φ.dQ)
    lmul!.(α, Φ.dRs)
    return Φ
end
function LinearAlgebra.rmul!(Φ::FiniteCMPSTangent, α)
    rmul!(Φ.dvL, α)
    rmul!(Φ.dvR, α)
    rmul!(Φ.dQ, α)
    rmul!.(Φ.dRs, α)
    return Φ
end
function LinearAlgebra.mul!(Φ₁::FiniteCMPSTangent, α, Φ₂::FiniteCMPSTangent)
    (base(Φ₁) == base(Φ₂) && Φ₁.indices == Φ₂.indices) || throw(DomainMismatch())

    mul!(Φ₁.dvL, α, Φ₂.dvL)
    mul!(Φ₁.dvR, α, Φ₂.dvR)
    mul!(Φ₁.dQ, α, Φ₂.dQ)
    mul!.(Φ₁.dRs, α, Φ₂.dRs)
    return Φ₁
end
function LinearAlgebra.mul!(Φ₁::FiniteCMPSTangent, Φ₂::FiniteCMPSTangent, α)
    (base(Φ₁) == base(Φ₂) && Φ₁.indices == Φ₂.indices) || throw(DomainMismatch())

    mul!(Φ₁.dvL, Φ₂.dvL, α)
    mul!(Φ₁.dvR, Φ₂.dvR, α)
    mul!(Φ₁.dQ, Φ₂.dQ, α)
    mul!.(Φ₁.dRs, Φ₂.dRs, α)
    return Φ₁
end

# We choose `dot` to represent a standard Euclidean norm.
# The actual metric of the manifold will be implemented as a preconditioner.
function LinearAlgebra.dot(Φ₁::FiniteCMPSTangent, Φ₂::FiniteCMPSTangent)
    (base(Φ₁) == base(Φ₂) && Φ₁.indices == Φ₂.indices) || throw(DomainMismatch())

    ind = Φ₁.indices
    s = dot(Φ₁.dvL, Φ₂.dvL) + dot(Φ₁.dvR , Φ₂.dvR)
    s += dot(view(nodevalues(Φ₁.dQ), ind), view(nodevalues(Φ₂.dQ), ind))
    for (dR₁, dR₂) in zip(Φ₁.dRs, Φ₂.dRs)
        s += dot(view(nodevalues(dR₁), ind), view(nodevalues(dR₂), ind))
    end
    return s
end
function LinearAlgebra.norm(Φ::FiniteCMPSTangent)
    ind = Φ.indices
    s = hypot(norm(Φ.dvL), norm(Φ.dvR))
    s = hypot(s, norm(view(nodevalues(Φ.dQ), ind)))
    for dR in Φ.dRs
        s = hypot(s, norm(view(nodevalues(dR), ind)))
    end
    return s
end

# Given the functional derivatives of some object with respect to Q, R and ∂R as instances
# of `AbstractPiecewise`, compute the corresponding `PiecewiseLinear` version resulting from
# applying the chain rule
function _project(𝒬̅, ℛ̅s, ∂ℛ̅s = nothing; gradindices = 1:length(nodes(𝒬̅)))
    (a, b) = domain(𝒬̅)
    grid = collect(nodes(𝒬̅))

    # Compute gradients with respect to PiecewiseLinear parameters
    Q̄ = [zero(𝒬̅(a)) for _ = 1:length(gradindices)]
    R̄s = map(ℛ̅->[zero(ℛ̅(a)) for _ = 1:length(gradindices)], ℛ̅s)

    k = gradindices[1] # == 1
    knext = gradindices[2]
    xc = grid[k]
    xb = grid[knext]
    t = TaylorSeries([1,-1/(xb-xc)], xc)
    Q̄i = Q̄[1]
    for l = k:(knext-1)
        t = shift!(t, offset(𝒬̅[l]))
        Q̄i .+= integrate(𝒬̅[l] * t, (grid[l], grid[l+1]))
    end
    Q̄[1] = Q̄i

    for i = 2:length(gradindices)-1
        k = gradindices[i]
        kprev = gradindices[i-1]
        knext = gradindices[i+1]
        xa = grid[kprev]
        xc = grid[k]
        xb = grid[knext]

        Q̄i = Q̄[i]
        R̄is = getindex.(R̄s, i)
        t = TaylorSeries([0,1/(xc-xa)], xa)
        for l = kprev:(k-1)
            t = shift!(t, offset(𝒬̅[l]))
            Q̄i .+= integrate(𝒬̅[l] * t, (grid[l], grid[l+1]))
            for (R̄i, ℛ̅) in zip(R̄is, ℛ̅s)
                R̄i .+= integrate(ℛ̅[l] * t, (grid[l], grid[l+1]))
            end
            if !isnothing(∂ℛ̅s)
                for (R̄i, ∂ℛ̅) in zip(R̄is, ∂ℛ̅s)
                    R̄i .+= integrate(∂ℛ̅[l]/(xc-xa), (grid[l], grid[l+1]))
                end
            end
        end
        t = TaylorSeries([1,-1/(xb-xc)], xc)
        for l = k:(knext-1)
            t = shift!(t, offset(𝒬̅[l]))
            Q̄i .+= integrate(𝒬̅[l] * t, (grid[l], grid[l+1]))
            for (R̄i, ℛ̅) in zip(R̄is, ℛ̅s)
                R̄i .+= integrate(ℛ̅[l] * t, (grid[l], grid[l+1]))
            end
            if !isnothing(∂ℛ̅s)
                for (R̄i, ∂ℛ̅) in zip(R̄is, ∂ℛ̅s)
                    R̄i .+= integrate(∂ℛ̅[l]/(xc-xb), (grid[l], grid[l+1]))
                end
            end
        end
        setindex!(Q̄, Q̄i, i)
        setindex!.(R̄s, R̄is, i)
    end

    k = gradindices[end]
    kprev = gradindices[end-1]
    xa = grid[kprev]
    xc = grid[k]

    Q̄i = Q̄[end]
    t = TaylorSeries([0,1/(xc-xa)], xa)
    for l = kprev:(k-1)
        t = shift!(t, offset(𝒬̅[l]))
        Q̄i .+= integrate(𝒬̅[l] * t, (grid[l], grid[l+1]))
    end
    Q̄[end] = Q̄i

    if gradindices == 1:length(grid)
        ∇Q = PiecewiseLinear(grid, Q̄)
        ∇Rs = PiecewiseLinear.((grid,), R̄s)
    else
        grid2 = grid[gradindices]
        ∇Q = PiecewiseLinear(grid2, Q̄)
        ∇Rs = PiecewiseLinear.((grid2,), R̄s)
        ∇Q = PiecewiseLinear(grid, ∇Q.(grid))
        ∇Rs = map(∇Rs) do ∇R
            PiecewiseLinear(grid, ∇R.(grid))
        end
    end

    return ∇Q, ∇Rs
end

# Actual metric acting on a given tangent vector
function metric(Φ::FiniteCMPSTangent, Ψρs::FiniteCMPSData;
                δ = 0, Kmax = 50, tol = defaulttol(base(Φ)),
                left_boundary = :free, right_boundary = :free,
                gradindices = 1:length(nodes(Φ.dQ)))

    Ψ, ρL, ρR = Ψρs

    base(Φ) == Ψ || throw(DomainMismatch())

    Q, Rs, vL, vR = Ψ
    (a, b) = domain(Ψ)
    T = scalartype(Ψ)

    fL = ρL * Φ.dQ
    fR = Φ.dQ * ρR
    temp = zero(ρL)
    for (R, dR) in zip(Rs, Φ.dRs)
        RdρL = mul!(temp, R', ρL)
        fL = mul!(fL, RdρL, dR, one(T), one(T))
        ρRRd = mul!(temp, ρR, R')
        fR = mul!(fR, dR, ρRRd, one(T), one(T))
    end
    FL, = lefttransfer(zero(fL(a)), fL, Ψ; Kmax = Kmax, tol = tol)
    FR, = righttransfer(zero(fR(b)), fR, Ψ; Kmax = Kmax, tol = tol)

    𝒬̅ = zero(FL)
    𝒬̅ = mul!(𝒬̅, FL, ρR, one(T), one(T))
    𝒬̅ = mul!(𝒬̅, ρL, FR, one(T), one(T))
    ℛ̅s = map(_->zero(FL), Rs)
    for (R, ℛ̅, dR) in zip(Rs, ℛ̅s, Φ.dRs)
        FLR = mul!(temp, FL, R)
        ℛ̅ = mul!(ℛ̅, FLR, ρR, one(T), one(T))
        RFR = mul!(temp, R, FR)
        ℛ̅ = mul!(ℛ̅, ρL, RFR, one(T), one(T))
        dRρR = mul!(temp, dR, ρR)
        ℛ̅ = mul!(ℛ̅, ρL, dRρR, one(T), one(T))
    end
    return _project(𝒬̅, ℛ̅s; gradindices = gradindices)
end
