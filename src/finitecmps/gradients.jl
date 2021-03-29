function gradient(H::LocalHamiltonian, Ψρs::FiniteCMPSData, HL = nothing, HR = nothing;
                    gradindices = 1:length(nodes(Ψρs[1].Q)), kwargs...)

    @assert first(gradindices) == 1
    @assert last(gradindices) == length(nodes(Ψρs[1].Q))
    @assert issorted(gradindices)

    Ψ, ρL, ρR = Ψρs
    if isnothing(HL)
        HL, = leftenv(H, Ψρs; kwargs...)
    end
    if isnothing(HR)
        HR, = rightenv(H, Ψρs; kwargs...)
    end
    a, b = domain(Ψ)
    Q, Rs, vL, vR = Ψ

    Z = dot(vR, ρL(b)*vR)
    E = dot(vR, HL(b)*vR)/Z

    gradQ, gradRs, grad∂Rs = _gradient(H, Ψρs, HL, HR, Z, E)

    # Compute gradients with respect to PiecewiseLinear parameters
    Q̄ = [zero(Q(a)) for _ = 1:length(gradindices)]
    R̄s = map(R->[zero(Q(a)) for _ = 1:length(gradindices)], Rs)
    v̄L = (HR(a)*vL - E*ρR(a)*vL)/Z
    v̄R = (HL(b)*vR - E*ρL(b)*vR)/Z

    grid = nodes(Q)

    k = gradindices[1] # == 1
    knext = gradindices[2]
    xc = grid[k]
    xb = grid[knext]
    t = TaylorSeries([1,-1/(xb-xc)], xc)
    Q̄i = Q̄[1]
    for l = k:(knext-1)
        t = shift!(t, offset(gradQ[l]))
        Q̄i .+= integrate(gradQ[l] * t, (grid[l], grid[l+1]))
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
            t = shift!(t, offset(gradQ[l]))
            Q̄i .+= integrate(gradQ[l] * t, (grid[l], grid[l+1]))
            for (R̄i, gradR, grad∂R) in zip(R̄is, gradRs, grad∂Rs)
                R̄i .+= integrate(gradR[l] * t, (grid[l], grid[l+1]))
                R̄i .+= integrate(grad∂R[l]/(xc-xa), (grid[l], grid[l+1]))
            end
        end
        t = TaylorSeries([1,-1/(xb-xc)], xc)
        for l = k:(knext-1)
            t = shift!(t, offset(gradQ[l]))
            Q̄i .+= integrate(gradQ[l] * t, (grid[l], grid[l+1]))
            for (R̄i, gradR, grad∂R) in zip(R̄is, gradRs, grad∂Rs)
                R̄i .+= integrate(gradR[l] * t, (grid[l], grid[l+1]))
                R̄i .+= integrate(grad∂R[l]/(xc-xb), (grid[l], grid[l+1]))
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
        t = shift!(t, offset(gradQ[l]))
        Q̄i .+= integrate(gradQ[l] * t, (grid[l], grid[l+1]))
    end
    Q̄[end] = Q̄i

    return Q̄, R̄s, v̄L, v̄R
end

function gradient2(H::LocalHamiltonian, Ψρs::FiniteCMPSData, HL = nothing, HR = nothing;
                    gradindices = 1:length(nodes(Ψρs[1].Q)), kwargs...)

    Q̄, R̄s, v̄L, v̄R = gradient(H, Ψρs, HL, HR; gradindices = gradindices, kwargs...)

    Ψ, ρL, ρR = Ψρs
    Q, Rs, vL, vR = Ψ

    grid = nodes(Q)
    gradgrid = grid[gradindices]
    gQ = Qmetric(gradgrid)
    gR = Rmetric(gradgrid)
    igQ = inv(gQ)
    igR = inv(gR)


    Q̄ = PiecewiseLinear(gradgrid, igQ*Q̄)
    R̄s = map(R̄->PiecewiseLinear(gradgrid, igR*R̄), R̄s)

    if gradindices != 1:length(grid)
        # resample
        Q̄ = PiecewiseLinear(grid, Q̄.(grid))
        R̄s = map(R̄->PiecewiseLinear(grid, R̄.(grid)), R̄s)
    end

    return Q̄, R̄s, v̄L, v̄R
end

# Compute the gradient as a continuous function, represented as Piecewise{TaylorSeries}
function _gradient(H::LocalHamiltonian, Ψρs::FiniteCMPSData, HL, HR, Z, E)
    Ψ, ρL, ρR = Ψρs
    Q, Rs, vL, vR = Ψ

    # gradQ = ∑(coeff * localgradientQ)  +  HL*ρR + ρL*HR
    gradQ = zero(ρL)
    for (coeff, op) in zip(coefficients(H.h), operators(H.h))
        if op isa ContainsDifferentiatedCreation
            if coeff isa Number
                axpy!(coeff/Z, localgradientQ(op, Q, Rs, ρL, ρR), gradQ)
            else
                mul!(gradQ, coeff, localgradientQ(op, Q, Rs, ρL, ρR), 1/Z, 1)
            end
        end
    end
    mul!(gradQ, HL, ρR, 1/Z, 1)
    mul!(gradQ, ρL, HR, 1/Z, 1)
    mul!(gradQ, ρL, ρR, -E/Z, 1)

    # gradR = ∑(coeff * localgradientR)  +  HL*R*ρR + ρL*R*HR
    # however, we treat R and ∂R as independent variables at first in computing the gradient
    gradRs = map(R->zero(ρL), Rs)
    for (coeff, op) in zip(coefficients(H.h), operators(H.h))
        if coeff isa Number
            axpy!.(coeff/Z, localgradientRs(op, Q, Rs, ρL, ρR), gradRs)
        else
            mul!.(gradRs, (coeff,), localgradientRs(op, Q, Rs, ρL, ρR), 1/Z, 1)
        end
    end
    RρRs = Rs .* (ρR,)
    mul!.(gradRs, (HL,), RρRs, 1/Z, 1)
    mul!.(gradRs, (ρL,), Rs .* (HR,), 1/Z, 1)
    mul!.(gradRs, (ρL,), RρRs, -E/Z, 1)

    grad∂Rs = map(R->zero(ρL), Rs)
    for (coeff, op) in zip(coefficients(H.h), operators(H.h))
        if op isa ContainsDifferentiatedCreation
            if coeff isa Number
                axpy!.(coeff/Z, localgradient∂Rs(op, Q, Rs, ρL, ρR), grad∂Rs)
            else
                mul!.(grad∂Rs, (coeff,), localgradient∂Rs(op, Q, Rs, ρL, ρR), 1/Z, 1)
            end
        end
    end

    return gradQ, gradRs, grad∂Rs
end

function Qmetric(grid)
    N = length(grid)
    dv = zeros(N)
    ev = zeros(N-1)
    @inbounds for i = 1:N
        dv[i] = (grid[min(N, i+1)] - grid[max(i-1, 1)])/3
        ev[i] = (grid[i+1] - grid[i])/6
    end
    return SymTridiagonal(dv, ev)
end
function Rmetric(grid)
    g = Qmetric(grid)
    g.dv[1] = g.dv[end] = 1
    g.ev[1] = g.ev[end] = 0
    return g
end
