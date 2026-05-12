function gradient(H::LocalHamiltonian, Ψρs::FiniteCMPSData, HL=nothing, HR=nothing;
                  gradindices=1:length(nodes(Ψρs[1].Q)),
                  left_boundary=:fixed,
                  right_boundary=:fixed,
                  kwargs...)
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

    Z = dot(vR, ρL(b) * vR)
    E = dot(vR, HL(b) * vR) / Z

    gradQ, gradRs, grad∂Rs = _gradient(H, Ψρs, HL, HR, Z, E)
    ∇Q, ∇Rs = _project(gradQ, gradRs, grad∂Rs; gradindices=gradindices)

    ∇vL = left_boundary == :fixed ? zero(vL) : (HR(a) * vL - E * ρR(a) * vL) / Z
    ∇vR = right_boundary == :fixed ? zero(vR) : (HL(b) * vR - E * ρL(b) * vR) / Z

    return FiniteCMPSTangent(Ψ, ∇Q, ∇Rs, ∇vL, ∇vR, gradindices)
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
                axpy!(coeff / Z, localgradientQ(op, Ψ, ρL, ρR), gradQ)
            else
                mul!(gradQ, coeff, localgradientQ(op, Ψ, ρL, ρR), 1 / Z, 1)
            end
        end
    end
    mul!(gradQ, HL, ρR, 1 / Z, 1)
    mul!(gradQ, ρL, HR, 1 / Z, 1)
    mul!(gradQ, ρL, ρR, -E / Z, 1)

    # gradR = ∑(coeff * localgradientR)  +  HL*R*ρR + ρL*R*HR
    # however, we treat R and ∂R as independent variables at first in computing the gradient
    gradRs = map(R -> zero(ρL), Rs)
    for (coeff, op) in zip(coefficients(H.h), operators(H.h))
        if coeff isa Number
            axpy!.(coeff / Z, localgradientRs(op, Ψ, ρL, ρR), gradRs)
        else
            mul!.(gradRs, (coeff,), localgradientRs(op, Ψ, ρL, ρR), 1 / Z, 1)
        end
    end
    RρRs = Rs .* (ρR,)
    mul!.(gradRs, (HL,), RρRs, 1 / Z, 1)
    mul!.(gradRs, (ρL,), Rs .* (HR,), 1 / Z, 1)
    mul!.(gradRs, (ρL,), RρRs, -E / Z, 1)

    grad∂Rs = map(R -> zero(ρL), Rs)
    for (coeff, op) in zip(coefficients(H.h), operators(H.h))
        if op isa ContainsDifferentiatedCreation
            if coeff isa Number
                axpy!.(coeff / Z, localgradient∂Rs(op, Ψ, ρL, ρR), grad∂Rs)
            else
                mul!.(grad∂Rs, (coeff,), localgradient∂Rs(op, Ψ, ρL, ρR), 1 / Z, 1)
            end
        end
    end

    return gradQ, gradRs, grad∂Rs
end

function Qmetric(grid)
    N = length(grid)
    dv = zeros(N)
    ev = zeros(N - 1)
    @inbounds for i in 1:N
        dv[i] = (grid[min(N, i + 1)] - grid[max(i - 1, 1)]) / 3
        ev[i] = (grid[i + 1] - grid[i]) / 6
    end
    return SymTridiagonal(dv, ev)
end

function Rmetric(grid)
    g = Qmetric(grid)
    g.dv[1] = g.dv[end] = 1
    g.ev[1] = g.ev[end] = 0
    return g
end
