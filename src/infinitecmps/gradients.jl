function gradient(H::LocalHamiltonian, Ψρs::InfiniteCMPSData, HL = nothing, HR = nothing;
                    kwargs...)
    Ψ, ρL, ρR = Ψρs
    if isnothing(HL)
        HL, = leftenv(H, Ψρs; kwargs...)
    end
    if isnothing(HR)
        HR, = rightenv(H, Ψρs; kwargs...)
    end

    Q = Ψ.Q
    Rs = Ψ.Rs

    # gradQ = ∑(coeff * localgradientQ)  +  HL*ρR + ρL*HR
    gradQ = zero(Q)
    for (coeff, op) in zip(coefficients(H.h), operators(H.h))
        if coeff isa Number
            axpy!(coeff, localgradientQ(op, Ψ, ρL, ρR), gradQ)
        else
            mul!(gradQ, coeff, localgradientQ(op, Ψ, ρL, ρR), 1, 1)
        end
    end
    mul!(gradQ, HL, ρR, 1, 1)
    mul!(gradQ, ρL, HR, 1, 1)

    # gradR = ∑(coeff * localgradientR)  +  HL*R*ρR + ρL*R*HR
    gradRs = zero.(Rs)
    for (coeff, op) in zip(coefficients(H.h), operators(H.h))
        if coeff isa Number
            axpy!.(coeff, localgradientRs(op, Ψ, ρL, ρR), gradRs)
            if op isa ContainsDifferentiatedCreation && !(Q isa Constant)
                grad∂Rs = localgradient∂Rs(op, Ψ, ρL, ρR)
                axpy!.(-coeff, ∂.(grad∂Rs), gradRs)
            end
        else
            mul!.(gradRs, (coeff,), localgradientRs(op, Ψ, ρL, ρR), 1, 1)
            if op isa ContainsDifferentiatedCreation && !(Q isa Constant)
                grad∂Rs = localgradient∂Rs(op, Ψ, ρL, ρR)
                mul!.(gradRs, (-coeff,), ∂.(grad∂Rs), 1, 1)
            end
        end
    end
    mul!.(gradRs, (HL,), Rs .* (ρR,), 1, 1)
    mul!.(gradRs, (ρL,), Rs .* (HR,), 1, 1)

    return gradQ, gradRs
end
