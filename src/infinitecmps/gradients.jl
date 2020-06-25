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

    gradQ = zero(Q)
    for (coeff, op) in zip(coefficients(H.h), operators(H.h))
        gradQ = gradQ + coeff * localgradientQ(op, Q, Rs, ρL, ρR)
    end
    gradQ += HL*ρR + ρL*HR

    gradRs = zero.(Rs)
    for (coeff, op) in zip(coefficients(H.h), operators(H.h))
        gradRs = gradRs .+ Ref(coeff) .* localgradientRs(op, Q, Rs, ρL, ρR)
    end
    gradRs = gradRs .+ Ref(HL) .* Rs .* Ref(ρR) .+ Ref(ρL) .* Rs .* Ref(HR)

    return gradQ, gradRs
end
