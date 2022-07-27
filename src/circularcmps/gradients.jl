function gradient(H::LocalHamiltonian, Ψ::CircularCMPS, E = nothing, EH = nothing; kwargs...)
    Q, Rs = Ψ.Q, Ψ.Rs
    if isnothing(EH)
        EH, E = environment(H, Ψ; kwargs...)
    elseif isnothing(E)
        E = environment(Ψ; kwargs...)
    end
    D = size(Q(0), 1)

    Z = tr(E(0))
    e = tr(EH(0))/Z
    EH′ = axpy!(-e, E, EH)
    gradQ = map_linear(x->permutedims(partialtrace1(x, D, D)), EH′)
    gradRs = map(Rs) do R
        RU = map_linear(x->(x ⊗ one(x)), R)
        return map_linear(x->permutedims(partialtrace1(x, D, D)), EH′*RU)
    end
    ops = H.h
    for (c, op) in zip(coefficients(ops), operators(ops))
        E1 = mul!(zero(E), map_bilinear(⊗, _ketfactor(op, Q, Rs), one(Q)), E)
        E2 = map_linear(x->permutedims(partialtrace1(x, D, D)), E1)
        gradQ = axpy!(c, _localgradientQ(op, Q, Rs)(E2), gradQ)
        gradRs = axpy!.(c, _localgradientRs(op, Q, Rs)(E2), gradRs)
    end
    gradQ = rmul!(gradQ, 1/Z)
    gradRs = rmul!.(gradRs, 1/Z)
    return gradQ, gradRs
end
