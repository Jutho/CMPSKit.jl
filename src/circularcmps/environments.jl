function environment(Ψ₁::CircularCMPS, Ψ₂::CircularCMPS=Ψ₁; kwargs...)
    period(Ψ₁) == period(Ψ₂) || throw(DomainMismatch())

    T = _full(RightTransfer(Ψ₁, Ψ₂); kwargs...)
    if T isa Constant
        return Constant(exp(period(Ψ₁) * T[]))
    else
        # TODO
        # idea: use Floquet
        # Pexp(∫₀ˣ T(y) dy) = exp(B1 * x) * periodicfunction1(x)
        # Pexp(∫ₓᴸ T(y) dy) = periodicfunction2(x) * exp(B2 * (L - x))
        # then Pexp(∫ₓˣ⁺ᴸ T(y) dy) = pf2(x) * exp(B2*(L-x)) * exp(B1*x) * pf1(x)
        # This should be a periodic function: does this imply B1 == B2 ?
        # What is then tr(Pexp(∫ₓˣ⁺ᴸ T(y) dy)), which should be constant
        # Note that pf1(0) == pf1(L) == pf2(L) == pf2(0) == 1
        # We should thus have tr(Pexp(∫₀ᴸ T(y) dy)) = tr(exp(B1*L)) = tr(exp(B2*L))
        # We can always make the choice B1 = B2 = 1/L * log[ Pexp(∫₀ᴸ T(y) dy) ]
    end
end

function environment(H::LocalHamiltonian, Ψ₁::CircularCMPS, Ψ₂::CircularCMPS=Ψ₁; kwargs...)
    period(Ψ₁) == period(Ψ₂) || throw(DomainMismatch())
    L = period(Ψ₁)

    Q₁, R₁s = Ψ₁.Q, Ψ₁.Rs
    Q₂, R₂s = Ψ₂.Q, Ψ₂.Rs
    T = _full(RightTransfer(Ψ₁, Ψ₂); kwargs...)

    if T isa Constant
        HH = zero(T)
        ops = H.h
        for (c, op) in zip(coefficients(ops), operators(ops))
            addkronecker!(HH[], _ketfactor(op, Q₁, R₁s)[], _brafactor(op, Q₂, R₂s)[], c)
        end
        HH = rmul!(HH, L)
        T = rmul!(T, L)
        E, EH = Constant.(exp_blocktriangular(T[], HH[]))
        return EH, E
    else
        # TODO
    end
end
