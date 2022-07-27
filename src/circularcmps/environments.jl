function environment(Ψ₁::CircularCMPS, Ψ₂::CircularCMPS = Ψ₁; kwargs...)
    period(Ψ₁) == period(Ψ₂) || throw(DomainMismatch())

    T = _full(RightTransfer(Ψ₁, Ψ₂); kwargs...)
    if T isa Constant
        return Constant(exp(period(Ψ₁)*T[]))
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

function environment(H::LocalHamiltonian, Ψ₁::CircularCMPS, Ψ₂::CircularCMPS = Ψ₁; kwargs...)
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

"""
`tangent_space_metric(Ψ::UniformCircularCMPS)`

Returns the Fubini-Study metric of the UniformCircularCMPS manifold at the point `Ψ`,
in the form of a function `metric = tangent_space_metric(Ψ)` that maps the parameters
`V` and `Ws = (W₁, W₂, ...)` of a tangent vector ``|Φ(V,W₁,W₂,...)⟩`` to equivalent values
`GV, GWs = metric(V, Ws)`, such that, for any `Ṽ` and `W̃s = (W̃₁, W̃₂, ...)` parameterising
a second tangent vector, we have an equality between
```julia
dot(Ṽ, GV) + sum(dot.(W̃s, GWs))
```
and
```math
⟨ Φ(Ṽ,W̃₁,W̃₂,…) | [ 1 - |Ψ⟩⟨Ψ|/⟨Ψ|Ψ⟩ ] | Φ(V,W₁,W₂,…) ⟩ / ⟨Ψ|Ψ⟩.
```
"""
function tangent_space_metric(Ψ::UniformCircularCMPS; kwargs...)
    Q, Rs = Ψ.Q, Ψ.Rs
    T = _full(RightTransfer(Ψ); kwargs...)
    L = period(Ψ)
    E_, Eenv = exp_blocktriangular_lazy!(L*T[])
    Z = real(tr(E_)) # norm(Ψ)^2 = ⟨Ψ|Ψ⟩
    E = Constant(E_)
    function metric(V, Ws)
        D = size(V[], 1)
        GWs = map(Ws) do W
            map_linear(x->permutedims(partialtrace1(x, D, D)), map_bilinear(⊗, W, one(W)) * E)
        end

        VWs = map_bilinear(⊗, V, one(V))
        for (R, W) in zip(Rs, Ws)
            axpy!(1, map_bilinear(⊗, W, conj(R)), VWs)
        end
        EVWs = rmul!(map_linear(Eenv, VWs), L)
        Ψoverlap = tr(EVWs[]) # ⟨Ψ|Φ(V,W)⟩
        EVWs = axpy!(-Ψoverlap/Z, E, EVWs)

        GV = map_linear(x->permutedims(partialtrace1(x, D, D)), EVWs)
        for (R, GW) in zip(Rs, GWs)
            R1 = map_bilinear(⊗, R, one(R))
            axpy!(1, map_linear(x->permutedims(partialtrace1(x, D, D)), EVWs*R1), GW)
        end

        return rmul!(GV, 1/Z), rmul!.(GWs, 1/Z)
        return GV, GWs
    end
    return metric
end
