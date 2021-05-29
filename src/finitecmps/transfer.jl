"""
    lefttransfer(HL₀::AbstractMatrix, hL::Union{Nothing, AbstractPiecewise},
                    Ψ₁::FiniteCMPS, Ψ₂::FiniteCMPS = Ψ₁;
                    Kmax = 50, tol = defaulttol(Ψ₁))

Solve the non-homogeneous (or homogeneous if hL === nothing) Lindblad equation

```math
∂ₓ(HL) = Q₂' * HL + HL * Q₁ + R₂' * HL * R₁ + hL
```

with `Q₁` and `R₁` the matrices of the finite cMPS `Ψ₁` (similar for `Q₂` and `R₂`, where
`Ψ₂ = Ψ₁` by default), over the whole domain `(a,b) = domain(Ψ₁)`, starting from the initial
condition `HL(a) = HL₀`, all the way up to `b`. The states `Ψ₁` and `Ψ₂` are assumed to have
equal domain and grid; some checks are in place.

The result is given by

```math
⟨HL(x)| = ⟨HL₀|exp(∫ₐˣ T(z) dz) + ∫ₐˣ [ ⟨hL(u)|exp(∫ᵤˣ T(z) dz) ] du
```

and returned as `HL, converged` with `HL::Piecewise` and where `converged::Bool` is `true`
or `false` depending on convergence up to the requested tolerance.
"""
function lefttransfer(HL₀::AbstractMatrix, hL::Union{Nothing, AbstractPiecewise},
                        Ψ₁::FiniteCMPS, Ψ₂::FiniteCMPS = Ψ₁;
                        Kmax = 50, tol = defaulttol(Ψ₁))

    Q₁ = Ψ₁.Q
    R₁s = Ψ₁.Rs
    Q₂ = Ψ₂.Q
    R₂s = Ψ₂.Rs
    grid = nodes(Q₁)
    N = length(grid) - 1
    a = first(grid)
    b = last(grid)
    if Ψ₁ !== Ψ₂
        grid == nodes(Q₂) || throw(DomainMismatch())
    end
    if !isnothing(hL)
        grid == nodes(hL) || throw(DomainMismatch())
    end
    HLᵢ = TaylorSeries([HL₀], a)
    HLs = Vector{typeof(HLᵢ)}(undef, N)
    HLs[1] = HLᵢ
    infoL = true
    for i = 1:N
        HLᵢ = HLs[i]
        xᵢ = grid[i]
        xⱼ = grid[i+1]
        Δxᵢ = grid[i+1] - grid[i]
        Q₁ᵢ = shift!(Q₁[i], xᵢ)
        R₁ᵢs = shift!.(getindex.(R₁s, i), xᵢ)
        Q₂ᵢ = shift!(Q₂[i], xᵢ)
        R₂ᵢs = shift!.(getindex.(R₂s, i), xᵢ)
        hLᵢ = isnothing(hL) ? nothing : shift(hL[i], xᵢ)

        # build Taylor coefficients (i.e. solve triangular problem)
        HLᵢ, converged = _lefttransfer!(HLᵢ, hLᵢ, Q₁ᵢ, R₁ᵢs, Δxᵢ, Q₂ᵢ, R₂ᵢs;
                                        Kmax = Kmax, tol = tol)
        infoL &= converged
        shift!(HLᵢ, (xᵢ+xⱼ)/2)

        # initialize next ρ element
        if i < N
            HLs[i+1] = TaylorSeries([HLᵢ(xⱼ)], xⱼ)
        end
    end
    HL = Piecewise(grid, HLs)
    return HL, infoL
end

"""
    righttransfer(HR₀::AbstractMatrix, hR::Union{Nothing, AbstractPiecewise},
                    Ψ₁::FiniteCMPS, Ψ₂::FiniteCMPS = Ψ₁;
                    Kmax = 50, tol = defaulttol(Ψ₁))

Solve the non-homogeneous (or homogeneous if hR === nothing) Lindblad equation

```math
-∂ₓ(HR) = Q₁ * HR + HR * Q₂' + R₁ * HR * R₂ + hR
```

with `Q₁` and `R₁` the matrices of the finite cMPS `Ψ₁` (similar for `Q₂` and `R₂`, where
`Ψ₂ = Ψ₁` by default), over the whole domain `(a,b) = domain(Ψ₁)`, starting from the initial
condition `HR(b) = HR₀`, all the way down to `a`. The states `Ψ₁` and `Ψ₂` are assumed to
have equal domain and grid; some checks are in place.

The result is given by

```math
|HR(x)⟩ = exp(∫ₓᵇ T(z) dz)|HR₀⟩ + ∫ₓᵇ exp(∫ₓᵘ T(z) dz)|hR(u)⟩ du
```

and returned as `HL, converged` with `HL::Piecewise` and where `converged::Bool` is `true`
or `false` depending on convergence up to the requested tolerance.
"""
function righttransfer(HR₀::AbstractMatrix, hR::Union{Nothing, AbstractPiecewise},
                        Ψ₁::FiniteCMPS, Ψ₂::FiniteCMPS = Ψ₁;
                        Kmax = 50, tol = defaulttol(Ψ₁))

    Q₁ = Ψ₁.Q
    R₁s = Ψ₁.Rs
    Q₂ = Ψ₂.Q
    R₂s = Ψ₂.Rs
    grid = nodes(Q₁)
    N = length(grid) - 1
    a = first(grid)
    b = last(grid)
    if Ψ₁ !== Ψ₂
        grid == nodes(Q₂) || throw(DomainMismatch())
    end
    if !isnothing(hR)
        grid == nodes(hR) || throw(DomainMismatch())
    end
    HRᵢ = TaylorSeries([HR₀], b)
    HRs = Vector{typeof(HRᵢ)}(undef, N)
    HRs[N] = HRᵢ
    infoR = true
    for i = N:-1:1
        HRᵢ = HRs[i]
        xᵢ = grid[i]
        xⱼ = grid[i+1]
        Δxᵢ = grid[i+1] - grid[i]
        Q₁ᵢ = shift!(Q₁[i], xⱼ)
        R₁ᵢs = shift!.(getindex.(R₁s, i), xⱼ)
        Q₂ᵢ = shift!(Q₂[i], xⱼ)
        R₂ᵢs = shift!.(getindex.(R₂s, i), xⱼ)
        hRᵢ = isnothing(hR) ? nothing : shift(hR[i], xⱼ)

        # build Taylor coefficients (i.e. solve triangular problem)
        HRᵢ, converged = _righttransfer!(HRᵢ, hRᵢ, Q₁ᵢ, R₁ᵢs, Δxᵢ, Q₂ᵢ, R₂ᵢs;
                                            Kmax = Kmax, tol = tol)
        infoR &= converged
        shift!(HRᵢ, (xᵢ+xⱼ)/2)

        # initialize next ρ element
        if i > 1
            HRs[i-1] = TaylorSeries([HRᵢ(xᵢ)], xᵢ)
        end
    end
    HR = Piecewise(grid, HRs)
    return HR, infoR
end

"""
    _lefttransfer!(ρ::TaylorSeries, h::TaylorSeries,
                    Q₁::TaylorSeries, R₁s::Tuple{Vararg{TaylorSeries}}, Δx,
                    Q₂::TaylorSeries = Q₁, R₂s::Tuple{Vararg{TaylorSeries}} = R₁s;
                    Kmax = 50, tol = defaulttol(ρ))

Solve the non-homogeneous (or homogeneous if h === nothing) Lindblad equation

```math
∂ₓρ = Q₂'*ρ + ρ*Q₁ + R₂'*ρ*R₁ + h
```

in place (i.e. the Taylor coefficients of `ρ` will be inserted), starting from the initial
conditions `ρ[0]` at position `x = offset(ρ)`. The solution will have `degree(ρ) <= Kmax`
and is found up to a tolerance `tol` in the interval `Δx`, i.e. the last few Taylor
coefficients of `ρ` and `h` should satisfy `ρ[k]*Δx^k < tol`. The result is returned as `ρ,
converged`, where `converged::Bool` is `true` or `false` depending on convergence up to the
given tolerance.

Note that the coefficients of `h` are destroyed in the process, if you want to preserve
those you should take a copy beforehand.
"""
function _lefttransfer!(ρ::TaylorSeries, h::Union{Nothing, TaylorSeries},
                            Q₁::TaylorSeries, R₁s::Tuple{Vararg{TaylorSeries}}, Δx,
                            Q₂::TaylorSeries = Q₁, R₂s::Tuple{Vararg{TaylorSeries}} = R₁s;
                            Kmax = 50, tol = defaulttol(ρ))

    Kmin = max(degree(Q₁), degree(Q₂), maximum(degree, R₁s), maximum(degree, R₂s)) + 1
    temp = zero(ρ[0])
    T = eltype(ρ[0])
    for k = 1:Kmax
        ρᵏ = isnothing(h) ? zero(ρ[0]) : h[k-1]
        # solve triangular system:
        for l = 0:min(k-1, degree(Q₁))
            mul!(ρᵏ, ρ[k-1-l], Q₁[l], one(T), one(T))
        end
        for l = 0:min(k-1, degree(Q₂))
            mul!(ρᵏ, Q₂[l]', ρ[k-1-l], one(T), one(T))
        end
        for (R₁, R₂) in zip(R₁s, R₂s)
            for l = 0:min(k-1, degree(R₂))
                for m = 0:min(k-1-l, degree(R₁))
                    mul!(temp, ρ[k-1-l-m], R₁[m], one(T), zero(T))
                    mul!(ρᵏ, R₂[l]', temp, one(T), one(T))
                end
            end
        end
        rmul!(ρᵏ, 1/k)
        ρ[k] = ρᵏ
        # check for convergence
        if k > Kmin
            converged = true
            for l = k-Kmin+1:k
                if norm(ρ[l]) * (Δx^l) > tol
                    converged = false
                end
            end
            if converged
                return ρ, true
            end
        end
    end
    return ρ, false
end


"""
    _righttransfer!(ρ::TaylorSeries, h::TaylorSeries,
                    Q₁::TaylorSeries, R₁s::Tuple{Vararg{TaylorSeries}}, Δx,
                    Q₂::TaylorSeries = Q₁, R₂s::Tuple{Vararg{TaylorSeries}} = R₁s;
                    Kmax = 50, tol = defaulttol(ρ))

Solve the non-homogeneous (or homogeneous if h === nothing) Lindblad equation

```math
∂ₓρ = -(Q₂'*ρ + ρ*Q₁ + R₂'*ρ*R₁) - h
```

in place (i.e. the Taylor coefficients of `ρ` will be inserted), starting from the initial
conditions `ρ[0]` at position `x = offset(ρ)`. The solution will have `degree(ρ) <= Kmax`
and is found up to a tolerance `tol` in the interval `Δx`, i.e. the last few Taylor
coefficients of `ρ` and `h` should satisfy `ρ[k]*Δx^k < tol`. The result is returned as `ρ,
converged`, where `converged::Bool` is `true` or `false` depending on convergence up to the
given tolerance.

Note that the coefficients of `h` are destroyed in the process, if you want to preserve
those you should take a copy beforehand. Also note that all TaylorSeries are assumed to have
the same offset.
"""
function _righttransfer!(ρ::TaylorSeries, h::Union{Nothing, TaylorSeries},
                            Q₁::TaylorSeries, R₁s::Tuple{Vararg{TaylorSeries}}, Δx,
                            Q₂::TaylorSeries = Q₁, R₂s::Tuple{Vararg{TaylorSeries}} = R₁s;
                            Kmax = 50, tol = defaulttol(ρ))

    Kmin = max(degree(Q₁), degree(Q₂), maximum(degree, R₁s), maximum(degree, R₂s)) + 1
    temp = zero(ρ[0])
    T = eltype(ρ[0])
    for k = 1:Kmax
        ρᵏ = isnothing(h) ? zero(ρ[0]) : rmul!(h[k-1], -1)
        # solve triangular system:
        for l = 0:min(k-1, degree(Q₁))
            mul!(ρᵏ, Q₁[l], ρ[k-1-l], -one(T), one(T))
        end
        for l = 0:min(k-1, degree(Q₂))
            mul!(ρᵏ, ρ[k-1-l], Q₂[l]', -one(T), one(T))
        end
        for (R₁, R₂) in zip(R₁s, R₂s)
            for l = 0:min(k-1, degree(R₁))
                for m = 0:min(k-1-l, degree(R₂))
                    mul!(temp, ρ[k-1-l-m], R₂[m]', one(T), zero(T))
                    mul!(ρᵏ, R₁[l], temp, -one(T), one(T))
                end
            end
        end
        rmul!(ρᵏ, 1/k)
        ρ[k] = ρᵏ
        # check for convergence
        if k > Kmin
            converged = true
            for l = k-Kmin+1:k
                if norm(ρ[l]) * (Δx^l) > tol
                    converged = false
                end
            end
            if converged
                return ρ, true
            end
        end
    end
    # not converged:
    return ρ, false
end
