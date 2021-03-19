"""
    transferleft!(ρ::TaylorSeries, h::TaylorSeries,
                    Q::TaylorSeries, Rs::Tuple{TaylorSeries...}, Δx;
                    Kmax = 50, tol = eps())

    Solve the non-homogeneous (or homogeneous if h == zero(h)) Lindblad equation

    dρ/dx = Q'*ρ + ρ*Q + R'*ρ*R + h

    in place (i.e. the Taylor coefficients of `ρ` will be inserted), starting from the
    initial conditions `ρ[0]` at position `x = offset(ρ)`. The solution will have
    `degree(ρ) <= Kmax` and is found up to a tolerance `tol` in the interval `Δx`, i.e. the
    last few Taylor coefficients of `ρ` and `h` should satisfy `ρ[k]*Δx^k < tol`. The
    result is returned as `ρ, converged`, where `converged::Bool` is `true` or `false`
    depending on convergence up to the given tolerance.

    Note that the coefficients of `h` are destroyed in the process, if you want to preserve those you should take a copy beforehand.
"""
function _transferleft!(ρ::TaylorSeries, h::TaylorSeries,
                            Q::TaylorSeries, Rs::Tuple{Vararg{TaylorSeries}},
                            Δx, Kmax, tol)

    Kmin = max(degree(Q), maximum(degree, Rs)) + 1
    temp = zero(Q[0])
    T = eltype(h[0])
    for k = 1:Kmax
        ρᵏ = h[k-1]
        # solve triangular system:
        for l = 0:min(k-1, degree(Q))
            mul!(ρᵏ, Q[l]', ρ[k-1-l], one(T), one(T))
            mul!(ρᵏ, ρ[k-1-l], Q[l], one(T), one(T))
        end
        for R in Rs
            for l = 0:min(k-1, degree(R))
                for m = 0:min(k-1-l, degree(R))
                    mul!(temp, ρ[k-1-l-m], R[m], one(T), zero(T))
                    mul!(ρᵏ, R[l]', temp, one(T), one(T))
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
    transferright!(ρ::TaylorSeries, h::TaylorSeries,
                    Q::TaylorSeries, Rs::Tuple{TaylorSeries...}, Δx;
                    Kmax = 50, tol = eps())

    Solve the non-homogeneous (or homogeneous if h == zero(h)) Lindblad equation

    dρ/dx = -(Q'*ρ + ρ*Q + R'*ρ*R) - h

    in place (i.e. the Taylor coefficients of `ρ` will be inserted), starting from the
    initial conditions `ρ[0]` at position `x = offset(ρ)`. The solution will have
    `degree(ρ) <= Kmax` and is found up to a tolerance `tol` in the interval `Δx`, i.e. the
    last few Taylor coefficients of `ρ` and `h` should satisfy `ρ[k]*Δx^k < tol`. The
    result is returned as `ρ, converged`, where `converged::Bool` is `true` or `false`
    depending on convergence up to the given tolerance.

    Note that the coefficients of `h` are destroyed in the process, if you want to preserve
    those you should take a copy beforehand. Also note that all TaylorSeries are assumed to
    have the same offset.
"""
function _transferright!(ρ::TaylorSeries, h::TaylorSeries,
                            Q::TaylorSeries, Rs::Tuple{Vararg{TaylorSeries}},
                            Δx, Kmax, tol)

    Kmin = max(degree(Q), maximum(degree, Rs)) + 1
    temp = zero(Q[0])
    T = eltype(h[0])
    for k = 1:Kmax
        ρᵏ = rmul!(h[k-1], -1)
        # solve triangular system:
        for l = 0:min(k-1, degree(Q))
            mul!(ρᵏ, Q[l], ρ[k-1-l], -one(T), one(T))
            mul!(ρᵏ, ρ[k-1-l], Q[l]', -one(T), one(T))
        end
        for R in Rs
            for l = 0:min(k-1, degree(R))
                for m = 0:min(k-1-l, degree(R))
                    mul!(temp, ρ[k-1-l-m], R[m]', one(T), zero(T))
                    mul!(ρᵏ, R[l], temp, -one(T), one(T))
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
