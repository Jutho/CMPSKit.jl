function ctns2cmps(V,α,χ)
    Q = zeros(Float64, χ, χ)
    R = zeros(Float64, χ, χ)
    s = (V^2 - abs(α)^4)^(1/4)
    for n = 1:χ
        Q[n,n] += 4*(n-1)^2*sqrt(s)
        Q[n,n] -= sqrt(2)*2*(n-1)*sqrt(n-1)
        Q[n,n] += 1/(2*s)*(V+1)*(2*n-1)
        if n+1 <= χ
            R[n+1,n] = α/s*sqrt(n/2)
        end
        if n-1 > 0
            R[n-1,n] = α/s*sqrt((n-1)/2)
        end
        if n+2 <= χ
            Q[n+2,n] -= sqrt(2)*(n-1)*sqrt(n)
            Q[n+2,n] += 1/(2*s)*(V+1)*sqrt((n+1)*(n))
        end
        if n-2 > 0
            Q[n-2,n] -= sqrt(2)*(n-3)*sqrt(n-2)
            Q[n-2,n] += 1/(2*s)*(V+1)*sqrt((n-1)*(n-2))
        end
    end
    return Q, R
end



function ctns2cmps(V,α,χ)
    M = real([V (- α * adjoint(α) / 2); (- conj(α) * transpose(α) / 2) conj(V)])
    Msq = sqrt(M)
    if !isreal(Msq)
        error("invalid state")
    end
    Ω = real(Msq)


    D,U = eigen(M)

    Q = zeros(Float64, χ, χ)
    R = zeros(Float64, χ, χ)
    s = (V^2 - abs(α)^4)^(1/4)
    for n = 1:χ
        Q[n,n] += 4*(n-1)^2*sqrt(s)
        Q[n,n] -= sqrt(2)*2*(n-1)*sqrt(n-1)
        Q[n,n] += 1/(2*s)*(V+1)*(2*n-1)
        if n+1 <= χ
            R[n+1,n] = α/s*sqrt(n/2)
        end
        if n-1 > 0
            R[n-1,n] = α/s*sqrt((n-1)/2)
        end
        if n+2 <= χ
            Q[n+2,n] -= sqrt(2)*(n-1)*sqrt(n)
            Q[n+2,n] += 1/(2*s)*(V+1)*sqrt((n+1)*(n))
        end
        if n-2 > 0
            Q[n-2,n] -= sqrt(2)*(n-3)*sqrt(n-2)
            Q[n-2,n] += 1/(2*s)*(V+1)*sqrt((n-1)*(n-2))
        end
    end
    return Q, R
end
