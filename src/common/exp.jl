using LinearAlgebra
import LinearAlgebra: BlasFloat

exp_blocktriangular(A::AbstractMatrix, B::AbstractMatrix) =
    exp_blocktriangular!(copy(A), copy(B))

exp_blocktriangular(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix) =
    exp_blocktriangular!(copy(A), copy(B), copy(C))

exp_blocktriangular!(A::StridedMatrix{T}, B::StridedMatrix{T}) where {T<:BlasFloat} =
    exp_blocktriangular!(A, B, A)[1:2]

function exp_blocktriangular!(A::StridedMatrix{T}, B::StridedMatrix{T}, C::StridedMatrix{T}) where {T<:BlasFloat}
    # Dimension checking
    n1 = LinearAlgebra.checksquare(A)
    n2 = LinearAlgebra.checksquare(C)
    (n1, n2) == size(B) ||
        throw(DimensionMismatch("Size of $B is $(size(B)), expected ($n1, $n2)"))

    AeqC = A === C
    # Balancing
    ilo1, ihi1, scale1 = LAPACK.gebal!('B', A)    # modifies A
    if AeqC
        ilo2, ihi2, scale2 = ilo1, ihi1, scale1
    else
        ilo2, ihi2, scale2 = LAPACK.gebal!('B', C)    # modifies C
    end
    B = _balance2!(B, ilo1, ihi1, scale1, ilo2, ihi2, scale2) # modifies B
    nAC = max(opnorm(A, 1), opnorm(C, 1))

    ## For sufficiently small nAC, use lower order Padé-Approximations
    if (nAC <= 2.1)
        s = 0
        if nAC > 0.95
            AUpV, AVmU, BUpV, BVmU, CUpV, CVmU = exp_blocktriangular_pade9(A, B, C)
        elseif nAC > 0.25
            AUpV, AVmU, BUpV, BVmU, CUpV, CVmU = exp_blocktriangular_pade7(A, B, C)
        elseif nAC > 0.015
            AUpV, AVmU, BUpV, BVmU, CUpV, CVmU = exp_blocktriangular_pade5(A, B, C)
        else
            AUpV, AVmU, BUpV, BVmU, CUpV, CVmU = exp_blocktriangular_pade3(A, B, C)
        end
    else
        s  = ceil(Int, log2(nAC/5.4)) # power of 2 later reversed by squaring
        if s > 0
            factor = convert(T, 2^s)
            A ./= factor
            B ./= factor
            if !AeqC
                C ./= factor
            end
        end
        AUpV, AVmU, BUpV, BVmU, CUpV, CVmU = exp_blocktriangular_pade13(A, B, C)
     end
     AF = lu!(AVmU)
     XA = ldiv!(AF, AUpV)
     XC = AeqC ? XA : ldiv!(lu!(CVmU), CUpV)
     XB = ldiv!(AF, mul!(BUpV, BVmU, XC, -1, 1))

     if s > 0
         # recylce memory
         XA′ = AVmU; XB′ = BVmU; XC′ = CVmU;
         for t = 1:s
             XA′ = mul!(XA′, XA, XA)
             XB′ = mul!(mul!(XB′, XA, XB), XB, XC, true, true)
             XA, XA′ = XA′, XA
             XB, XB′ = XB′, XB
             if AeqC
                 XC = XA
             else
                 XC′ = mul!(XC′, XC, XC)
                 XC, XC′ = XC′, XC
             end
         end
     end

    # Undo the balancing
    XA = _unbalance!(XA, ilo1, ihi1, scale1) # modifies XA
    XC = AeqC ? XA : _unbalance!(XC, ilo2, ihi2, scale2) # modifies XC
    XB = _unbalance2!(XB, ilo1, ihi1, scale1, ilo2, ihi2, scale2) # modifies XB
    return XA, XB, XC
end

function exp_blocktriangular_pade13(A, B, C)
    T = eltype(A)
    coeffs = T[64764752532480000., 32382376266240000., 7771770303897600., 1187353796428800.,
            129060195264000., 10559470521600., 670442572800., 33522128640., 1323241920.,
            40840800., 960960., 16380., 182., 1.]

    AeqC = A === C
    A0 = one(A)
    B0 = zero(B)
    C0 = AeqC ? A0 : one(C)
    A2 = A * A
    C2 = AeqC ? A2 : C * C
    B2 = mul!(A * B, B, C, true, true) # A * B + B * C
    A4 = A2 * A2
    C4 = AeqC ? A4 : C2 * C2
    B4 = mul!(A2 * B2, B2, C2, true, true) # A2 * B2 + B2 * C2
    A6 = A4 * A2
    C6 = AeqC ? A6 : C4 * C2
    B6 = mul!(A4 * B2, B4, C2, true, true) # A2 * B2 + B2 * C2

    AV′ = coeffs[13] .* A6 .+ coeffs[11] .* A4 .+ coeffs[9] .* A2
    AV = coeffs[7] .* A6 .+ coeffs[5] .* A4 .+ coeffs[3] .* A2 .+ coeffs[1] .* A0
    AV = mul!(AV, A6, AV′, true, true)

    if AeqC
        CV, CV′ = AV, AV′
    else
        CV′ = coeffs[13] .* C6 .+ coeffs[11] .* C4 .+ coeffs[9] .* C2
        CV = coeffs[7] .* C6 .+ coeffs[5] .* C4 .+ coeffs[3] .* C2 .+ coeffs[1] .* C0
        CV = mul!(CV, C6, CV′, true, true)
    end

    BV′ = coeffs[13] .* B6 .+ coeffs[11] .* B4 .+ coeffs[9] .* B2
    BV = coeffs[7] .* B6 .+ coeffs[5] .* B4 .+ coeffs[3] .* B2 .+ coeffs[1] .* B0
    BV = mul!(mul!(BV, A6, BV′, true, true), B6, CV′, true, true)

    AW = A0; BW = B0; CW = C0; AW′ = AV′; BW′ = BV′; CW′ = CV′;

    AW′ .= coeffs[14] .* A6 .+ coeffs[12] .* A4 .+ coeffs[10] .* A2
    AW .= coeffs[8] .* A6 .+ coeffs[6] .* A4 .+ coeffs[4] .* A2 .+ coeffs[2] .* A0
    AW = mul!(AW, A6, AW′, true, true)

    if !AeqC
        CW′ .= coeffs[14] .* C6 .+ coeffs[12] .* C4 .+ coeffs[10] .* C2
        CW .= coeffs[8] .* C6 .+ coeffs[6] .* C4 .+ coeffs[4] .* C2 .+ coeffs[2] .* C0
        CW = mul!(CW, C6, CW′, true, true)
    end

    BW′ .= coeffs[14] .* B6 .+ coeffs[12] .* B4 .+ coeffs[10] .* B2
    BW .= coeffs[8] .* B6 .+ coeffs[6] .* B4 .+ coeffs[4] .* B2 .+ coeffs[2] .* B0
    BW = mul!(mul!(BW, A6, BW′, true, true), B6, CW′, true, true)

    AU = mul!(A2, A, AW)
    BU = mul!(mul!(B2, A, BW), B, CW, true, true)
    CU = AeqC ? AU : mul!(C2, C, CW)

    A0 .= AU .+ AV
    B0 .= BU .+ BV
    if !AeqC
        C0 .= CU .+ CV
    end

    AV .-= AU
    BV .-= BU
    if !AeqC
        CV .-= CU
    end

    return A0, AV, B0, BV, C0, CV
end

function exp_blocktriangular_pade9(A, B, C)
    T = eltype(A)
    coeffs = T[17643225600., 8821612800., 2075673600., 302702400., 30270240., 2162160.,
                110880., 3960., 90., 1.]

    AeqC = A === C
    A0 = one(A)
    B0 = zero(B)
    C0 = AeqC ? A0 : one(C)
    A2 = A * A
    C2 = AeqC ? A2 : C * C
    B2 = mul!(A * B, B, C, true, true) # A * B + B * C
    A4 = A2 * A2
    C4 = AeqC ? A4 : C2 * C2
    B4 = mul!(A2 * B2, B2, C2, true, true) # A2 * B2 + B2 * C2
    A6 = A4 * A2
    C6 = AeqC ? A6 : C4 * C2
    B6 = mul!(A4 * B2, B4, C2, true, true) # A2 * B2 + B2 * C2

    AV = coeffs[7] .* A6 .+ coeffs[5] .* A4 .+ coeffs[3] .* A2 .+ coeffs[1] .* A0
    BV = coeffs[7] .* B6 .+ coeffs[5] .* B4 .+ coeffs[3] .* B2 .+ coeffs[1] .* B0
    if AeqC
        CV = AV
    else
        CV = coeffs[7] .* C6 .+ coeffs[5] .* C4 .+ coeffs[3] .* C2 .+ coeffs[1] .* C0
    end

    AW = A0; BW = B0; CW = C0;
    AW .= coeffs[8] .* A6 .+ coeffs[6] .* A4 .+ coeffs[4] .* A2 .+ coeffs[2] .* A0
    BW .= coeffs[8] .* B6 .+ coeffs[6] .* B4 .+ coeffs[4] .* B2 .+ coeffs[2] .* B0
    if !AeqC
        CW .= coeffs[8] .* C6 .+ coeffs[6] .* C4 .+ coeffs[4] .* C2 .+ coeffs[2] .* C0
    end

    A8 = mul!(A4, A6, A2)
    B8 = mul!(mul!(B4, A6, B2), B6, C2, true, true)
    C8 = AeqC ? A8 : mul!(C4, C6, C2)

    AV .+= coeffs[9] .* A8
    BV .+= coeffs[9] .* B8
    if !AeqC
        CV .+= coeffs[9] .* C8
    end

    AW .+= coeffs[10] .* A8
    BW .+= coeffs[10] .* B8
    if !AeqC
        CW .+= coeffs[10] .* C8
    end

    AU = mul!(A2, A, AW)
    BU = mul!(mul!(B2, A, BW), B, CW, true, true)
    CU = AeqC ? AU : mul!(C2, C, CW)

    A0 .= AU .+ AV
    AV .-= AU
    B0 .= BU .+ BV
    BV .-= BU
    if !AeqC
        C0 .= CU .+ CV
        CV .-= CU
    end

    return A0, AV, B0, BV, C0, CV
end

function exp_blocktriangular_pade7(A, B, C)
    T = eltype(A)
    coeffs = T[17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.]

    AeqC = A === C
    A0 = one(A)
    B0 = zero(B)
    C0 = AeqC ? A0 : one(C)
    A2 = A * A
    C2 = AeqC ? A2 : C * C
    B2 = mul!(A * B, B, C, true, true) # A * B + B * C
    A4 = A2 * A2
    C4 = AeqC ? A4 : C2 * C2
    B4 = mul!(A2 * B2, B2, C2, true, true) # A2 * B2 + B2 * C2
    A6 = A4 * A2
    C6 = AeqC ? A6 : C4 * C2
    B6 = mul!(A4 * B2, B4, C2, true, true) # A2 * B2 + B2 * C2

    AV = coeffs[7] .* A6 .+ coeffs[5] .* A4 .+ coeffs[3] .* A2 .+ coeffs[1] .* A0
    BV = coeffs[7] .* B6 .+ coeffs[5] .* B4 .+ coeffs[3] .* B2 .+ coeffs[1] .* B0
    if AeqC
        CV = AV
    else
        CV = coeffs[7] .* C6 .+ coeffs[5] .* C4 .+ coeffs[3] .* C2 .+ coeffs[1] .* C0
    end

    AW = A0; BW = B0; CW = C0;
    AW .= coeffs[8] .* A6 .+ coeffs[6] .* A4 .+ coeffs[4] .* A2 .+ coeffs[2] .* A0
    BW .= coeffs[8] .* B6 .+ coeffs[6] .* B4 .+ coeffs[4] .* B2 .+ coeffs[2] .* B0
    if !AeqC
        CW .= coeffs[8] .* C6 .+ coeffs[6] .* C4 .+ coeffs[4] .* C2 .+ coeffs[2] .* C0
    end

    AU = mul!(A2, A, AW)
    BU = mul!(mul!(B2, A, BW), B, CW, true, true)
    CU = AeqC ? AU : mul!(C2, C, CW)

    A0 .= AU .+ AV
    AV .-= AU
    B0 .= BU .+ BV
    BV .-= BU
    if !AeqC
        C0 .= CU .+ CV
        CV .-= CU
    end

    return A0, AV, B0, BV, C0, CV
end

function exp_blocktriangular_pade5(A, B, C)
    T = eltype(A)
    coeffs = T[30240., 15120., 3360., 420., 30., 1.]

    AeqC = A === C
    A0 = one(A)
    B0 = zero(B)
    C0 = AeqC ? A0 : one(C)
    A2 = A * A
    C2 = AeqC ? A2 : C * C
    B2 = mul!(A * B, B, C, true, true) # A * B + B * C
    A4 = A2 * A2
    C4 = AeqC ? A4 : C2 * C2
    B4 = mul!(A2 * B2, B2, C2, true, true) # A2 * B2 + B2 * C2

    AV = coeffs[5] .* A4 .+ coeffs[3] .* A2 .+ coeffs[1] .* A0
    BV = coeffs[5] .* B4 .+ coeffs[3] .* B2 .+ coeffs[1] .* B0
    if AeqC
        CV = AV
    else
        CV = coeffs[5] .* C4 .+ coeffs[3] .* C2 .+ coeffs[1] .* C0
    end

    AW = A0; BW = B0; CW = C0;
    AW .= coeffs[6] .* A4 .+ coeffs[4] .* A2 .+ coeffs[2] .* A0
    BW .= coeffs[6] .* B4 .+ coeffs[4] .* B2 .+ coeffs[2] .* B0
    if !AeqC
        CW .= coeffs[6] .* C4 .+ coeffs[4] .* C2 .+ coeffs[2] .* C0
    end

    AU = mul!(A2, A, AW)
    BU = mul!(mul!(B2, A, BW), B, CW, true, true)
    CU = AeqC ? AU : mul!(C2, C, CW)

    A0 .= AU .+ AV
    AV .-= AU
    B0 .= BU .+ BV
    BV .-= BU
    if !AeqC
        C0 .= CU .+ CV
        CV .-= CU
    end

    return A0, AV, B0, BV, C0, CV
end

function exp_blocktriangular_pade3(A, B, C)
    T = eltype(A)
    coeffs = T[120., 60., 12., 1.]

    AeqC = A === C
    A0 = one(A)
    B0 = zero(B)
    C0 = AeqC ? A0 : one(C)
    A2 = A * A
    C2 = AeqC ? A2 : C * C
    B2 = mul!(A * B, B, C, true, true) # A * B + B * C

    AV = coeffs[3] .* A2 .+ coeffs[1] .* A0
    BV = coeffs[3] .* B2 .+ coeffs[1] .* B0
    if AeqC
        CV = AV
    else
        CV = coeffs[3] .* C2 .+ coeffs[1] .* C0
    end

    AW = A0; BW = B0; CW = C0;
    AW .= coeffs[4] .* A2 .+ coeffs[2] .* A0
    BW .= coeffs[4] .* B2 .+ coeffs[2] .* B0
    if !AeqC
        CW .= coeffs[4] .* C2 .+ coeffs[2] .* C0
    end

    AU = mul!(A2, A, AW)
    BU = mul!(mul!(B2, A, BW), B, CW, true, true)
    CU = AeqC ? AU : mul!(C2, C, CW)

    A0 .= AU .+ AV
    AV .-= AU
    B0 .= BU .+ BV
    BV .-= BU
    if !AeqC
        C0 .= CU .+ CV
        CV .-= CU
    end

    return A0, AV, B0, BV, C0, CV
end

## Swap rows i and j and columns i and j in X
function _rcswap!(X::StridedMatrix, i::Integer, j::Integer)
    for k = 1:size(X,1)
        X[k,i], X[k,j] = X[k,j], X[k,i]
    end
    for k = 1:size(X,2)
        X[i,k], X[j,k] = X[j,k], X[i,k]
    end
    return X
end
function _rswap!(X::StridedMatrix, i::Integer, j::Integer)
    for k = 1:size(X,2)
        X[i,k], X[j,k] = X[j,k], X[i,k]
    end
    return X
end
function _cswap!(X::StridedMatrix, i::Integer, j::Integer)
    for k = 1:size(X,1)
        X[k,i], X[k,j] = X[k,j], X[k,i]
    end
    return X
end

function _balance!(X, ilo, ihi, scale)
    n = LinearAlgebra.checksquare(X)
    if ihi < n
        for j in n:-1:(ihi + 1)
            X = _rcswap!(X, j, Int(scale[j]))
        end
    end
    if ilo > 1
        for j in 1:(ilo - 1)
            X = _rcswap!(X, j, Int(scale[j]))
        end
    end

    for j in ilo:ihi
        scj = scale[j]
        for i in 1:n
            X[j, i] /= scj
        end
        for i in 1:n
            X[i, j] *= scj
        end
    end
    return X
end

# Reverse of _balance!
function _unbalance!(X, ilo, ihi, scale)
    n = LinearAlgebra.checksquare(X)
    for j in ilo:ihi
        scj = scale[j]
        for i in 1:n
            X[j, i] *= scj
        end
        for i in 1:n
            X[i, j] /= scj
        end
    end

    if ilo > 1
        for j in (ilo - 1):-1:1
            X = _rcswap!(X, j, Int(scale[j]))
        end
    end
    if ihi < n
        for j in (ihi + 1):n
            X = _rcswap!(X, j, Int(scale[j]))
        end
    end
    return X
end

function _balance2!(X, ilo1, ihi1, scale1, ilo2, ihi2, scale2)
    n1, n2 = size(X)
    if ihi1 < n1
        for j in n1:-1:(ihi1 + 1)
            X = _rswap!(X, j, Int(scale1[j]))
        end
    end
    if ilo1 > 1
        for j in 1:(ilo1 - 1)
            X = _rswap!(X, j, Int(scale1[j]))
        end
    end
    if ihi2 < n2
        for j in n2:-1:(ihi2 + 1)
            X = _cswap!(X, j, Int(scale2[j]))
        end
    end
    if ilo2 > 1
        for j in 1:(ilo2 - 1)
            X = _cswap!(X, j, Int(scale2[j]))
        end
    end
    for j in ilo1:ihi1
        scj = scale1[j]
        for i in 1:n2
            X[j, i] /= scj
        end
    end
    for j in ilo2:ihi2
        scj = scale2[j]
        for i in 1:n1
            X[i, j] *= scj
        end
    end
    return X
end

# Reverse of _balance2!
function _unbalance2!(X, ilo1, ihi1, scale1, ilo2, ihi2, scale2)
    n1, n2 = size(X)
    for j in ilo2:ihi2
        scj = scale2[j]
        for i in 1:n1
            X[i, j] /= scj
        end
    end
    for j in ilo1:ihi1
        scj = scale1[j]
        for i in 1:n2
            X[j, i] *= scj
        end
    end
    if ilo2 > 1
        for j in (ilo2 - 1):-1:1
            X = _cswap!(X, j, Int(scale2[j]))
        end
    end
    if ihi2 < n2
        for j in (ihi2 + 1):n2
            X = _cswap!(X, j, Int(scale2[j]))
        end
    end
    if ilo1 > 1
        for j in (ilo1 - 1):-1:1
            X = _rswap!(X, j, Int(scale1[j]))
        end
    end
    if ihi1 < n1
        for j in (ihi1 + 1):n1
            X = _rswap!(X, j, Int(scale1[j]))
        end
    end
    return X
end
