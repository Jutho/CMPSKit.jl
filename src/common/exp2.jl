using LinearAlgebra
import LinearAlgebra: BlasFloat

function exp_blocktriangular2(A11::AbstractMatrix, A22::AbstractMatrix, A33::AbstractMatrix,
                              B12::AbstractMatrix, B23::AbstractMatrix)
    return exp_blocktriangular2!(copy.((A11, A22, A33, B12, B23))...)
end

function exp_blocktriangular2!(A11::StridedMatrix{T}, A22::StridedMatrix{T},
                               A33::StridedMatrix{T}, B12::StridedMatrix{T},
                               B23::StridedMatrix{T}) where {T<:BlasFloat}
    # Dimension checking
    n1 = LinearAlgebra.checksquare(A11)
    n2 = LinearAlgebra.checksquare(A22)
    n3 = LinearAlgebra.checksquare(A33)
    (n1, n2) == size(B12) ||
        throw(DimensionMismatch("Size of $B is $(size(B12)), expected ($n1, $n2)"))
    (n2, n3) == size(B23) ||
        throw(DimensionMismatch("Size of $B is $(size(B23)), expected ($n2, $n3)"))

    # Balancing
    ilo1, ihi1, scale1 = LAPACK.gebal!('B', A11)    # modifies A11
    ilo2, ihi2, scale2 = LAPACK.gebal!('B', A22)    # modifies A22
    ilo3, ihi3, scale3 = LAPACK.gebal!('B', A33)    # modifies A33
    B12 = _balance2!(B12, ilo1, ihi1, scale1, ilo2, ihi2, scale2) # modifies B12
    B23 = _balance2!(B23, ilo2, ihi2, scale2, ilo3, ihi3, scale3) # modifies B23
    nA = max(opnorm(A11, 1), opnorm(A22, 1), opnorm(A33, 1))

    ## For sufficiently small nA, use lower order Padé-Approximations
    if (nA <= 2.1)
        s = 0
        if nA > 0.95
            A1UpV, A1VmU, A2UpV, A2VmU, A3UpV, A3VmU,
            B12UpV, B12VmU, B23UpV, B23VmU, C13UpV, C13VmU = exp_blocktriangular2_pade9(A11,
                                                                                        A22,
                                                                                        A33,
                                                                                        B12,
                                                                                        B23)
        elseif nA > 0.25
            A1UpV, A1VmU, A2UpV, A2VmU, A3UpV, A3VmU,
            B12UpV, B12VmU, B23UpV, B23VmU, C13UpV, C13VmU = exp_blocktriangular2_pade7(A11,
                                                                                        A22,
                                                                                        A33,
                                                                                        B12,
                                                                                        B23)
        elseif nA > 0.015
            A1UpV, A1VmU, A2UpV, A2VmU, A3UpV, A3VmU,
            B12UpV, B12VmU, B23UpV, B23VmU, C13UpV, C13VmU = exp_blocktriangular2_pade5(A11,
                                                                                        A22,
                                                                                        A33,
                                                                                        B12,
                                                                                        B23)
        else
            A1UpV, A1VmU, A2UpV, A2VmU, A3UpV, A3VmU,
            B12UpV, B12VmU, B23UpV, B23VmU, C13UpV, C13VmU = exp_blocktriangular2_pade3(A11,
                                                                                        A22,
                                                                                        A33,
                                                                                        B12,
                                                                                        B23)
        end
    else
        s = ceil(Int, log2(nA / 5.4)) # power of 2 later reversed by squaring
        if s > 0
            factor = convert(T, 2^s)
            A11 ./= factor
            A22 ./= factor
            A33 ./= factor
            B12 ./= factor
            B23 ./= factor
        end
        A1UpV, A1VmU, A2UpV, A2VmU, A3UpV, A3VmU,
        B12UpV, B12VmU, B23UpV, B23VmU, C13UpV, C13VmU = exp_blocktriangular2_pade13(A11,
                                                                                     A22,
                                                                                     A33,
                                                                                     B12,
                                                                                     B23)
    end

    A1F = lu!(A1VmU)
    XA1 = ldiv!(A1F, A1UpV)
    A2F = lu!(A2VmU)
    XA2 = ldiv!(A2F, A2UpV)
    A3F = lu!(A3VmU)
    XA3 = ldiv!(A3F, A3UpV)

    XB12 = ldiv!(A1F, mul!(B12UpV, B12VmU, XA2, -1, 1))
    XB23 = ldiv!(A2F, mul!(B23UpV, B23VmU, XA3, -1, 1))
    XC13 = ldiv!(A1F, mul!(mul!(C13UpV, C13VmU, XA3, -1, 1), B12VmU, XB23, -1, 1))

    if s > 0
        # recylce memory
        XA1′ = A1VmU
        XA2′ = A2VmU
        XA3′ = A3VmU
        XB12′ = B12VmU
        XB23′ = B23VmU
        XC13′ = C13VmU
        for t in 1:s
            XA1′ = mul!(XA1′, XA1, XA1)
            XA2′ = mul!(XA2′, XA2, XA2)
            XA3′ = mul!(XA3′, XA3, XA3)
            XB12′ = mul!(mul!(XB12′, XA1, XB12), XB12, XA2, true, true)
            XB23′ = mul!(mul!(XB23′, XA2, XB23), XB23, XA3, true, true)
            XC13′ = mul!(mul!(mul!(XC13′, XB12, XB23), XA1, XC13, true, true), XC13, XA3,
                         true, true)
            XA1, XA1′ = XA1′, XA1
            XA2, XA2′ = XA2′, XA2
            XA3, XA3′ = XA3′, XA3
            XB12, XB12′ = XB12′, XB12
            XB23, XB23′ = XB23′, XB23
            XC13, XC13′ = XC13′, XC13
        end
    end

    # Undo the balancing
    XA1 = _unbalance!(XA1, ilo1, ihi1, scale1) # modifies XA
    XA2 = _unbalance!(XA2, ilo2, ihi2, scale2) # modifies XA
    XA3 = _unbalance!(XA3, ilo3, ihi3, scale3) # modifies XA
    XB12 = _unbalance2!(XB12, ilo1, ihi1, scale1, ilo2, ihi2, scale2) # modifies XB
    XB23 = _unbalance2!(XB23, ilo2, ihi2, scale2, ilo3, ihi3, scale3) # modifies XB
    XC13 = _unbalance2!(XC13, ilo1, ihi1, scale1, ilo3, ihi3, scale3) # modifies XB
    return XA1, XA2, XA3, XB12, XB23, XC13
end

function exp_blocktriangular2_pade13(A11, A22, A33, B12, B23)
    T = eltype(A11)
    coeffs = T[64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
               1187353796428800.0,
               129060195264000.0, 10559470521600.0, 670442572800.0, 33522128640.0,
               1323241920.0,
               40840800.0, 960960.0, 16380.0, 182.0, 1.0]

    A11_0 = one(A11)
    A22_0 = one(A22)
    A33_0 = one(A33)
    B12_0 = zero(B12)
    B23_0 = zero(B23)
    A11_2 = A11 * A11
    A22_2 = A22 * A22
    A33_2 = A33 * A33
    B12_2 = mul!(A11 * B12, B12, A22, true, true)
    B23_2 = mul!(A22 * B23, B23, A33, true, true)
    C13_2 = B12 * B23
    C13_0 = zero(C13_2)
    A11_4 = A11_2 * A11_2
    A22_4 = A22_2 * A22_2
    A33_4 = A33_2 * A33_2
    B12_4 = mul!(A11_2 * B12_2, B12_2, A22_2, true, true)
    B23_4 = mul!(A22_2 * B23_2, B23_2, A33_2, true, true)
    C13_4 = mul!(mul!(B12_2 * B23_2, A11_2, C13_2, true, true), C13_2, A33_2, true, true)
    A11_6 = A11_4 * A11_2
    A22_6 = A22_4 * A22_2
    A33_6 = A33_4 * A33_2
    B12_6 = mul!(A11_4 * B12_2, B12_4, A22_2, true, true)
    B23_6 = mul!(A22_4 * B23_2, B23_4, A33_2, true, true)
    C13_6 = mul!(mul!(B12_4 * B23_2, A11_4, C13_2, true, true), C13_4, A33_2, true, true)

    A11_V′ = coeffs[13] .* A11_6 .+ coeffs[11] .* A11_4 .+ coeffs[9] .* A11_2
    A22_V′ = coeffs[13] .* A22_6 .+ coeffs[11] .* A22_4 .+ coeffs[9] .* A22_2
    A33_V′ = coeffs[13] .* A33_6 .+ coeffs[11] .* A33_4 .+ coeffs[9] .* A33_2
    B12_V′ = coeffs[13] .* B12_6 .+ coeffs[11] .* B12_4 .+ coeffs[9] .* B12_2
    B23_V′ = coeffs[13] .* B23_6 .+ coeffs[11] .* B23_4 .+ coeffs[9] .* B23_2
    C13_V′ = coeffs[13] .* C13_6 .+ coeffs[11] .* C13_4 .+ coeffs[9] .* C13_2
    A11_V = coeffs[7] .* A11_6 .+ coeffs[5] .* A11_4 .+ coeffs[3] .* A11_2 .+
            coeffs[1] .* A11_0
    A22_V = coeffs[7] .* A22_6 .+ coeffs[5] .* A22_4 .+ coeffs[3] .* A22_2 .+
            coeffs[1] .* A22_0
    A33_V = coeffs[7] .* A33_6 .+ coeffs[5] .* A33_4 .+ coeffs[3] .* A33_2 .+
            coeffs[1] .* A33_0
    B12_V = coeffs[7] .* B12_6 .+ coeffs[5] .* B12_4 .+ coeffs[3] .* B12_2 .+
            coeffs[1] .* B12_0
    B23_V = coeffs[7] .* B23_6 .+ coeffs[5] .* B23_4 .+ coeffs[3] .* B23_2 .+
            coeffs[1] .* B23_0
    C13_V = coeffs[7] .* C13_6 .+ coeffs[5] .* C13_4 .+ coeffs[3] .* C13_2 .+
            coeffs[1] .* C13_0
    A11_V = mul!(A11_V, A11_6, A11_V′, true, true)
    A22_V = mul!(A22_V, A22_6, A22_V′, true, true)
    A33_V = mul!(A33_V, A33_6, A33_V′, true, true)
    B12_V = mul!(mul!(B12_V, A11_6, B12_V′, true, true), B12_6, A22_V′, true, true)
    B23_V = mul!(mul!(B23_V, A22_6, B23_V′, true, true), B23_6, A33_V′, true, true)
    C13_V = mul!(mul!(mul!(C13_V, B12_6, B23_V′, true, true), A11_6, C13_V′, true, true),
                 C13_6, A33_V′, true, true)

    A11_W′ = A11_V′
    A22_W′ = A22_V′
    A33_W′ = A33_V′
    B12_W′ = B12_V′
    B23_W′ = B23_V′
    C13_W′ = C13_V′
    A11_W = A11_0
    A22_W = A22_0
    A33_W = A33_0
    B12_W = B12_0
    B23_W = B23_0
    C13_W = C13_0
    A11_W′ .= coeffs[14] .* A11_6 .+ coeffs[12] .* A11_4 .+ coeffs[10] .* A11_2
    A22_W′ .= coeffs[14] .* A22_6 .+ coeffs[12] .* A22_4 .+ coeffs[10] .* A22_2
    A33_W′ .= coeffs[14] .* A33_6 .+ coeffs[12] .* A33_4 .+ coeffs[10] .* A33_2
    B12_W′ .= coeffs[14] .* B12_6 .+ coeffs[12] .* B12_4 .+ coeffs[10] .* B12_2
    B23_W′ .= coeffs[14] .* B23_6 .+ coeffs[12] .* B23_4 .+ coeffs[10] .* B23_2
    C13_W′ .= coeffs[14] .* C13_6 .+ coeffs[12] .* C13_4 .+ coeffs[10] .* C13_2
    A11_W .= coeffs[8] .* A11_6 .+ coeffs[6] .* A11_4 .+ coeffs[4] .* A11_2 .+
             coeffs[2] .* A11_0
    A22_W .= coeffs[8] .* A22_6 .+ coeffs[6] .* A22_4 .+ coeffs[4] .* A22_2 .+
             coeffs[2] .* A22_0
    A33_W .= coeffs[8] .* A33_6 .+ coeffs[6] .* A33_4 .+ coeffs[4] .* A33_2 .+
             coeffs[2] .* A33_0
    B12_W .= coeffs[8] .* B12_6 .+ coeffs[6] .* B12_4 .+ coeffs[4] .* B12_2 .+
             coeffs[2] .* B12_0
    B23_W .= coeffs[8] .* B23_6 .+ coeffs[6] .* B23_4 .+ coeffs[4] .* B23_2 .+
             coeffs[2] .* B23_0
    C13_W .= coeffs[8] .* C13_6 .+ coeffs[6] .* C13_4 .+ coeffs[4] .* C13_2 .+
             coeffs[2] .* C13_0
    A11_W = mul!(A11_W, A11_6, A11_W′, true, true)
    A22_W = mul!(A22_W, A22_6, A22_W′, true, true)
    A33_W = mul!(A33_W, A33_6, A33_W′, true, true)
    B12_W = mul!(mul!(B12_W, A11_6, B12_W′, true, true), B12_6, A22_W′, true, true)
    B23_W = mul!(mul!(B23_W, A22_6, B23_W′, true, true), B23_6, A33_W′, true, true)
    C13_W = mul!(mul!(mul!(C13_W, B12_6, B23_W′, true, true), A11_6, C13_W′, true, true),
                 C13_6, A33_W′, true, true)

    A11_U = mul!(A11_2, A11, A11_W)
    A22_U = mul!(A22_2, A22, A22_W)
    A33_U = mul!(A33_2, A33, A33_W)
    B12_U = mul!(mul!(B12_2, A11, B12_W), B12, A22_W, true, true)
    B23_U = mul!(mul!(B23_2, A22, B23_W), B23, A33_W, true, true)
    C13_U = mul!(mul!(C13_2, A11, C13_W), B12, B23_W, true, true) # + C13 * A33_W, but C13=0

    A11_0 .= A11_U .+ A11_V
    A22_0 .= A22_U .+ A22_V
    A33_0 .= A33_U .+ A33_V
    B12_0 .= B12_U .+ B12_V
    B23_0 .= B23_U .+ B23_V
    C13_0 .= C13_U .+ C13_V
    A11_V .-= A11_U
    A22_V .-= A22_U
    A33_V .-= A33_U
    B12_V .-= B12_U
    B23_V .-= B23_U
    C13_V .-= C13_U

    return A11_0, A11_V, A22_0, A22_V, A33_0, A33_V,
           B12_0, B12_V, B23_0, B23_V, C13_0, C13_V
end

function exp_blocktriangular2_pade9(A11, A22, A33, B12, B23)
    T = eltype(A11)
    coeffs = T[17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 30270240.0,
               2162160.0,
               110880.0, 3960.0, 90.0, 1.0]

    A11_0 = one(A11)
    A22_0 = one(A22)
    A33_0 = one(A33)
    B12_0 = zero(B12)
    B23_0 = zero(B23)
    A11_2 = A11 * A11
    A22_2 = A22 * A22
    A33_2 = A33 * A33
    B12_2 = mul!(A11 * B12, B12, A22, true, true)
    B23_2 = mul!(A22 * B23, B23, A33, true, true)
    C13_2 = B12 * B23
    C13_0 = zero(C13_2)
    A11_4 = A11_2 * A11_2
    A22_4 = A22_2 * A22_2
    A33_4 = A33_2 * A33_2
    B12_4 = mul!(A11_2 * B12_2, B12_2, A22_2, true, true)
    B23_4 = mul!(A22_2 * B23_2, B23_2, A33_2, true, true)
    C13_4 = mul!(mul!(B12_2 * B23_2, A11_2, C13_2, true, true), C13_2, A33_2, true, true)
    A11_6 = A11_4 * A11_2
    A22_6 = A22_4 * A22_2
    A33_6 = A33_4 * A33_2
    B12_6 = mul!(A11_4 * B12_2, B12_4, A22_2, true, true)
    B23_6 = mul!(A22_4 * B23_2, B23_4, A33_2, true, true)
    C13_6 = mul!(mul!(B12_4 * B23_2, A11_4, C13_2, true, true), C13_4, A33_2, true, true)

    A11_V = coeffs[7] .* A11_6 .+ coeffs[5] .* A11_4 .+ coeffs[3] .* A11_2 .+
            coeffs[1] .* A11_0
    A22_V = coeffs[7] .* A22_6 .+ coeffs[5] .* A22_4 .+ coeffs[3] .* A22_2 .+
            coeffs[1] .* A22_0
    A33_V = coeffs[7] .* A33_6 .+ coeffs[5] .* A33_4 .+ coeffs[3] .* A33_2 .+
            coeffs[1] .* A33_0
    B12_V = coeffs[7] .* B12_6 .+ coeffs[5] .* B12_4 .+ coeffs[3] .* B12_2 .+
            coeffs[1] .* B12_0
    B23_V = coeffs[7] .* B23_6 .+ coeffs[5] .* B23_4 .+ coeffs[3] .* B23_2 .+
            coeffs[1] .* B23_0
    C13_V = coeffs[7] .* C13_6 .+ coeffs[5] .* C13_4 .+ coeffs[3] .* C13_2 .+
            coeffs[1] .* C13_0

    A11_W = A11_0
    A22_W = A22_0
    A33_W = A33_0
    B12_W = B12_0
    B23_W = B23_0
    C13_W = C13_0
    A11_W .= coeffs[8] .* A11_6 .+ coeffs[6] .* A11_4 .+ coeffs[4] .* A11_2 .+
             coeffs[2] .* A11_0
    A22_W .= coeffs[8] .* A22_6 .+ coeffs[6] .* A22_4 .+ coeffs[4] .* A22_2 .+
             coeffs[2] .* A22_0
    A33_W .= coeffs[8] .* A33_6 .+ coeffs[6] .* A33_4 .+ coeffs[4] .* A33_2 .+
             coeffs[2] .* A33_0
    B12_W .= coeffs[8] .* B12_6 .+ coeffs[6] .* B12_4 .+ coeffs[4] .* B12_2 .+
             coeffs[2] .* B12_0
    B23_W .= coeffs[8] .* B23_6 .+ coeffs[6] .* B23_4 .+ coeffs[4] .* B23_2 .+
             coeffs[2] .* B23_0
    C13_W .= coeffs[8] .* C13_6 .+ coeffs[6] .* C13_4 .+ coeffs[4] .* C13_2 .+
             coeffs[2] .* C13_0

    A11_8 = mul!(A11_4, A11_6, A11_2)
    A22_8 = mul!(A22_4, A22_6, A22_2)
    A33_8 = mul!(A33_4, A33_6, A33_2)
    B12_8 = mul!(mul!(B12_4, A11_6, B12_2), B12_6, A22_2, true, true)
    B23_8 = mul!(mul!(B23_4, A22_6, B23_2), B23_6, A33_2, true, true)
    C13_8 = mul!(mul!(mul!(C13_4, B12_6, B23_2), A11_6, C13_2, true, true), C13_6, A33_2,
                 true, true)

    A11_V .+= coeffs[9] .* A11_8
    A22_V .+= coeffs[9] .* A22_8
    A33_V .+= coeffs[9] .* A33_8
    B12_V .+= coeffs[9] .* B12_8
    B23_V .+= coeffs[9] .* B23_8
    C13_V .+= coeffs[9] .* C13_8

    A11_W .+= coeffs[10] .* A11_8
    A22_W .+= coeffs[10] .* A22_8
    A33_W .+= coeffs[10] .* A33_8
    B12_W .+= coeffs[10] .* B12_8
    B23_W .+= coeffs[10] .* B23_8
    C13_W .+= coeffs[10] .* C13_8

    A11_U = mul!(A11_2, A11, A11_W)
    A22_U = mul!(A22_2, A22, A22_W)
    A33_U = mul!(A33_2, A33, A33_W)
    B12_U = mul!(mul!(B12_2, A11, B12_W), B12, A22_W, true, true)
    B23_U = mul!(mul!(B23_2, A22, B23_W), B23, A33_W, true, true)
    C13_U = mul!(mul!(C13_2, A11, C13_W), B12, B23_W, true, true) # + C13 * A33_W, but C13=0

    A11_0 .= A11_U .+ A11_V
    A22_0 .= A22_U .+ A22_V
    A33_0 .= A33_U .+ A33_V
    B12_0 .= B12_U .+ B12_V
    B23_0 .= B23_U .+ B23_V
    C13_0 .= C13_U .+ C13_V
    A11_V .-= A11_U
    A22_V .-= A22_U
    A33_V .-= A33_U
    B12_V .-= B12_U
    B23_V .-= B23_U
    C13_V .-= C13_U

    return A11_0, A11_V, A22_0, A22_V, A33_0, A33_V,
           B12_0, B12_V, B23_0, B23_V, C13_0, C13_V
end

function exp_blocktriangular2_pade7(A11, A22, A33, B12, B23)
    T = eltype(A11)
    coeffs = T[17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0]

    A11_0 = one(A11)
    A22_0 = one(A22)
    A33_0 = one(A33)
    B12_0 = zero(B12)
    B23_0 = zero(B23)
    A11_2 = A11 * A11
    A22_2 = A22 * A22
    A33_2 = A33 * A33
    B12_2 = mul!(A11 * B12, B12, A22, true, true)
    B23_2 = mul!(A22 * B23, B23, A33, true, true)
    C13_2 = B12 * B23
    C13_0 = zero(C13_2)
    A11_4 = A11_2 * A11_2
    A22_4 = A22_2 * A22_2
    A33_4 = A33_2 * A33_2
    B12_4 = mul!(A11_2 * B12_2, B12_2, A22_2, true, true)
    B23_4 = mul!(A22_2 * B23_2, B23_2, A33_2, true, true)
    C13_4 = mul!(mul!(B12_2 * B23_2, A11_2, C13_2, true, true), C13_2, A33_2, true, true)
    A11_6 = A11_4 * A11_2
    A22_6 = A22_4 * A22_2
    A33_6 = A33_4 * A33_2
    B12_6 = mul!(A11_4 * B12_2, B12_4, A22_2, true, true)
    B23_6 = mul!(A22_4 * B23_2, B23_4, A33_2, true, true)
    C13_6 = mul!(mul!(B12_4 * B23_2, A11_4, C13_2, true, true), C13_4, A33_2, true, true)

    A11_V = coeffs[7] .* A11_6 .+ coeffs[5] .* A11_4 .+ coeffs[3] .* A11_2 .+
            coeffs[1] .* A11_0
    A22_V = coeffs[7] .* A22_6 .+ coeffs[5] .* A22_4 .+ coeffs[3] .* A22_2 .+
            coeffs[1] .* A22_0
    A33_V = coeffs[7] .* A33_6 .+ coeffs[5] .* A33_4 .+ coeffs[3] .* A33_2 .+
            coeffs[1] .* A33_0
    B12_V = coeffs[7] .* B12_6 .+ coeffs[5] .* B12_4 .+ coeffs[3] .* B12_2 .+
            coeffs[1] .* B12_0
    B23_V = coeffs[7] .* B23_6 .+ coeffs[5] .* B23_4 .+ coeffs[3] .* B23_2 .+
            coeffs[1] .* B23_0
    C13_V = coeffs[7] .* C13_6 .+ coeffs[5] .* C13_4 .+ coeffs[3] .* C13_2 .+
            coeffs[1] .* C13_0

    A11_W = A11_0
    A22_W = A22_0
    A33_W = A33_0
    B12_W = B12_0
    B23_W = B23_0
    C13_W = C13_0
    A11_W .= coeffs[8] .* A11_6 .+ coeffs[6] .* A11_4 .+ coeffs[4] .* A11_2 .+
             coeffs[2] .* A11_0
    A22_W .= coeffs[8] .* A22_6 .+ coeffs[6] .* A22_4 .+ coeffs[4] .* A22_2 .+
             coeffs[2] .* A22_0
    A33_W .= coeffs[8] .* A33_6 .+ coeffs[6] .* A33_4 .+ coeffs[4] .* A33_2 .+
             coeffs[2] .* A33_0
    B12_W .= coeffs[8] .* B12_6 .+ coeffs[6] .* B12_4 .+ coeffs[4] .* B12_2 .+
             coeffs[2] .* B12_0
    B23_W .= coeffs[8] .* B23_6 .+ coeffs[6] .* B23_4 .+ coeffs[4] .* B23_2 .+
             coeffs[2] .* B23_0
    C13_W .= coeffs[8] .* C13_6 .+ coeffs[6] .* C13_4 .+ coeffs[4] .* C13_2 .+
             coeffs[2] .* C13_0

    A11_U = mul!(A11_2, A11, A11_W)
    A22_U = mul!(A22_2, A22, A22_W)
    A33_U = mul!(A33_2, A33, A33_W)
    B12_U = mul!(mul!(B12_2, A11, B12_W), B12, A22_W, true, true)
    B23_U = mul!(mul!(B23_2, A22, B23_W), B23, A33_W, true, true)
    C13_U = mul!(mul!(C13_2, A11, C13_W), B12, B23_W, true, true) # + C13 * A33_W, but C13=0

    A11_0 .= A11_U .+ A11_V
    A22_0 .= A22_U .+ A22_V
    A33_0 .= A33_U .+ A33_V
    B12_0 .= B12_U .+ B12_V
    B23_0 .= B23_U .+ B23_V
    C13_0 .= C13_U .+ C13_V
    A11_V .-= A11_U
    A22_V .-= A22_U
    A33_V .-= A33_U
    B12_V .-= B12_U
    B23_V .-= B23_U
    C13_V .-= C13_U

    return A11_0, A11_V, A22_0, A22_V, A33_0, A33_V,
           B12_0, B12_V, B23_0, B23_V, C13_0, C13_V
end

function exp_blocktriangular2_pade5(A11, A22, A33, B12, B23)
    T = eltype(A11)
    coeffs = T[30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0]

    A11_0 = one(A11)
    A22_0 = one(A22)
    A33_0 = one(A33)
    B12_0 = zero(B12)
    B23_0 = zero(B23)
    A11_2 = A11 * A11
    A22_2 = A22 * A22
    A33_2 = A33 * A33
    B12_2 = mul!(A11 * B12, B12, A22, true, true)
    B23_2 = mul!(A22 * B23, B23, A33, true, true)
    C13_2 = B12 * B23
    C13_0 = zero(C13_2)
    A11_4 = A11_2 * A11_2
    A22_4 = A22_2 * A22_2
    A33_4 = A33_2 * A33_2
    B12_4 = mul!(A11_2 * B12_2, B12_2, A22_2, true, true)
    B23_4 = mul!(A22_2 * B23_2, B23_2, A33_2, true, true)
    C13_4 = mul!(mul!(B12_2 * B23_2, A11_2, C13_2, true, true), C13_2, A33_2, true, true)

    A11_V = coeffs[5] .* A11_4 .+ coeffs[3] .* A11_2 .+ coeffs[1] .* A11_0
    A22_V = coeffs[5] .* A22_4 .+ coeffs[3] .* A22_2 .+ coeffs[1] .* A22_0
    A33_V = coeffs[5] .* A33_4 .+ coeffs[3] .* A33_2 .+ coeffs[1] .* A33_0
    B12_V = coeffs[5] .* B12_4 .+ coeffs[3] .* B12_2 .+ coeffs[1] .* B12_0
    B23_V = coeffs[5] .* B23_4 .+ coeffs[3] .* B23_2 .+ coeffs[1] .* B23_0
    C13_V = coeffs[5] .* C13_4 .+ coeffs[3] .* C13_2 .+ coeffs[1] .* C13_0

    A11_W = A11_0
    A22_W = A22_0
    A33_W = A33_0
    B12_W = B12_0
    B23_W = B23_0
    C13_W = C13_0
    A11_W .= coeffs[6] .* A11_4 .+ coeffs[4] .* A11_2 .+ coeffs[2] .* A11_0
    A22_W .= coeffs[6] .* A22_4 .+ coeffs[4] .* A22_2 .+ coeffs[2] .* A22_0
    A33_W .= coeffs[6] .* A33_4 .+ coeffs[4] .* A33_2 .+ coeffs[2] .* A33_0
    B12_W .= coeffs[6] .* B12_4 .+ coeffs[4] .* B12_2 .+ coeffs[2] .* B12_0
    B23_W .= coeffs[6] .* B23_4 .+ coeffs[4] .* B23_2 .+ coeffs[2] .* B23_0
    C13_W .= coeffs[6] .* C13_4 .+ coeffs[4] .* C13_2 .+ coeffs[2] .* C13_0

    A11_U = mul!(A11_2, A11, A11_W)
    A22_U = mul!(A22_2, A22, A22_W)
    A33_U = mul!(A33_2, A33, A33_W)
    B12_U = mul!(mul!(B12_2, A11, B12_W), B12, A22_W, true, true)
    B23_U = mul!(mul!(B23_2, A22, B23_W), B23, A33_W, true, true)
    C13_U = mul!(mul!(C13_2, A11, C13_W), B12, B23_W, true, true) # + C13 * A33_W, but C13=0

    A11_0 .= A11_U .+ A11_V
    A22_0 .= A22_U .+ A22_V
    A33_0 .= A33_U .+ A33_V
    B12_0 .= B12_U .+ B12_V
    B23_0 .= B23_U .+ B23_V
    C13_0 .= C13_U .+ C13_V
    A11_V .-= A11_U
    A22_V .-= A22_U
    A33_V .-= A33_U
    B12_V .-= B12_U
    B23_V .-= B23_U
    C13_V .-= C13_U

    return A11_0, A11_V, A22_0, A22_V, A33_0, A33_V,
           B12_0, B12_V, B23_0, B23_V, C13_0, C13_V
end

function exp_blocktriangular2_pade3(A11, A22, A33, B12, B23)
    T = eltype(A11)
    coeffs = T[120.0, 60.0, 12.0, 1.0]

    A11_0 = one(A11)
    A22_0 = one(A22)
    A33_0 = one(A33)
    B12_0 = zero(B12)
    B23_0 = zero(B23)
    A11_2 = A11 * A11
    A22_2 = A22 * A22
    A33_2 = A33 * A33
    B12_2 = mul!(A11 * B12, B12, A22, true, true)
    B23_2 = mul!(A22 * B23, B23, A33, true, true)
    C13_2 = B12 * B23
    C13_0 = zero(C13_2)

    A11_V = coeffs[3] .* A11_2 .+ coeffs[1] .* A11_0
    A22_V = coeffs[3] .* A22_2 .+ coeffs[1] .* A22_0
    A33_V = coeffs[3] .* A33_2 .+ coeffs[1] .* A33_0
    B12_V = coeffs[3] .* B12_2 .+ coeffs[1] .* B12_0
    B23_V = coeffs[3] .* B23_2 .+ coeffs[1] .* B23_0
    C13_V = coeffs[3] .* C13_2 .+ coeffs[1] .* C13_0

    A11_W = A11_0
    A22_W = A22_0
    A33_W = A33_0
    B12_W = B12_0
    B23_W = B23_0
    C13_W = C13_0
    A11_W .= coeffs[4] .* A11_2 .+ coeffs[2] .* A11_0
    A22_W .= coeffs[4] .* A22_2 .+ coeffs[2] .* A22_0
    A33_W .= coeffs[4] .* A33_2 .+ coeffs[2] .* A33_0
    B12_W .= coeffs[4] .* B12_2 .+ coeffs[2] .* B12_0
    B23_W .= coeffs[4] .* B23_2 .+ coeffs[2] .* B23_0
    C13_W .= coeffs[4] .* C13_2 .+ coeffs[2] .* C13_0

    A11_U = mul!(A11_2, A11, A11_W)
    A22_U = mul!(A22_2, A22, A22_W)
    A33_U = mul!(A33_2, A33, A33_W)
    B12_U = mul!(mul!(B12_2, A11, B12_W), B12, A22_W, true, true)
    B23_U = mul!(mul!(B23_2, A22, B23_W), B23, A33_W, true, true)
    C13_U = mul!(mul!(C13_2, A11, C13_W), B12, B23_W, true, true) # + C13 * A33_W, but C13=0

    A11_0 .= A11_U .+ A11_V
    A22_0 .= A22_U .+ A22_V
    A33_0 .= A33_U .+ A33_V
    B12_0 .= B12_U .+ B12_V
    B23_0 .= B23_U .+ B23_V
    C13_0 .= C13_U .+ C13_V
    A11_V .-= A11_U
    A22_V .-= A22_U
    A33_V .-= A33_U
    B12_V .-= B12_U
    B23_V .-= B23_U
    C13_V .-= C13_U

    return A11_0, A11_V, A22_0, A22_V, A33_0, A33_V,
           B12_0, B12_V, B23_0, B23_V, C13_0, C13_V
end
