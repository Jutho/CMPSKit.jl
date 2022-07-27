using LinearAlgebra
import LinearAlgebra: BlasFloat

exp_blocktriangular_lazy(A::AbstractMatrix) =
    exp_blocktriangular_lazy!(copy(A))

exp_blocktriangular_lazy(A::AbstractMatrix, C::AbstractMatrix) =
    exp_blocktriangular_lazy!(copy(A), copy(C))

function exp_blocktriangular_lazy!(A::StridedMatrix{T}) where {T<:BlasFloat}
    XA, XC, fB = exp_blocktriangular_lazy!(A, A)
    return XA, fB
end

function exp_blocktriangular_lazy!(A::StridedMatrix{T}, C::StridedMatrix{T}) where {T<:BlasFloat}
    # Dimension checking
    n1 = LinearAlgebra.checksquare(A)
    n2 = LinearAlgebra.checksquare(C)

    AeqC = A === C
    # Balancing
    ilo1, ihi1, scale1 = LAPACK.gebal!('B', A)    # modifies A
    if AeqC
        ilo2, ihi2, scale2 = ilo1, ihi1, scale1
    else
        ilo2, ihi2, scale2 = LAPACK.gebal!('B', C)    # modifies C
    end
    nAC = max(opnorm(A, 1), opnorm(C, 1))

    ## For sufficiently small nAC, use lower order Padé-Approximations
    if (nAC <= 2.1)
        s = 0
        if nAC > 0.95
            AUpV, AVmU, CUpV, CVmU, Alist, Clist = exp_blocktriangular_pade9(A, C)
        elseif nAC > 0.25
            AUpV, AVmU, CUpV, CVmU, Alist, Clist = exp_blocktriangular_pade7(A, C)
        elseif nAC > 0.015
            AUpV, AVmU, CUpV, CVmU, Alist, Clist = exp_blocktriangular_pade5(A, C)
        else
            AUpV, AVmU, CUpV, CVmU, Alist, Clist = exp_blocktriangular_pade3(A, C)
        end
    else
        s  = ceil(Int, log2(nAC/5.4)) # power of 2 later reversed by squaring
        if s > 0
            factor = convert(T, 2^s)
            A ./= factor
            if !AeqC
                C ./= factor
            end
        end
        AUpV, AVmU, CUpV, CVmU, Alist, Clist = exp_blocktriangular_pade13(A, C)
    end
    AF = lu(AVmU)
    XA = ldiv!(AF, AUpV)
    XC = AeqC ? XA : ldiv!(lu!(CVmU), CUpV)
    XAlist = Vector{typeof(XA)}(undef, max(1, s))
    XClist = AeqC ? XAlist : Vector{typeof(XC)}(undef, max(1, s))
    XAlist[1] = copy(XA)
    if !AeqC
        XClist[1] = copy(XC)
    end
    if s > 0
        # recylce memory
        XA′ = AVmU
        XC′ = CVmU
        for t = 1:s
            XA′ = mul!(XA′, XA, XA)
            XA, XA′ = XA′, XA
            if AeqC
                XC = XA
            else
                XC′ = mul!(XC′, XC, XC)
                XC, XC′ = XC′, XC
            end
            if t < s
                XAlist[t+1] = copy(XA)
                if !AeqC
                    XClist[t+1] = copy(XC)
                end
            end
        end
    end

    # Undo the balancing
    XA = _unbalance!(XA, ilo1, ihi1, scale1) # modifies XA
    XC = AeqC ? XA : _unbalance!(XC, ilo2, ihi2, scale2) # modifies XC

    function expB_lazy(B)
        (n1, n2) == size(B) ||
            throw(DimensionMismatch("Size of $B is $(size(B)), expected ($n1, $n2)"))
        B = _balance2!(copy(B), ilo1, ihi1, scale1, ilo2, ihi2, scale2) # modifies B
        if (nAC <= 2.1)
            if nAC > 0.95
                BUpV, BVmU = exp_blocktriangular_pade9_lazy(B, Alist, Clist)
            elseif nAC > 0.25
                BUpV, BVmU = exp_blocktriangular_pade7_lazy(B, Alist, Clist)
            elseif nAC > 0.015
                BUpV, BVmU = exp_blocktriangular_pade5_lazy(B, Alist, Clist)
            else
                BUpV, BVmU = exp_blocktriangular_pade3_lazy(B, Alist, Clist)
            end
        else
            if s > 0
                factor = convert(T, 2^s)
                B ./= factor
            end
            BUpV, BVmU = exp_blocktriangular_pade13_lazy(B, Alist, Clist)
        end
        XB = ldiv!(AF, mul!(BUpV, BVmU, XClist[1], -1, 1))
        if s > 0
            # recylce memory
            XB′ = BVmU
            for t = 1:s
                XA = XAlist[t]
                XC = XClist[t]
                XB′ = mul!(mul!(XB′, XA, XB), XB, XC, true, true)
                XB, XB′ = XB′, XB
            end
        end
        XB = _unbalance2!(XB, ilo1, ihi1, scale1, ilo2, ihi2, scale2) # modifies XB
        return XB
    end

    return XA, XC, expB_lazy
end

# 13th order Pade approximation
function exp_blocktriangular_pade13(A, C)
    T = eltype(A)
    coeffs = T[64764752532480000., 32382376266240000., 7771770303897600., 1187353796428800.,
            129060195264000., 10559470521600., 670442572800., 33522128640., 1323241920.,
            40840800., 960960., 16380., 182., 1.]
    Alist = Vector{typeof(A)}(undef, 6)
    Clist = Vector{typeof(C)}(undef, 6)

    AeqC = A === C
    A0 = one(A)
    C0 = AeqC ? A0 : one(C)
    A2 = A * A
    C2 = AeqC ? A2 : C * C
    Alist[1] = A
    Clist[1] = C
    A4 = A2 * A2
    C4 = AeqC ? A4 : C2 * C2
    Alist[2] = copy(A2)
    Clist[2] = AeqC ? Alist[2] : copy(C2)
    A6 = A4 * A2
    C6 = AeqC ? A6 : C4 * C2
    Alist[3] = A4
    Clist[3] = Clist[2]

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
    Alist[4] = A6
    Clist[4] = copy(CV′)

    AW = A0
    AW′ = AV′
    AW′ .= coeffs[14] .* A6 .+ coeffs[12] .* A4 .+ coeffs[10] .* A2
    AW .= coeffs[8] .* A6 .+ coeffs[6] .* A4 .+ coeffs[4] .* A2 .+ coeffs[2] .* A0
    AW = mul!(AW, A6, AW′, true, true)

    if AeqC
        CW′ = AW′
        CW = AW
    else
        CW = C0
        CW′ = CV′
        CW′ .= coeffs[14] .* C6 .+ coeffs[12] .* C4 .+ coeffs[10] .* C2
        CW .= coeffs[8] .* C6 .+ coeffs[6] .* C4 .+ coeffs[4] .* C2 .+ coeffs[2] .* C0
        CW = mul!(CW, C6, CW′, true, true)
    end

    Alist[5] = Alist[4]
    Clist[5] = CW′

    AU = mul!(A2, A, AW)
    CU = AeqC ? AU : mul!(C2, C, CW)
    Alist[6] = A
    Clist[6] = copy(CW)

    A0 .= AU .+ AV
    if !AeqC
        C0 .= CU .+ CV
    end

    AV .-= AU
    if !AeqC
        CV .-= CU
    end

    return A0, AV, C0, CV, Alist, Clist
end

function exp_blocktriangular_pade13_lazy(B, Alist, Clist)
    T = eltype(B)
    coeffs = T[64764752532480000., 32382376266240000., 7771770303897600., 1187353796428800.,
            129060195264000., 10559470521600., 670442572800., 33522128640., 1323241920.,
            40840800., 960960., 16380., 182., 1.]

    B0 = zero(B)
    B2 = mul!(Alist[1] * B, B, Clist[1], true, true) # A * B + B * C
    B4 = mul!(Alist[2] * B2, B2, Clist[2], true, true) # A2 * B2 + B2 * C2
    B6 = mul!(Alist[3] * B2, B4, Clist[3], true, true) # A4 * B2 + B4 * C2

    BV′ = coeffs[13] .* B6 .+ coeffs[11] .* B4 .+ coeffs[9] .* B2
    BV = coeffs[7] .* B6 .+ coeffs[5] .* B4 .+ coeffs[3] .* B2 .+ coeffs[1] .* B0
    BV = mul!(mul!(BV, Alist[4], BV′, true, true), B6, Clist[4], true, true)

    BW = B0
    BW′ = BV′
    BW′ .= coeffs[14] .* B6 .+ coeffs[12] .* B4 .+ coeffs[10] .* B2
    BW .= coeffs[8] .* B6 .+ coeffs[6] .* B4 .+ coeffs[4] .* B2 .+ coeffs[2] .* B0
    BW = mul!(mul!(BW, Alist[5], BW′, true, true), B6, Clist[5], true, true)

    BU = mul!(mul!(B2, Alist[6], BW), B, Clist[6], true, true)

    B0 .= BU .+ BV
    BV .-= BU

    return B0, BV
end

# 9th order Pade approximation
function exp_blocktriangular_pade9(A, C)
    T = eltype(A)
    coeffs = T[17643225600., 8821612800., 2075673600., 302702400., 30270240., 2162160.,
                110880., 3960., 90., 1.]
    Alist = Vector{typeof(A)}(undef, 5)
    Clist = Vector{typeof(C)}(undef, 5)

    AeqC = A === C
    A0 = one(A)
    C0 = AeqC ? A0 : one(C)
    A2 = A * A
    C2 = AeqC ? A2 : C * C
    Alist[1] = A
    Clist[1] = C
    A4 = A2 * A2
    C4 = AeqC ? A4 : C2 * C2
    Alist[2] = copy(A2) # later overwritten
    Clist[2] = AeqC ? Alist[2] : copy(C2) # later overwritten
    A6 = A4 * A2
    C6 = AeqC ? A6 : C4 * C2
    Alist[3] = copy(A4) # later overwritten
    Clist[3] = Clist[2]

    AV = coeffs[7] .* A6 .+ coeffs[5] .* A4 .+ coeffs[3] .* A2 .+ coeffs[1] .* A0
    if AeqC
        CV = AV
    else
        CV = coeffs[7] .* C6 .+ coeffs[5] .* C4 .+ coeffs[3] .* C2 .+ coeffs[1] .* C0
    end

    AW = A0
    AW .= coeffs[8] .* A6 .+ coeffs[6] .* A4 .+ coeffs[4] .* A2 .+ coeffs[2] .* A0
    if AeqC
        CW = AW
    else
        CW = C0
        CW .= coeffs[8] .* C6 .+ coeffs[6] .* C4 .+ coeffs[4] .* C2 .+ coeffs[2] .* C0
    end

    A8 = mul!(A4, A6, A2)
    Alist[4] = A6
    Clist[4] = Clist[2]
    C8 = AeqC ? A8 : mul!(C4, C6, C2)

    AV .+= coeffs[9] .* A8
    if !AeqC
        CV .+= coeffs[9] .* C8
    end

    AW .+= coeffs[10] .* A8
    if !AeqC
        CW .+= coeffs[10] .* C8
    end

    AU = mul!(A2, A, AW)
    Alist[5] = A
    Clist[5] = copy(CW)
    CU = AeqC ? AU : mul!(C2, C, CW)

    A0 .= AU .+ AV
    AV .-= AU
    if !AeqC
        C0 .= CU .+ CV
        CV .-= CU
    end

    return A0, AV, C0, CV, Alist, Clist
end

function exp_blocktriangular_pade9_lazy(B, Alist, Clist)
    T = eltype(B)
    coeffs = T[17643225600., 8821612800., 2075673600., 302702400., 30270240., 2162160.,
                110880., 3960., 90., 1.]

    B0 = zero(B)
    B2 = mul!(Alist[1] * B, B, Clist[1], true, true) # A * B + B * C
    B4 = mul!(Alist[2] * B2, B2, Clist[2], true, true) # A2 * B2 + B2 * C2
    B6 = mul!(Alist[3] * B2, B4, Clist[3], true, true) # A4 * B2 + B4 * C2

    BV = coeffs[7] .* B6 .+ coeffs[5] .* B4 .+ coeffs[3] .* B2 .+ coeffs[1] .* B0
    BW = B0
    BW .= coeffs[8] .* B6 .+ coeffs[6] .* B4 .+ coeffs[4] .* B2 .+ coeffs[2] .* B0
    B8 = mul!(mul!(B4, Alist[4], B2), B6, Clist[4], true, true)

    BV .+= coeffs[9] .* B8
    BW .+= coeffs[10] .* B8
    BU = mul!(mul!(B2, Alist[5], BW), B, Clist[5], true, true)

    B0 .= BU .+ BV
    BV .-= BU

    return B0, BV
end

# 7th order Pade approximation
function exp_blocktriangular_pade7(A, C)
    T = eltype(A)
    coeffs = T[17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.]
    Alist = Vector{typeof(A)}(undef, 4)
    Clist = Vector{typeof(C)}(undef, 4)

    AeqC = A === C
    A0 = one(A)
    C0 = AeqC ? A0 : one(C)
    A2 = A * A
    C2 = AeqC ? A2 : C * C
    Alist[1] = A
    Clist[1] = C
    A4 = A2 * A2
    C4 = AeqC ? A4 : C2 * C2
    Alist[2] = copy(A2) # A2 is later overwritten
    Clist[2] = AeqC ? Alist[2] : copy(C2)
    A6 = A4 * A2
    C6 = AeqC ? A6 : C4 * C2
    Alist[3] = A4
    Clist[3] = Clist[2]

    AV = coeffs[7] .* A6 .+ coeffs[5] .* A4 .+ coeffs[3] .* A2 .+ coeffs[1] .* A0
    if AeqC
        CV = AV
    else
        CV = coeffs[7] .* C6 .+ coeffs[5] .* C4 .+ coeffs[3] .* C2 .+ coeffs[1] .* C0
    end

    AW = A0
    AW .= coeffs[8] .* A6 .+ coeffs[6] .* A4 .+ coeffs[4] .* A2 .+ coeffs[2] .* A0
    CW = C0
    if !AeqC
        CW .= coeffs[8] .* C6 .+ coeffs[6] .* C4 .+ coeffs[4] .* C2 .+ coeffs[2] .* C0
    end

    AU = mul!(A2, A, AW)
    Alist[4] = A
    Clist[4] = copy(CW) # CW === C0 is later overwritten
    CU = AeqC ? AU : mul!(C2, C, CW)

    A0 .= AU .+ AV
    AV .-= AU
    if !AeqC
        C0 .= CU .+ CV
        CV .-= CU
    end

    return A0, AV, C0, CV, Alist, Clist
end

function exp_blocktriangular_pade7_lazy(B, Alist, Clist)
    T = eltype(B)
    coeffs = T[17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.]

    B0 = zero(B)
    B2 = mul!(Alist[1] * B, B, Clist[1], true, true) # A * B + B * C
    B4 = mul!(Alist[2] * B2, B2, Clist[2], true, true) # A2 * B2 + B2 * C2
    B6 = mul!(Alist[3] * B2, B4, Clist[3], true, true) # A2 * B2 + B2 * C2

    BV = coeffs[7] .* B6 .+ coeffs[5] .* B4 .+ coeffs[3] .* B2 .+ coeffs[1] .* B0
    BW = B0
    BW .= coeffs[8] .* B6 .+ coeffs[6] .* B4 .+ coeffs[4] .* B2 .+ coeffs[2] .* B0

    BU = mul!(mul!(B2, Alist[4], BW), B, Clist[4], true, true)
    B0 .= BU .+ BV
    BV .-= BU

    return B0, BV
end

# 5th order Pade approximation
function exp_blocktriangular_pade5(A, C)
    T = eltype(A)
    coeffs = T[30240., 15120., 3360., 420., 30., 1.]
    Alist = Vector{typeof(A)}(undef, 3)
    Clist = Vector{typeof(C)}(undef, 3)

    AeqC = A === C
    A0 = one(A)
    C0 = AeqC ? A0 : one(C)
    A2 = A * A
    C2 = AeqC ? A2 : C * C
    Alist[1] = A
    Clist[1] = C
    A4 = A2 * A2
    C4 = AeqC ? A4 : C2 * C2
    Alist[2] = copy(A2)
    Clist[2] = AeqC ? Alist[2] : copy(C2)

    AV = coeffs[5] .* A4 .+ coeffs[3] .* A2 .+ coeffs[1] .* A0
    if AeqC
        CV = AV
    else
        CV = coeffs[5] .* C4 .+ coeffs[3] .* C2 .+ coeffs[1] .* C0
    end

    AW = A0
    AW .= coeffs[6] .* A4 .+ coeffs[4] .* A2 .+ coeffs[2] .* A0
    if AeqC
        CW = AW
    else
        CW = C0
        CW .= coeffs[6] .* C4 .+ coeffs[4] .* C2 .+ coeffs[2] .* C0
    end

    AU = mul!(A2, A, AW)
    Alist[3] = A
    Clist[3] = copy(CW)
    CU = AeqC ? AU : mul!(C2, C, CW)

    A0 .= AU .+ AV
    AV .-= AU
    if !AeqC
        C0 .= CU .+ CV
        CV .-= CU
    end

    return A0, AV, C0, CV, Alist, Clist
end

function exp_blocktriangular_pade5_lazy(B, Alist, Clist)
    T = eltype(B)
    coeffs = T[30240., 15120., 3360., 420., 30., 1.]

    B0 = zero(B)
    B2 = mul!(Alist[1] * B, B, Clist[1], true, true) # A * B + B * C
    B4 = mul!(Alist[2] * B2, B2, Clist[2], true, true) # A2 * B2 + B2 * C2

    BV = coeffs[5] .* B4 .+ coeffs[3] .* B2 .+ coeffs[1] .* B0
    BW = B0
    BW .= coeffs[6] .* B4 .+ coeffs[4] .* B2 .+ coeffs[2] .* B0

    BU = mul!(mul!(B2, Alist[3], BW), B, Clist[3], true, true)
    B0 .= BU .+ BV
    BV .-= BU
    return B0, BV
end

# 3rd order Pade approximation
function exp_blocktriangular_pade3(A, C)
    T = eltype(A)
    coeffs = T[120., 60., 12., 1.]
    Alist = Vector{typeof(A)}(undef, 2)
    Clist = Vector{typeof(C)}(undef, 2)

    AeqC = A === C
    A0 = one(A)
    C0 = AeqC ? A0 : one(C)
    A2 = A * A
    C2 = AeqC ? A2 : C * C
    Alist[1] = A
    Clist[1] = C

    AV = coeffs[3] .* A2 .+ coeffs[1] .* A0
    if AeqC
        CV = AV
    else
        CV = coeffs[3] .* C2 .+ coeffs[1] .* C0
    end

    AW = A0
    AW .= coeffs[4] .* A2 .+ coeffs[2] .* A0
    if AeqC
        CW = AW
    else
        CW = C0
        CW .= coeffs[4] .* C2 .+ coeffs[2] .* C0
    end

    AU = mul!(A2, A, AW)
    Alist[2] = A
    Clist[2] = copy(CW) # CW === C0 is later overwritten
    CU = AeqC ? AU : mul!(C2, C, CW)

    A0 .= AU .+ AV
    AV .-= AU
    if !AeqC
        C0 .= CU .+ CV
        CV .-= CU
    end

    return A0, AV, C0, CV, Alist, Clist
end

function exp_blocktriangular_pade3_lazy(B, Alist, Clist)
    T = eltype(B)
    coeffs = T[120., 60., 12., 1.]

    B0 = zero(B)
    B2 = mul!(Alist[1] * B, B, Clist[1], true, true) # A * B + B * C
    BV = coeffs[3] .* B2 .+ coeffs[1] .* B0
    BW = B0
    BW .= coeffs[4] .* B2 .+ coeffs[2] .* B0
    BU = mul!(mul!(B2, Alist[2], BW), B, Clist[2], true, true)
    B0 .= BU .+ BV
    BV .-= BU
    return B0, BV
end
