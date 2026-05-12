struct InfiniteCMPSExcitationSpace{T,N,S}
    momentum::S
    QL::T
    RLs::NTuple{N,T}
    QR::T
    RRs::NTuple{N,T}
    ρR::T
    ρL::T
    C::T
    topological::Bool
    function InfiniteCMPSExcitationSpace(p::S, QL::T, RLs::NTuple{N,T}, QR::T,
                                         RRs::NTuple{N,T}, ρR::T, ρL::T, C::T,
                                         topo::Bool) where {S,T,N}
        C = normalize!(C)
        @assert norm(QL + QL' + sum(adjoint.(RLs) .* RLs)) < defaulttol(C)
        @assert norm(QR + QR' + sum(RRs .* adjoint.(RRs))) < defaulttol(C)
        if !topo
            @assert QL * C ≈ C * QR
            @assert all(RLs .* Ref(C) .≈ Ref(C) .* RRs)
            @assert ρR ≈ C * C'
            @assert ρL ≈ C' * C
        end
        return new{T,N,S}(p, QL, RLs, QR, RRs, ρR, ρL, C, topo)
    end
end
struct InfiniteCMPSExcitation{T,N,S}
    state::NTuple{N,T}
    space::InfiniteCMPSExcitationSpace{T,N,S}
end

function InfiniteCMPSExcitationSpace(momentum, ΨL::State, ΨR::State=ΨL;
                                     kwargs...) where {State<:UniformCMPS}
    topo = !(ΨL === ΨR)

    ΨL, λL, CL = leftgauge(ΨL; kwargs...)
    ΨR, λR, CR = rightgauge(ΨR; kwargs...)

    C = CL * CR # even if topo, the following initial guesses for ρL and ρR are probably good
    ρR, = rightenv(ΨL, C * C'; kwargs...)
    ρR = rmul!(ρR, 1 / tr(ρR)[])
    ρL, = leftenv(ΨR, C' * C; kwargs...)
    ρL = rmul!(ρL, 1 / tr(ρL)[])

    QL, RLs = ΨL.Q, ΨL.Rs
    QR, RRs = ΨR.Q, ΨR.Rs

    p = convert(real(scalartype(QL)), momentum)
    return InfiniteCMPSExcitationSpace(p, QL, RLs, QR, RRs, ρR, ρL, C, topo)
end

function InfiniteCMPSExcitation(momentum, V, Ws, ΨL::State,
                                ΨR::State) where {State<:UniformCMPS}
    # TODO
end

const UniformCMPSExcitationSpace{A<:AbstractMatrix,N} = InfiniteCMPSExcitationSpace{<:Constant{A},
                                                                                    N}

function excitation_operator(Ĥ::LocalHamiltonian, space::UniformCMPSExcitationSpace;
                             kwargs...)
    ΨL = InfiniteCMPS(space.QL, space.RLs; gauge=:l)
    ρR = space.ρR
    ΨR = InfiniteCMPS(space.QR, space.RRs; gauge=:r)
    ρL = space.ρL

    HL, eL, _ = leftenv(Ĥ, (ΨL, one(ρR), ρR); kwargs...)
    HR, eR, _ = rightenv(Ĥ, (ΨR, ρL, one(ρL)); kwargs...)
    if !(abs(eL - eR) < defaulttol(ΨL))
        error("left and right ground state in excitation space have different expectation value for the given operator: $eL and $eR")
    end
    e = (eL + eR) / 2

    Heff = let C = space.C, QL = space.QL, RLs = space.RLs, QR = space.QR, RRs = space.RRs,
        p = space.momentum, topo = space.topological,
        HL = HL, HR = HR, TLR = RightTransfer(ΨL, ΨR), TRL = LeftTransfer(ΨR, ΨL)

        function (Xs)
            if !iszero(p)
                @assert scalartype(first(Xs)) <: Complex
            end
            Ws = Xs
            V = zero(first(Ws))
            for (W, RL) in zip(Ws, RLs)
                mul!(V, RL', W, -1, 1)
            end
            if !iszero(p)
                ∂Ws = im .* p .* Ws
            else
                ∂Ws = zero.(Ws)
            end

            HWs = zero.(Ws)
            HV = zero(V)

            # local terms: ket and bra excitations acting on local Hamiltonian terms
            for (coeff, op) in zip(coefficients(Ĥ.h), operators(Ĥ.h))
                op_ket = _ketfactor_tangent(op, QL, RLs, V, Ws, ∂Ws, QR, RRs)
                HWs = axpy!.(Ref(coeff),
                             _brafactor_cotangentRs(op, QL, RLs, QR, RRs)(op_ket), HWs)
                if op isa ContainsDifferentiatedCreation
                    HV = axpy!(coeff, _brafactor_cotangentQ(op, QL, RLs, QR, RRs)(op_ket),
                               HV)
                    if !iszero(p)
                        HWs = axpy!.(Ref(-im * p * coeff),
                                     _brafactor_cotangent∂Rs(op, QL, RLs, QR, RRs)(op_ket),
                                     HWs)
                    end
                end
            end

            # ket and bra excitation on same position, hamiltonian left or right thereof
            for (HW, W) in zip(HWs, Ws)
                mul!(HW, HL, W, 1, 1)
                mul!(HW, W, HR, 1, 1)
            end

            # non-local contributions
            # gL: ket excitation on Hamiltonian + (Hamiltonian -> transfer -> ket excitation)
            gL = HL * V
            Xtemp = zero(V)
            for (RL, W) in zip(RLs, Ws)
                Xtemp = mul!(Xtemp, HL, W)
                gL = mul!(gL, RL', Xtemp, 1, 1)
            end
            for (coeff, op) in zip(coefficients(Ĥ.h), operators(Ĥ.h))
                ketf = _ketfactor_tangent(op, QL, RLs, V, Ws, ∂Ws, QR, RRs)
                braf = _brafactor(op, QL, RLs)
                gL = mul!(gL, braf', ketf, coeff, 1)
            end

            # gR: ket excitation to the right
            gR = copy(V)
            for (W, RR) in zip(Ws, RRs)
                gR = mul!(gR, W, RR', 1, 1)
            end

            # GL: gL -> transfer - i*p
            # GR: transfer + i*p -> gR
            if topo
                GR, = linsolve(-gR, zero(gR); kwargs...) do x
                    y = TLR(x)
                    if !iszero(p)
                        y = axpy!(im * p, x, y)
                    end
                    return y
                end

                GL, = linsolve(-gL, zero(gL); kwargs...) do x
                    y = TRL(x)
                    if !iszero(p)
                        y = axpy!(-im * p, x, y)
                    end
                    return y
                end
            else
                gR = axpy!(-tr(C' * gR)[], C, gR) # tr should be zero by construction of V and W
                GR, = linsolve(-gR, zero(gR); kwargs...) do x
                    y = TLR(x)
                    if !iszero(p)
                        y = axpy!(im * p, x, y)
                    end
                    return y = axpy!(tr(C' * x)[], C, y)
                end
                GR = axpy!(-tr(C' * GR)[], C, GR) # tr should be zero anyway

                gL = axpy!(-tr(gL * C')[], C, gL) # tr should be zero for good ground state approximation with normgrad ≈ 0
                GL, = linsolve(-gL, zero(gL); kwargs...) do x
                    y = TRL(x)
                    if !iszero(p)
                        y = axpy!(-im * p, x, y)
                    end
                    return y = axpy!(tr(x * C')[], C, y)
                end
                GL = axpy!(-tr(GL * C')[], C, GL)
            end

            # Contributions of GL
            HV = mul!(axpy!(1, GL, HV), HL, GR, 1, 1)
            for (HW, RL, RR) in zip(HWs, RLs, RRs)
                Xtemp = mul!(Xtemp, RL, GR)
                HW = mul!(mul!(HW, GL, RR, 1, 1), HL, Xtemp, 1, 1)
            end

            # Final contribution: bra excitation on Hamiltonian -> GR
            for (coeff, op) in zip(coefficients(Ĥ.h), operators(Ĥ.h))
                op_ket = _ketfactor(op, QL, RLs)
                Xtemp = mul!(Xtemp, op_ket, GR)
                HWs = axpy!.(Ref(coeff),
                             _brafactor_cotangentRs(op, QL, RLs, QR, RRs)(Xtemp), HWs)
                if op isa ContainsDifferentiatedCreation
                    HV = axpy!(coeff, _brafactor_cotangentQ(op, QL, RLs, QR, RRs)(Xtemp),
                               HV)
                    if !iszero(p)
                        HWs = axpy!.(Ref(-im * p * coeff),
                                     _brafactor_cotangent∂Rs(op, QL, RLs, QR, RRs)(Xtemp),
                                     HWs)
                    end
                end
            end

            Ys = mul!.(HWs, RLs, Ref(HV), -1, 1)
            return Ys
        end
    end
    return Heff
end

function excitation_metric(space::UniformCMPSExcitationSpace; kwargs...)
    return identity
end
