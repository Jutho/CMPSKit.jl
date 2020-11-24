Dlist = (2,4,12,29,71)

constant1(x) = Constant(x)
constant2(x) = FourierSeries([x])

@testset "UniformCMPS: environments and gauging with bond dimension $D" for D in Dlist
    for constant in (constant1, constant2)
        for T in (Float64, ComplexF64)
            Q = constant(randn(T, (D,D)))
            R1 = constant(randn(T, (D,D)))
            R2 = constant(randn(T, (D,D)))
            Ψ = InfiniteCMPS(Q, (R1,R2))
            ρL, λL, infoL = leftenv(Ψ)
            TL = LeftTransfer(Ψ)
            @test norm(TL(ρL) - 2*λL*ρL) <= 10*infoL.normres[1]
            @test ρL == ρL'

            ρR, λR, infoR = rightenv(Ψ)
            TR = RightTransfer(Ψ)
            @test norm(TR(ρR) - 2*λR*ρR) <= 10*infoR.normres[1]
            @test λL ≈ λR
            @test ρR == ρR'

            Z = dot(ρL, ρR)
            @test imag(Z) <= sqrt(infoL.normres[1]*infoR.normres[1])

            Ψn = copy(Ψ)
            ρL2, = leftenv!(Ψn)
            @test ρL2 ≈ ρL
            Qn = Ψn.Q
            R1n, R2n = Ψn.Rs
            @test norm(Qn'*ρL + ρL*Qn + R1n'*ρL*R1n + R2n'*ρL*R2n) <= 1e-9

            Ψn = copy(Ψ)
            ρR2, = rightenv!(Ψn)
            @test ρR2 ≈ ρR
            Qn = Ψn.Q
            R1n, R2n = Ψn.Rs
            @test norm(Qn*ρR + ρR*Qn' + R1n*ρR*R1n' + R2n*ρR*R2n') <= 1e-9

            Ψn = copy(Ψ)
            ρL2, ρR2 = environments!(Ψn)
            @test ρL2 ≈ ρL/sqrt(Z)
            @test ρR2 ≈ ρR/sqrt(Z)
            @test norm(LeftTransfer(Ψn)(ρL2)) <= 1e-9
            @test norm(RightTransfer(Ψn)(ρR2)) <= 1e-9

            if constant == constant1 # actual UniformCMPS
                ΨL,λ,CL,info = leftgauge(Ψ)
                @test λ ≈ λL
                @test CL'*CL ≈ ρL/tr(ρL[])
                QL = ΨL.Q
                R1L, R2L = ΨL.Rs
                @test CL*Qn ≈ QL*CL
                @test CL*R1 ≈ R1L*CL
                @test CL*R2 ≈ R2L*CL
                @test norm(QL + QL' + R1L'*R1L + R2L'*R2L) <= 1e-9

                ΨR,λ,CR,info = rightgauge(Ψ; tol = 1e-12)
                @test λ ≈ λR
                @test CR*CR' ≈ ρR/tr(ρR[])
                QR = ΨR.Q
                R1R, R2R = ΨR.Rs
                @test Qn*CR ≈ CR*QR
                @test R1*CR ≈ CR*R1R
                @test R2*CR ≈ CR*R2R
                @test norm(QR + QR' + R1R*R1R' + R2R*R2R') <= 1e-9
            end
        end
    end
end

@testset "UniformCMPS: energy environments with bond dimension $D" for D in Dlist
    H = ∫(∂ψ'*∂ψ - 0.5*ψ'*ψ + 0.3*(ψ*ψ + ψ'*ψ') + 2.5*(ψ')^2*ψ^2, (-Inf,+Inf))
    for constant in (constant1, constant2)
        for T in (Float64, ComplexF64)
            Q = constant(randn(T, (D,D)))
            R = constant(randn(T, (D,D)))
            Ψ = InfiniteCMPS(Q, R)

            ρL, ρR = environments!(Ψ)
            HL, EL, eL, hL = leftenv(H, (Ψ,ρL,ρR))
            HR, ER, eR, hR = rightenv(H, (Ψ,ρL,ρR))

            @test eL ≈ eR
            @test eL(0) ≈ EL
            @test eR(0) ≈ ER
            @test norm(LeftTransfer(Ψ)(HL) + hL) <= 1e-9*norm(HL)
            @test norm(RightTransfer(Ψ)(HR) + hR) <= 1e-9*norm(HR)
            @test abs(dot(HL,ρR)) <= 1e-9*norm(HL)
            @test abs(dot(ρL,HR)) <= 1e-9*norm(HR)
        end
    end
end

@testset "UniformCMPS: local gradients with bond dimension D = $D" for D in Dlist
    for constant in (constant1, constant2)
        for T in (Float64, ComplexF64)
            Q = constant(randn(T, (D,D)))
            R1 = constant(randn(T, (D,D)))
            R2 = constant(randn(T, (D,D)))
            ρL = constant(randn(T, (D,D)))
            ρL = (ρL + ρL')/2 # does not need to be the actual ρL for this test
            ρR = constant(randn(T, (D,D)))
            ρR = (ρR + ρR')/2 # does not need to be the actual ρL for this test
            Rs = (R1,R2)

            QR1 = Q*R1 - R1*Q
            QR2 = Q*R2 - R2*Q

            @test CMPSKit.localgradientQ(ψ[1], Q, Rs, ρL, ρR) == zero(Q)
            @test CMPSKit.localgradientQ(ψ[1]^2, Q, Rs, ρL, ρR) == zero(Q)
            @test CMPSKit.localgradientQ(∂ψ[1], Q, Rs, ρL, ρR) == zero(Q)
            @test CMPSKit.localgradientQ(ψ[2]', Q, Rs, ρL, ρR) == zero(Q)
            @test CMPSKit.localgradientQ((ψ[2]')^2, Q, Rs, ρL, ρR) == zero(Q)
            @test CMPSKit.localgradientQ(∂(ψ[2]'), Q, Rs, ρL, ρR) ≈ ρL*ρR*R2' - R2'*ρL*ρR
            @test CMPSKit.localgradientQ((∂ψ[1])'*∂ψ[2], Q, Rs, ρL, ρR) ≈
                    ρL*QR2*ρR*R1' - R1'*ρL*QR2*ρR
            @test CMPSKit.localgradientQ(ψ[1]'*ψ[2], Q, Rs, ρL, ρR) == zero(Q)
            @test CMPSKit.localgradientQ((ψ[1]')^2*ψ[1]^2, Q, Rs, ρL, ρR) == zero(Q)

            @test CMPSKit.localgradientRs(ψ[1], Q, Rs, ρL, ρR) == (zero(Q), zero(Q))
            @test CMPSKit.localgradientRs(ψ[1]^2, Q, Rs, ρL, ρR) == (zero(Q), zero(Q))
            @test CMPSKit.localgradientRs(∂ψ[1], Q, Rs, ρL, ρR) == (zero(Q), zero(Q))
            @test all(isapprox.(CMPSKit.localgradientRs(ψ[2]', Q, Rs, ρL, ρR),
                                (zero(Q), ρL*ρR)))
            @test all(isapprox.(CMPSKit.localgradientRs((ψ[2]')^2, Q, Rs, ρL, ρR),
                                (zero(Q), R2'*ρL*ρR + ρL*ρR*R2')))
            @test all(isapprox.(CMPSKit.localgradientRs(∂(ψ[2]'), Q, Rs, ρL, ρR),
                                (zero(Q), Q'*ρL*ρR - ρL*ρR*Q' - ∂(ρL*ρR))))
            @test all(isapprox.(CMPSKit.localgradientRs(∂(ψ[2]'), Q, Rs, ρL, ρR),
                                (zero(Q), Q'*ρL*ρR - ρL*ρR*Q' - ∂(ρL*ρR))))
            @test all(isapprox.(CMPSKit.localgradientRs((∂ψ[1])'*∂ψ[2], Q, Rs, ρL, ρR),
                                (Q'*ρL*QR2*ρR - ρL*QR2*ρR*Q' - ∂(ρL*QR2*ρR), zero(Q))))
            @test all(isapprox.(CMPSKit.localgradientRs(ψ[1]'*ψ[2], Q, Rs, ρL, ρR),
                                (ρL*R2*ρR, zero(Q))))
            @test all(isapprox.(CMPSKit.localgradientRs((ψ[1]')^2*ψ[1]^2, Q, Rs, ρL, ρR),
                                (R1'*ρL*R1*R1*ρR + ρL*R1*R1*ρR*R1', zero(Q))))

        end
    end
end

@testset "UniformCMPS: global gradients with bond dimension D = $D" for D in Dlist
    α = rand()
    β = rand()
    γ = rand()
    H = ∫(∂ψ'*∂ψ + α*ψ'*ψ + β*(ψ*ψ + ψ'*ψ') + γ*(ψ')^2*ψ^2, (-Inf,+Inf))
    for T in (Float64, ComplexF64)
        Q = Constant(randn(T, (D,D)))
        R = Constant(randn(T, (D,D)))

        ΨL, = leftgauge!(InfiniteCMPS(Q, R))
        QL = ΨL.Q
        RL = ΨL.Rs[1]
        ρR, = rightenv(ΨL)
        ρL = one(ρR)
        HL, = leftenv(H, (ΨL, ρL, ρR))
        HR, = rightenv(H, (ΨL, ρL, ρR))

        gradQ, gradRs = gradient(H, (ΨL, ρL, ρR), HL, HR)

        QRL = QL*RL-RL*QL
        gradR = gradRs[1]

        @test gradQ ≈ ρL*QRL*ρR*RL' - RL'*ρL*QRL*ρR + HL*ρR + ρL*HR

        @test gradR ≈ QL'*ρL*QRL*ρR - ρL*QRL*ρR*QL' + α*ρL*RL*ρR +
                        β*RL'*ρL*ρR + β*ρL*ρR*RL' + γ*RL'*ρL*RL*RL*ρR + γ*ρL*RL*RL*ρR*RL' +
                        HL*RL*ρR + ρL*RL*HR
    end
end
