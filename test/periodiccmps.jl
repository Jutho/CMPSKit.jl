Dlist = (2,5,7,12)
@testset "PeriodicCMPS: environments and gauging with bond dimension $D" for D in Dlist
    for T in (Float64, ComplexF64)
        Q = FourierSeries([exp(-4*(j>>1))*randn(T, (D,D)) for j=1:5])
        R = FourierSeries([exp(-4*(j>>1))*randn(T, (D,D)) for j=1:3])
        Ψ = InfiniteCMPS(Q, R)

        ρL, λL, infoL = leftenv(Ψ; Kmax = 20)
        TL = LeftTransfer(Ψ)
        @test norm(-∂(ρL) + TL(ρL) - 2*λL*ρL) <= 10*infoL.normres[1]
        @test ρL == ρL'

        ρR, λR, infoR = rightenv(Ψ; Kmax = 20)
        TR = RightTransfer(Ψ)
        @test norm(∂(ρR) + TR(ρR) - 2*λR*ρR) <= 10*infoR.normres[1]
        @test λL ≈ λR
        @test ρR == ρR'

        Zf = localdot(ρL, ρR)
        Z = Zf(0)
        @test Zf ≈ FourierSeries([Z])
        @test imag(Z) <= sqrt(infoL.normres[1]*infoR.normres[1])

        Ψn = copy(Ψ)
        ρL2, = leftenv!(Ψn; Kmax = 20)
        @test ρL2 ≈ ρL
        Qn = Ψn.Q
        Rn, = Ψn.Rs
        @test norm(-∂(ρL)+Qn'*ρL2 + ρL*Qn + Rn'*ρL*Rn) <= 1e-9

        Ψn = copy(Ψ)
        ρR2, = rightenv!(Ψn; Kmax = 20)
        @test ρR2 ≈ ρR
        Qn = Ψn.Q
        Rn, = Ψn.Rs
        @test norm(∂(ρR) + Qn*ρR + ρR*Qn' + Rn*ρR*Rn') <= 1e-9

        Ψn = copy(Ψ)
        ρL2, ρR2 = environments!(Ψn; Kmax = 20)
        @test ρL2 ≈ ρL/sqrt(Z)
        @test ρR2 ≈ ρR/sqrt(Z)
        Qn = Ψn.Q
        Rn, = Ψn.Rs
        @test norm(-∂(ρL) + LeftTransfer(Ψn)(ρL)) <= 1e-9
        @test norm(∂(ρR) + RightTransfer(Ψn)(ρR)) <= 1e-9
    end
end


@testset "PeriodicCMS: energy environments for bond dimension $D" for D in Dlist
    v = fit(x->-3+sin(x), FourierSeries; Kmax = 1)
    H = ∫(∂ψ'*∂ψ + v*ψ'*ψ + 0.3*(ψ*ψ + ψ'*ψ') + 2.5*(ψ')^2*ψ^2, (-Inf,+Inf))
    for T in (Float64, ComplexF64)
        Q = FourierSeries([exp(-4*(j>>1))*randn(T, (D,D)) for j=1:5])
        R = FourierSeries([exp(-4*(j>>1))*randn(T, (D,D)) for j=1:3])
        Ψ = InfiniteCMPS(Q, R)


        ρL, ρR = environments!(Ψ; Kmax = 20)
        HL, eL, hL = leftenv(H, (Ψ,ρL,ρR); Kmax = 30)
        HR, eR, hR = rightenv(H, (Ψ,ρL,ρR); Kmax = 30)

        @test eL ≈ eR
        @test norm(∂(HL) - LeftTransfer(Ψ)(HL) - hL) <= 1e-9*norm(HL)
        @test norm(∂(HR) + RightTransfer(Ψ)(HR) + hR) <= 1e-9*norm(HR)
        @test abs(dot(HL,ρR)) <= 1e-9*norm(HL)
        @test abs(dot(ρL,HR)) <= 1e-9*norm(HR)
    end
end

@testset "PeriodicCMS: local gradients with bond dimension $D" for D in Dlist
    for T in (Float64, ComplexF64)
        Q = FourierSeries([exp(-4*(j>>1))*randn(T, (D,D)) for j=1:5])
        R1 = FourierSeries([exp(-4*(j>>1))*randn(T, (D,D)) for j=1:3])
        R2 = FourierSeries([exp(-4*(j>>1))*randn(T, (D,D)) for j=1:3])
        Rs = (R1, R2)
        Ψ = InfiniteCMPS(Q, Rs)
        ρL, ρR = environments!(Ψ; Kmax = 20)

        QR1 = Q*R1 - R1*Q + ∂(R1)
        QR2 = Q*R2 - R2*Q + ∂(R2)

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

        @test CMPSKit.localgradientRs(ψ[1], Q, Rs, ρL, ρR) == zero(Q)
        @test CMPSKit.localgradientRs(ψ[1]^2, Q, Rs, ρL, ρR) == zero(Q)
        @test CMPSKit.localgradientRs(∂ψ[1], Q, Rs, ρL, ρR) == zero(Q)
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
