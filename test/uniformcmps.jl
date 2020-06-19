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
            ρL, λL, infoL = leftenv(Ψ; krylovdim = 50, tol = 1e-12)
            TL = LeftTransfer(Ψ)
            @test norm(TL(ρL) - 2*λL*ρL) <= 10*infoL.normres[1]
            @test ρL == ρL'

            ρR, λR, infoR = rightenv(Ψ; krylovdim = 50, tol = 1e-12)
            TR = RightTransfer(Ψ)
            @test norm(TR(ρR) - 2*λR*ρR) <= 10*infoR.normres[1]
            @test λL ≈ λR
            @test ρR == ρR'

            Z = dot(ρL, ρR)
            @test imag(Z) <= sqrt(infoL.normres[1]*infoR.normres[1])

            Ψn = copy(Ψ)
            ρL2, = leftenv!(Ψn; krylovdim = min(D^2,50), tol = 1e-12)
            @test ρL2 ≈ ρL
            Qn = Ψn.Q
            R1n, R2n = Ψn.Rs
            @test norm(Qn'*ρL + ρL*Qn + R1n'*ρL*R1n + R2n'*ρL*R2n) <= 1e-9

            Ψn = copy(Ψ)
            ρR2, = rightenv!(Ψn; krylovdim = 50, tol = 1e-12)
            @test ρR2 ≈ ρR
            Qn = Ψn.Q
            R1n, R2n = Ψn.Rs
            @test norm(Qn*ρR + ρR*Qn' + R1n*ρR*R1n' + R2n*ρR*R2n') <= 1e-9

            Ψn = copy(Ψ)
            ρL2, ρR2 = environments!(Ψn; krylovdim = 50, tol = 1e-12)
            @test ρL2 ≈ ρL/sqrt(Z)
            @test ρR2 ≈ ρR/sqrt(Z)
            @test norm(LeftTransfer(Ψn)(ρL2)) <= 1e-9
            @test norm(RightTransfer(Ψn)(ρR2)) <= 1e-9

            if constant == constant1 # actual UniformCMPS
                ΨL,λ,CL,info = leftgauge(Ψ; tol = 1e-12)
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

@testset "UniformCMPS: energy environments for bond dimension $D" for D in Dlist
    H = ∫(∂ψ'*∂ψ - 0.5*ψ'*ψ + 0.3*(ψ*ψ + ψ'*ψ') + 2.5*(ψ')^2*ψ^2, (-Inf,+Inf))
    for constant in (constant1, constant2)
        for T in (Float64, ComplexF64)
            Q = constant(randn(T, (D,D)))
            R = constant(randn(T, (D,D)))
            Ψ = InfiniteCMPS(Q, R)

            ρL, ρR = environments!(Ψ; krylovdim = 100, tol = 1e-12)
            HL, eL, hL = leftenv(H, (Ψ,ρL,ρR); krylovdim = 100, tol = 1e-12)
            HR, eR, hR = rightenv(H, (Ψ,ρL,ρR); krylovdim = 100, tol = 1e-12)

            @test eL ≈ eR
            @test norm(LeftTransfer(Ψ)(HL) + hL) <= 1e-9
            @test norm(RightTransfer(Ψ)(HR) + hR) <= 1e-9
            @test abs(dot(HL,ρR)) <= 1e-9
            @test abs(dot(ρL,HR)) <= 1e-9
        end
    end
end
