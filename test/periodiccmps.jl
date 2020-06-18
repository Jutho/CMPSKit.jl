Dlist = (2,5,7,12)
@testset "PeriodicCMPS: environments and gauging with bond dimension $D" for D in Dlist
    for T in (Float64, ComplexF64)
        Q = FourierSeries([exp(-4*(j>>1))*randn(T, (D,D)) for j=1:5])
        R = FourierSeries([exp(-4*(j>>1))*randn(T, (D,D)) for j=1:3])
        Ψ = InfiniteCMPS(Q, R)
        ρL, λL, infoL = leftenv(Ψ; Kmax = 20, krylovdim = 50, tol = 1e-12)
        @test norm(-∂(ρL) + Q'*ρL + ρL*Q + R'*ρL*R - 2*λL*ρL) <= 10*infoL.normres[1]
        @test ρL == ρL'

        ρR, λR, infoR = rightenv(Ψ; Kmax = 20, krylovdim = 50, tol = 1e-12)
        @test norm(∂(ρR) + Q*ρR + ρR*Q' + R*ρR*R' - 2*λR*ρR) <= 10*infoR.normres[1]
        @test λL ≈ λR
        @test ρR == ρR'

        Zf = localdot(ρL, ρR)
        Z = Zf(0)
        @test Zf ≈ FourierSeries([Z])
        @test imag(Z) <= sqrt(infoL.normres[1]*infoR.normres[1])

        Ψn = copy(Ψ)
        ρL2, = leftenv!(Ψn; Kmax = 20, krylovdim = 50, tol = 1e-12)
        @test ρL2 ≈ ρL
        Qn = Ψn.Q
        Rn, = Ψn.Rs
        @test norm(-∂(ρL)+Qn'*ρL2 + ρL*Qn + Rn'*ρL*Rn) <= 1e-9

        Ψn = copy(Ψ)
        ρR2, = rightenv!(Ψn; Kmax = 20, krylovdim = 50, tol = 1e-12)
        @test ρR2 ≈ ρR
        Qn = Ψn.Q
        Rn, = Ψn.Rs
        @test norm(∂(ρR) + Qn*ρR + ρR*Qn' + Rn*ρR*Rn') <= 1e-9

        Ψn = copy(Ψ)
        ρL2, ρR2 = environments!(Ψn; Kmax = 20, krylovdim = 50, tol = 1e-12)
        @test ρL2 ≈ ρL/sqrt(Z)
        @test ρR2 ≈ ρR/sqrt(Z)
        Qn = Ψn.Q
        Rn, = Ψn.Rs
        @test norm(-∂(ρL)+Qn'*ρL + ρL*Qn + Rn'*ρL*Rn) <= 1e-9
        @test norm(∂(ρR) + Qn*ρR + ρR*Qn' + Rn*ρR*Rn') <= 1e-9
    end
end
