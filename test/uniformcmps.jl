Dlist = (2,4,12,29,71)

@testset "UniformCMPS: environments and gauging with bond dimension $D" for D in Dlist
    for T in (Float64, ComplexF64)
        Q = Constant(randn(T, (D,D)))
        R1 = Constant(randn(T, (D,D)))
        R2 = Constant(randn(T, (D,D)))
        Ψ = InfiniteCMPS(Q, (R1,R2))
        ρL, λL, infoL = leftenv(Ψ; krylovdim = min(D^2,50))
        @test norm(Q'*ρL + ρL*Q + R1'*ρL*R1 + R2'*ρL*R2 - 2*λL*ρL) <= infoL.normres[1]
        @test ρL == ρL'

        ρR, λR, infoR = rightenv(Ψ; krylovdim = min(D^2,50))
        @test norm(Q*ρR + ρR*Q' + R1*ρR*R1' + R2*ρR*R2' - 2*λR*ρR) <= infoR.normres[1]
        @test λL ≈ λR
        @test ρR == ρR'

        Z = dot(ρL, ρR)
        @test imag(Z) <= sqrt(infoL.normres[1]*infoR.normres[1])

        Ψn = copy(Ψ)
        ρL2, ρR2, = environments!(Ψn; krylovdim = min(D^2,50))
        @test ρL2 ≈ ρL/sqrt(Z)
        @test ρR2 ≈ ρR/sqrt(Z)
        Qn = Ψn.Q
        R1n, R2n = Ψn.Rs
        @test norm(Qn'*ρL + ρL*Qn + R1n'*ρL*R1n + R2n'*ρL*R2n) <= eps(real(T))^(2/3)
        @test norm(Qn*ρR + ρR*Qn' + R1n*ρR*R1n' + R2n*ρR*R2n') <= eps(real(T))^(2/3)

        ΨL,λ,CL,info = leftgauge(Ψ; tol = 1e-13)
        @test λ ≈ λL
        @test CL'*CL ≈ ρL/tr(ρL[])
        QL = ΨL.Q
        R1L, R2L = ΨL.Rs
        @test CL*Qn ≈ QL*CL
        @test CL*R1 ≈ R1L*CL
        @test CL*R2 ≈ R2L*CL
        @test norm(QL + QL' + R1L'*R1L + R2L'*R2L) <= 1e-10

        ΨR,λ,CR,info = rightgauge(Ψ; tol = 1e-13)
        @test λ ≈ λR
        @test CR*CR' ≈ ρR/tr(ρR[])
        QR = ΨR.Q
        R1R, R2R = ΨR.Rs
        @test Qn*CR ≈ CR*QR
        @test R1*CR ≈ CR*R1R
        @test R2*CR ≈ CR*R2R
        @test norm(QR + QR' + R1R*R1R' + R2R*R2R') <= 1e-10
    end
end
