@testset "exp_blocktriangular (generic)" begin
    @testset "with eltype $T" for T in (Float64, ComplexF64)
        for D1 in (2, 4, 10, 50, 100, 200)
            for D2 in (2, 4, 10, 50, 100, 200)
                A = randn(T, D1, D1)
                B = randn(T, D1, D2)
                C = randn(T, D2, D2)

                XA, XB, XC = CMPSKit.exp_blocktriangular(A, B, C);
                XA_, XC_, expB = CMPSKit.exp_blocktriangular_lazy(A, C); XB_ = expB(B);
                @test all((XA ≈ XA_, XB ≈ XB_, XC ≈ XC_))
                @test exp([A B; zero(B') C]) ≈ [XA XB; zero(XB') XC]
            end
        end
    end
end

@testset "exp_blocktriangular (square)" begin
    @testset "with eltype $T" for T in (Float64, ComplexF64)
        for D in (2, 4, 10, 50, 100, 200)
            A = randn(T, D, D)
            B = randn(T, D, D)

            while norm(A, 1) > 0.015
                A ./= 2
                XA, XB = CMPSKit.exp_blocktriangular(A, B);
                XA_, expB = CMPSKit.exp_blocktriangular_lazy(A); XB_ = expB(B);
                @test all((XA ≈ XA_, XB ≈ XB_))
                @test exp([A B; zero(B') A]) ≈ [XA XB; zero(XB') XA]
            end

        end
    end
end
