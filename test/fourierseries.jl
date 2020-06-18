@testset "FourierSeries" begin
    p = 3*rand()
    F1 = FourierSeries([randn(Float64, (2,2)) for k = 1:5], p)
    F2 = FourierSeries([randn(ComplexF64, (2,2)) for k = 1:3], p)

    x = range(-p/2, 3*p/2, length = 201)

    α, β = randn(), randn(ComplexF64)
    A = randn(2,2)

    @test (F1 + F2).(x) ≈ F1.(x) .+ F2.(x)
    @test (F1 - F2).(x) ≈ F1.(x) .- F2.(x)
    @test (F2 - F1).(x) ≈ F2.(x) .- F1.(x)
    @test (α*F1).(x) ≈ α .* F1.(x)
    @test (β*F1).(x) ≈ β .* F1.(x)
    @test (β*F1 + α*F2).(x) ≈ β .* F1.(x) .+ α .* F2.(x)
    @test (β*F1 - α\F2).(x) ≈ β .* F1.(x) .- α .\ F2.(x)
    @test (A*F2*A).(x) ≈ Ref(A) .* F2.(x) .* Ref(A)
    @test (F1 * F2).(x) ≈ F1.(x) .* F2.(x)
    @test (transpose(F2)).(x) ≈ transpose.(F2.(x))
    @test (conj(F2)).(x) ≈ conj.(F2.(x))
    @test (adjoint(F2)).(x) ≈ adjoint.(F2.(x))
    @test (real(F2)).(x) ≈ real.(F2.(x))
    @test (imag(F2)).(x) ≈ imag.(F2.(x))
    @test localdot(F1, F2).(x) ≈ dot.(F1.(x), F2.(x))

    @test rmul!(copy(F1), α).(x) ≈ α .* F1.(x)
    @test rmul!(copy(F2), α).(x) ≈ α .* F2.(x)
    @test rmul!(copy(F2), β).(x) ≈ β .* F2.(x)

    @test lmul!(α, copy(F1)).(x) ≈ α .* F1.(x)
    @test lmul!(α, copy(F2)).(x) ≈ α .* F2.(x)
    @test lmul!(β, copy(F2)).(x) ≈ β .* F2.(x)

    @test mul!(zero(F2), β, F1).(x) ≈ β .* F1.(x)

    @test axpy!(β, F1, copy(F2)).(x) ≈ β .* F1.(x) .+ F2.(x)
    @test axpby!(β, F1, α, copy(F2)).(x) ≈ β .* F1.(x) .+ α .* F2.(x)

    @test rmul!(copy(F1), α) ≈ F1 * α
    @test lmul!(β, copy(F2)) ≈ β * F2

    @test fit(F1, FourierSeries, p) ≈ F1
    @test fit(F2, FourierSeries, p) ≈ F2
end
