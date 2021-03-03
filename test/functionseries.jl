@testset "Constant" begin
    F1 = Constant(randn(Float64, (2,2)))
    F2 = Constant(randn(ComplexF64, (2,2)))

    p = 3*rand()
    a = rand()
    ab = (a, p+a)
    x = range(ab..., length = 11)

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

    @test fit(F1, Constant, ab) ≈ F1
    @test fit(F2, Constant, ab) ≈ F2
end

@testset "FourierSeries" begin
    p = 3*rand()
    a = rand()
    ab = (a, a+p)
    x = range(ab..., length = 201)

    F1 = FourierSeries([randn(Float64, (2,2)) for k = 1:5], p)
    F2 = FourierSeries([randn(ComplexF64, (2,2)) for k = 1:3], p)

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

    @test fit(F1, FourierSeries, ab) ≈ F1
    @test fit(F2, FourierSeries, ab) ≈ F2
end

@testset "TaylorSeries" begin
    p = 3*rand()
    a = rand()
    ab = (a, a+p)
    x = range(ab..., length = 201)

    F1 = TaylorSeries([randn(Float64, (2,2)) for k = 1:5], a)
    F2 = TaylorSeries([randn(ComplexF64, (2,2)) for k = 1:3], a)

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

    @test rmul!(copy(F1), α).(x) ≈ (F1 * α).(x)
    @test lmul!(β, copy(F2)).(x) ≈ (β * F2).(x)

    @test fit(F1, TaylorSeries, ab).(x) ≈ F1.(x)
    @test fit(F2, TaylorSeries, ab).(x) ≈ F2.(x)
end
