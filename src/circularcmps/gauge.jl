leftgauge(Ψ::UniformCircularCMPS, args...; kwargs...) =
    leftgauge!(copy(Ψ), args...; kwargs...)
rightgauge(Ψ::UniformCircularCMPS, args...; kwargs...) =
    rightgauge!(copy(Ψ), args...; kwargs...)

function leftgauge!(Ψ::UniformCircularCMPS, C₀ = one(Ψ.Q);
                    maxreorth = 10,
                    eigalg = Arnoldi(; krylovdim = min(64, length(Ψ.Q[0]))),
                    kwargs...)

    Ψ∞, λ, C, info = leftgauge!(InfiniteCMPS(Ψ.Q, Ψ.Rs))
    Ψ.Q = Ψ∞.Q
    Ψ.Rs = Ψ∞.Rs
    return Ψ, λ, C, info
end
function rightgauge!(Ψ::UniformCircularCMPS, C₀ = one(Ψ.Q);
                    maxreorth = 10,
                    eigalg = Arnoldi(; krylovdim = min(64, length(Ψ.Q[0]))),
                    kwargs...)

    Ψ∞, λ, C, info = rightgauge!(InfiniteCMPS(Ψ.Q, Ψ.Rs))
    Ψ.Q = Ψ∞.Q
    Ψ.Rs = Ψ∞.Rs
    return Ψ, λ, C, info
end
