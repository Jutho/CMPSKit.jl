theme(:wong)
default(; fontfamily="Computer Modern", label=nothing, dpi=100, framestyle=:box)

# plotting utilities
function piticklabel(x::Rational, ::Val{:latex})
    iszero(x) && return L"0"
    S = x < 0 ? "-" : ""
    n, d = abs(numerator(x)), denominator(x)
    N = n == 1 ? "" : repr(n)
    d == 1 && return L"%$S%$N\pi"
    L"%$S\frac{%$N\pi}{%$d}"
end

function pitick(start, stop, denom; mode=:text)
    a = Int(cld(start, π / denom))
    b = Int(fld(stop, π / denom))
    tick = range(a * π / denom, b * π / denom; step=π / denom)
    ticklabel = piticklabel.((a:b) .// denom, Val(mode))
    return tick, ticklabel
end

# Local densities:
n̂ = ψ̂' * ψ̂; # particle density operator
k̂ = ∂ψ̂' * ∂ψ̂; # kinetic energy density operator
û = (ψ̂')^2 * ψ̂^2; # interaction operator

H(; c, μ, Δ) = ∫(k̂ - μ * n̂ + c * û, (-Inf, +Inf));

function _find_groundstate(D, H; gradtol=1e-10,
                           optalg=LBFGS(80; verbosity=0, maxiter=3000, gradtol=gradtol),
                           state=nothing)
    if isnothing(state)
        T = Float64 # scalar type
        Q₀ = randn(T, (D, D))
        R₀ = randn(T, (D, D))
        state = InfiniteCMPS(Constant(Q₀), Constant(R₀))
    end

    state, ρL, ρR, E, e, normgrad, numfg, history = groundstate(H, state; optalg=optalg,
                                                                verbosity=1,
                                                                gradtol=gradtol)
    return state
end

# increase bond dimension while somewhat preserving the state
function enlarge_state(state, Df; k=30, ϵ=0.1)
    Qi, Ri = state.Q[], state.Rs[1][]

    Di = size(Qi, 1)
    dD = Df - Di
    Qf = zeros(eltype(Qi), Df, Df)
    Rf = zeros(eltype(Ri), Df, Df)

    # Q → Q ⊕ -kI
    Qf[1:Di, 1:Di] .= Qi
    Qf[(Di + 1):end, (Di + 1):end] .= -k * I(dD)

    # R → R ⊕ 0
    Rf[1:Di, 1:Di] .= Ri

    Qf .+= ϵ * rand(size(Qf))
    Rf .+= ϵ * rand(size(Rf))

    return InfiniteCMPS(Constant(Qf), Constant(Rf))
end

# ramp up bond dimension
function find_groundstate(Ds, H; k=50.0, kwargs...)
    T = Float64 # scalar type
    Q₀ = randn(T, (Ds[1], Ds[1]))
    R₀ = randn(T, (Ds[1], Ds[1]))
    state = InfiniteCMPS(Constant(Q₀), Constant(R₀))

    for D in Ds
        println("Optimizing D=$D")
        @time state, ρL, ρR, E, e, normgrad, numfg, history = groundstate(H, state;
                                                                          kwargs...)
        println("---------------")
    end

    return state
end

## get ground state
c, μ, Δ = 15.0, 10.0, -0.0
tol = 1e-10
hamiltonian = H(; c=c, μ=μ, Δ=Δ)
hamiltonian_density = density(hamiltonian)

Ds = [4, 8, 12, 16]
state = find_groundstate(Ds, hamiltonian;
                         optalg=LBFGS(80; verbosity=1, maxiter=7000, gradtol=tol),
                         gradtol=tol)
D = maximum(Ds)
println("Energy density: ", expval(hamiltonian_density, state)[], "\n Particle density: ",
        expval(n̂, state)[], "\n Order parameter: ", expval(ψ̂, state)[])

# get trivial excitation spectrum
nvals = 10

ρ = expval(n̂, state)[]
ps = range(0, 4π * ρ, 20)
trivial_excitations = zeros(ComplexF64, length(ps), nvals);

for idx in eachindex(ps)
    space = InfiniteCMPSExcitationSpace(ps[idx], state, state)
    Heff = excitation_operator(hamiltonian, space)
    trivial_excitations[idx, :] .= eigsolve(Heff,
                                            (Constant(rand(ComplexF64, size(state.Q[]))),),
                                            nvals, :SR)[1][1:nvals]
end

scatter(ps ./ ρ, real.(trivial_excitations) ./ ρ^2; ylims=[0, 20],
        lab=permutedims(["Trivial"; fill("", nvals - 1)]), ms=2, c=:green,
        markerstrokewidth=0)
plot!(; xtick=pitick(0, 4π, 1; mode=:latex), ylabel="Excitation energy " * L"e/\rho^2",
      xlabel="Momentum " * L"p/\rho",
      title="Lieb-Liniger excitation spectrum\n\n" *
            L"\mu=%$(μ) \ | \ c=%$(c) \ | \ \gamma = %$(round(c/ρ, digits=3))\ | \ D=%$(D)",
      margins=7Plots.mm)