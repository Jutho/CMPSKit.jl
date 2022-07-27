D = 8 # bond dimension
T = Float64 # element type
L = 1. # length

# particle number
N̂ = ∫(ψ̂'*ψ̂, (0,L))

# Hamiltonian, for example Lieb-Liniger
#--------------------------------------
# interaction strength
g = 10000.
# for example, large g => tonks girardau limit => similar to free fermions

# chemical potential, for example, make all states with momentum (2*pi*k/L) be occupied
# for all k = -M+1/2, ..., +M-1/2
M = 10
μ = (2*pi*M/L)^2
# expected particle number is now
N = 2*M
# expected energy is roughly
E = sum( k->(2*pi*k/L)^2 - μ, (-M+1//2):(M-1//2))

Ĥ = ∫(∂ψ̂'*∂ψ̂ - μ * ψ̂'*ψ̂ + g * ψ̂'*ψ̂'*ψ̂*ψ̂, (0, L))

# initial cMPS
Q, R = Constant.((randn(T, D, D), randn(T, D, D)))
Ψ = CircularCMPS(Q, R, L)

ΨL, = CMPSKit.groundstate(Ĥ, Ψ; gradtol = 1e-6, optalg = CMPSKit.LBFGS(20; gradtol = 1e-5, verbosity = 0, maxiter = 2000))

expval(N̂, ΨL) # compare to N
expval(Ĥ, ΨL) # compare to E

# converge some further
ΨL, = CMPSKit.groundstate(Ĥ, ΨL; gradtol = 1e-6, optalg = CMPSKit.LBFGS(20; gradtol = 1e-5, verbosity = 0, maxiter = 5000))
