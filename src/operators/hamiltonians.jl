struct LocalHamiltonian{O<:SumOfLocalTerms, S<:Real}
    h::O
    domain::Tuple{S,S}
end

∫(o::SumOfLocalTerms, domain::Tuple{<:Real,<:Real}) =
    LocalHamiltonian(o, promote(domain...))

∫(o::LocalOperator, domain::Tuple{<:Real,<:Real}) = ∫(1*o, domain)

domain(H::LocalHamiltonian) = H.domain

density(H::LocalHamiltonian) = H.h
