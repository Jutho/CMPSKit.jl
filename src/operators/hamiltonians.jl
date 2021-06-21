struct LocalHamiltonian{O<:SumOfLocalTerms, S<:Real}
    h::O
    domain::Tuple{S,S}
end

∫(o::SumOfLocalTerms, domain::Tuple{<:Real,<:Real}) =
    LocalHamiltonian(o, promote(domain...))

∫(o::LocalOperator, domain::Tuple{<:Real,<:Real}) = ∫(1*o, domain)

domain(H::LocalHamiltonian) = H.domain

density(H::LocalHamiltonian) = H.h

Base.:*(α::Number, H::LocalHamiltonian) = LocalHamiltonian(α * H.h, H.domain)
Base.:*(H::LocalHamiltonian, α::Number) = LocalHamiltonian(H.h * α, H.domain)
Base.:\(α::Number, H::LocalHamiltonian) = LocalHamiltonian(α \ H.h, H.domain)
Base.:/(H::LocalHamiltonian, α::Number) = LocalHamiltonian(H.h / α, H.domain)
Base.:+(H::LocalHamiltonian) = LocalHamiltonian(+H.h, H.domain)
Base.:-(H::LocalHamiltonian) = LocalHamiltonian(-H.h, H.domain)

function Base.:+(H1::LocalHamiltonian, H2::LocalHamiltonian)
    H1.domain == H2.domain || error("non-matching domains")
    return LocalHamiltonian(H1.h + H2.h, H1.domain)
end
function Base.:-(H1::LocalHamiltonian, H2::LocalHamiltonian)
    H1.domain == H2.domain || error("non-matching domains")
    return LocalHamiltonian(H1.h - H2.h, H1.domain)
end
