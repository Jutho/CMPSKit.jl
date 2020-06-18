# in place adjoint of square matrices

const _adjointblocksize = 32
function _adjoint!(A::AbstractMatrix, r = axes(A,1), c = axes(A,2))
    @assert r == c
    if length(r) > _adjointblocksize
        r1, r2 = _split(r)
        _adjoint!(A, r1, r1)
        _adjoint!(A, r2, r2)
        _swapadjoint!(A, r1, r2)
    else
        @inbounds for j in c
            @simd for i = first(r):(j-1)
                A[i,j], A[j,i] = conj(A[j,i]), conj(A[i,j])
            end
            A[j,j] = conj(A[j,j])
        end
    end
    return A
end

function _swapadjoint!(A::AbstractMatrix, r, c)
    if length(r) > length(c) && length(r) > _adjointblocksize
        r1, r2 = _split(r)
        _swapadjoint!(A, r1, c)
        _swapadjoint!(A, r2, c)
    elseif length(c) > _adjointblocksize
        c1, c2 = _split(c)
        _swapadjoint!(A, r, c1)
        _swapadjoint!(A, r, c2)
    else
        @inbounds for j in c
            @simd for i in r
                A[i,j], A[j,i] = conj(A[j,i]), conj(A[i,j])
            end
        end
    end
    return A
end

function _split(r::AbstractUnitRange{Int})
    i = first(r)
    f = last(r)
    m = (i+f) >> 1
    return (i:m, m+1:f)
end
