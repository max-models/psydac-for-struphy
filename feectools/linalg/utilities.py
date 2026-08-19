# coding: utf-8

import itertools
from math import sqrt

import cunumpy as xp
import numpy as np
from scipy import sparse

from feectools.ddm.mpi        import MockComm
from feectools.ddm.mpi        import mpi as MPI
from feectools.linalg.basic   import Vector
from feectools.linalg.stencil import StencilVector, StencilVectorSpace
from feectools.linalg.block   import BlockVector, BlockVectorSpace
from feectools.linalg.topetsc import petsc_local_to_psydac, get_npts_per_block

__all__ = (
    'array_to_psydac',
    'petsc_to_psydac',
    'tosparse_via_matvec',
    '_sym_ortho',
)

#==============================================================================
def array_to_psydac(x, V):
    """ 
    Convert a NumPy array to a Vector of the space V. This function is designed to be the inverse of the method .toarray() of the class Vector.
    Note: This function works in parallel but it is very costly and should be avoided if performance is a priority.

    Parameters
    ----------
    x : numpy.ndarray
        Array to be converted. It only contains the true data, the ghost regions must not be included.

    V : feectools.linalg.stencil.StencilVectorSpace or feectools.linalg.block.BlockVectorSpace
        Space of the final Psydac Vector.

    Returns
    -------
    u : feectools.linalg.stencil.StencilVector or feectools.linalg.block.BlockVector
        Element of space V, the coefficients of which (excluding ghost regions) are the entries of x. The ghost regions of u are up to date.

    """

    assert x.ndim == 1, 'Array must be 1D.'
    if x.dtype==complex:
        assert V.dtype==complex, 'Complex array cannot be converted to a real StencilVector'
    assert x.size == V.dimension, 'Array must have the same global size as the space.'

    u = V.zeros()
    _array_to_psydac_recursive(x, u)
    u.update_ghost_regions()

    return u


def _array_to_psydac_recursive(x, u):
    """
    Recursive function filling in the coefficients of each block of u.
    """
    assert isinstance(u, Vector)
    V = u.space

    assert x.ndim == 1, 'Array must be 1D.'
    if x.dtype==complex:
        assert V.dtype==complex, 'Complex array cannot be converted to a real StencilVector'
    assert x.size == V.dimension, 'Array must have the same global size as the space.'    

    if isinstance(V, BlockVectorSpace):
        for i, V_i in enumerate(V.spaces):
            x_i = x[:V_i.dimension]
            x   = x[V_i.dimension:]
            u_i = u[i]
            _array_to_psydac_recursive(x_i, u_i)

    elif isinstance(V, StencilVectorSpace):
        index_global = tuple(slice(s, e+1) for s, e in zip(V.starts, V.ends))
        u[index_global] = x.reshape(V.npts)[index_global]

    else:
        raise NotImplementedError(f'Can only handle StencilVector or BlockVector spaces, got {type(V)} instead')
    
#==============================================================================
def tosparse_via_matvec(op, format="csc"):
    """
    Assemble the full global sparse matrix of a `LinearOperator` by applying it to every
    global unit vector via `.dot()`, rather than via `.tosparse()`.

    Every operator's `.dot()` is already exercised (and therefore correct, including
    cross-rank ghost/boundary coupling) every time it is actually used, unlike
    `.tosparse()`, which several composed/derivative operators only implement correctly
    in serial (see e.g. `feectools.feec.derivatives.DirectionalDerivativeOperator.tosparse`).
    This is a port of `struphy.feec.linear_operators.LinOpWithTransp.toarray_struphy`'s
    `is_sparse=True` branch into feectools (which `DirectSolver` -- the caller this exists
    for -- must not import struphy from): same Allgather-starts/ends plus
    unit-vector-`dot()` plus gather/broadcast-triplets algorithm, so every rank ends up
    with an identical copy of the full global matrix (a "replicated" assembly, not a
    distributed one -- deliberate, see `feectools.linalg.solvers.DirectSolver`).

    Cost: O(N) collective `.dot()` calls, N = `op.domain.dimension` -- does not shrink
    with rank count (every call needs every rank's participation), so this is only
    appropriate as a one-time, cached setup cost, not something to call every step.

    Parameters
    ----------
    op : feectools.linalg.basic.LinearOperator
        Operator to assemble. `op.domain`/`op.codomain` must each be a
        `StencilVectorSpace` or `BlockVectorSpace`.

    format : str
        scipy.sparse matrix format of the result ("csr", "csc", "coo", ...).

    Returns
    -------
    out : scipy.sparse matrix
        The full `(op.codomain.dimension, op.domain.dimension)` matrix, identical on
        every rank.
    """
    v    = op.domain.zeros()
    tmp2 = op.codomain.zeros()

    if isinstance(op.domain, BlockVectorSpace):
        comm = op.domain.spaces[0].cart.comm
    elif isinstance(op.domain, StencilVectorSpace):
        comm = op.domain.cart.comm
    else:
        raise NotImplementedError(
            f'tosparse_via_matvec only supports StencilVectorSpace/BlockVectorSpace domains, got {type(op.domain)}',
        )

    if comm is None or isinstance(comm, MockComm):
        rank = 0
        size = 1
    else:
        rank = comm.Get_rank()
        size = comm.Get_size()

    numrows = op.codomain.dimension
    numcols = op.domain.dimension
    data, row, col = [], [], []

    def local_nonzero_rows(stencil_vec, row_offset):
        """(global_row_indices, values) for `stencil_vec`'s LOCAL interior data only.

        Avoids `stencil_vec.toarray()`: under the parallel branch that allocates a
        fresh, full-`codomain.dimension`-sized array and device->host-transfers it in
        full, every single call -- the dominant cost under CuPy (a device alloc, a
        device-side scatter-write kernel, and a full-size device->host copy per unit
        vector, even though a stencil operator's column is actually sparse/local).
        Reading only the local interior slice and adding `starts` to get the global row
        index (same approach `StencilMatrix._tocoo_no_pads` already uses for columns)
        transfers only the local, typically-mostly-zero data instead.
        """
        space = stencil_vec.space
        idx_local = tuple(
            slice(m * p, -m * p) if p != 0 else slice(0, None)
            for p, m in zip(stencil_vec.pads, space.shifts)
        )
        local_data = xp.to_numpy(stencil_vec._data[idx_local])
        nz = np.nonzero(local_data)
        starts = space.starts
        global_multi = tuple(nz[d] + int(starts[d]) for d in range(len(nz)))
        rows = row_offset + np.ravel_multi_index(global_multi, space.npts)
        return rows, local_data[nz]

    def codomain_local_nonzero_rows(vec):
        """`local_nonzero_rows`, dispatched over `op.codomain`'s type."""
        if isinstance(op.codomain, BlockVectorSpace):
            all_rows, all_vals = [], []
            row_offset = 0
            for b, sp in enumerate(op.codomain.spaces):
                r, val = local_nonzero_rows(vec[b], row_offset)
                all_rows.append(r)
                all_vals.append(val)
                row_offset += sp.dimension
            return np.concatenate(all_rows), np.concatenate(all_vals)
        return local_nonzero_rows(vec, 0)

    if isinstance(op.domain, BlockVectorSpace):
        starts = [vi.starts for vi in v]
        ends   = [vi.ends for vi in v]
        npts   = [sp.npts for sp in op.domain.spaces]
        nsp    = len(op.domain.spaces)
        ndim   = [sp.ndim for sp in op.domain.spaces]

        # Plain NumPy throughout: this is tiny host-side index bookkeeping (rank
        # starts/ends, a running column count), never device compute -- `xp.array`
        # under the CuPy backend would produce 0-d CuPy scalars that `range()` (and
        # plain Python int arithmetic below) cannot consume, the same class of
        # NumPy-vs-CuPy scalar-typing trap documented for `AdhocTorus`/`xp.sqrt`.
        startsarr = np.array([starts[i][j] for i in range(nsp) for j in range(ndim[i])], dtype=int)
        allstarts = np.empty(size * len(startsarr), dtype=int)
        if comm is None or isinstance(comm, MockComm):
            allstarts = startsarr
        else:
            comm.Allgather(startsarr, allstarts)
        allstarts = allstarts.reshape((size, len(startsarr)))

        endsarr = np.array([ends[i][j] for i in range(nsp) for j in range(ndim[i])], dtype=int)
        allends = np.empty(size * len(endsarr), dtype=int)
        if comm is None or isinstance(comm, MockComm):
            allends = endsarr
        else:
            comm.Allgather(endsarr, allends)
        allends = allends.reshape((size, len(endsarr)))

        for currentrank in range(size):
            spoint  = 0
            npredim = 0
            for h in range(nsp):
                iterables = [
                    range(int(allstarts[currentrank][i + npredim]), int(allends[currentrank][i + npredim]) + 1)
                    for i in range(ndim[h])
                ]
                for i in itertools.product(*iterables):
                    if rank == currentrank:
                        v[h][i] = 1.0
                    v[h].update_ghost_regions()
                    tmp2 *= 0.0
                    op.dot(v, out=tmp2)
                    c = spoint + int(np.ravel_multi_index(i, npts[h]))
                    rs, vals = codomain_local_nonzero_rows(tmp2)
                    row.append(rs)
                    col.append(np.full(rs.shape, c))
                    data.append(vals)
                    if rank == currentrank:
                        # No `update_ghost_regions()` here: resetting this rank's own
                        # entry back to 0 only needs to be visible to neighbors before
                        # their *own* next `dot()` call, which is exactly what the
                        # `update_ghost_regions()` at the top of every iteration (run by
                        # every rank, every iteration, whether or not it owns that
                        # iteration's unit vector) already provides -- an extra call
                        # here would just be the same synchronization done twice.
                        v[h][i] = 0.0
                cumulative = 1
                for i in range(ndim[h]):
                    cumulative *= npts[h][i]
                spoint  += cumulative
                npredim += ndim[h]

    else:
        starts = v.starts
        ends   = v.ends
        npts   = op.domain.npts
        ndim   = op.domain.ndim

        # Plain NumPy, same reasoning as the BlockVectorSpace branch above.
        startsarr = np.array([starts[j] for j in range(ndim)], dtype=int)
        allstarts = np.empty(size * len(startsarr), dtype=int)
        if comm is None or isinstance(comm, MockComm):
            allstarts = startsarr
        else:
            comm.Allgather(startsarr, allstarts)
        allstarts = allstarts.reshape((size, len(startsarr)))

        endsarr = np.array([ends[j] for j in range(ndim)], dtype=int)
        allends = np.empty(size * len(endsarr), dtype=int)
        if comm is None or isinstance(comm, MockComm):
            allends = endsarr
        else:
            comm.Allgather(endsarr, allends)
        allends = allends.reshape((size, len(endsarr)))

        for currentrank in range(size):
            iterables = [
                range(int(allstarts[currentrank][i]), int(allends[currentrank][i]) + 1) for i in range(ndim)
            ]
            for i in itertools.product(*iterables):
                if rank == currentrank:
                    v[i] = 1.0
                v.update_ghost_regions()
                op.dot(v, out=tmp2)
                c = int(np.ravel_multi_index(i, npts))
                rs, vals = codomain_local_nonzero_rows(tmp2)
                row.append(rs)
                col.append(np.full(rs.shape, c))
                data.append(vals)
                if rank == currentrank:
                    # See the matching comment in the BlockVectorSpace branch above --
                    # no `update_ghost_regions()` needed here, the one at the top of the
                    # next iteration already covers it.
                    v[i] = 0.0

    if comm is None or isinstance(comm, MockComm):
        all_rows, all_cols, all_data = row, col, data
    else:
        gathered_rows = comm.gather(row, root=0)
        gathered_cols = comm.gather(col, root=0)
        gathered_data = comm.gather(data, root=0)
        if rank == 0:
            all_rows = [item for sublist in gathered_rows for item in sublist]
            all_cols = [item for sublist in gathered_cols for item in sublist]
            all_data = [item for sublist in gathered_data for item in sublist]
            comm.bcast(all_rows, root=0)
            comm.bcast(all_cols, root=0)
            comm.bcast(all_data, root=0)
        else:
            all_rows = comm.bcast(None, root=0)
            all_cols = comm.bcast(None, root=0)
            all_data = comm.bcast(None, root=0)

    # `row`/`col`/`data` (and therefore `all_rows`/`all_cols`/`all_data`) are lists of
    # small per-iteration arrays -- one nonzero-entries batch per unit vector, from
    # `codomain_local_nonzero_rows` -- not lists of scalars, so concatenate before
    # handing them to `coo_matrix`, which expects flat 1D array-likes.
    if all_rows:
        all_rows = np.concatenate(all_rows)
        all_cols = np.concatenate(all_cols)
        all_data = np.concatenate(all_data)
    else:
        all_rows = all_cols = all_data = np.empty(0, dtype=int)

    mat = sparse.coo_matrix((all_data, (all_rows, all_cols)), shape=(numrows, numcols), dtype=op.dtype)
    return mat.asformat(format)

#==============================================================================
def petsc_to_psydac(x, Xh, out=None):
    """
    Convert a PETSc.Vec object to a StencilVector or BlockVector. It assumes that PETSc was installed with the configuration for complex numbers.
    Uses the index conversion functions in feectools.linalg.topetsc.py.

    Parameters
    ----------
    x : PETSc.Vec
      PETSc vector

    Xh : feectools.linalg.stencil.StencilVectorSpace | feectools.linalg.block.BlockVectorSpace
      Space of the coefficients of the Psydac vector.

    out : feectools.linalg.stencil.StencilVector | feectools.linalg.block.BlockVector, optional
      The Psydac vector where to store the result.

    Returns
    -------
    u : feectools.linalg.stencil.StencilVector | feectools.linalg.block.BlockVector
        Psydac vector. In the case of a BlockVector, the blocks must be StencilVector. The general case is not yet implemented.
    """
    
    if isinstance(Xh, BlockVectorSpace):
        if any([isinstance(Xh.spaces[b], BlockVectorSpace) for b in range(len(Xh.spaces))]):
            raise NotImplementedError('Block of blocks not implemented.')
        
        if out is not None:
            assert isinstance(out, BlockVector)
            assert out.space is Xh
            u = out
        else:
            u = BlockVector(Xh)

        comm       = x.comm
        dtype      = Xh._dtype
        localsize, globalsize = x.getSizes()
        assert globalsize == u.shape[0], 'Sizes of global vectors do not match'

        # Find shift for process k:
        # ..get number of points for each block, each process and each dimension:
        npts_local_per_block_per_process = xp.array(get_npts_per_block(Xh)) #indexed [b,k,d] for block b and process k and dimension d
        # ..get local sizes for each block and each process:
        local_sizes_per_block_per_process = xp.prod(npts_local_per_block_per_process, axis=-1) #indexed [b,k] for block b and process k
        # ..sum the sizes over all the blocks and the previous processes:
        index_shift = 0 + xp.sum(local_sizes_per_block_per_process[:,:comm.Get_rank()], dtype=int) #global variable

        for local_petsc_index in range(localsize):
            block_index, psydac_index = petsc_local_to_psydac(Xh, local_petsc_index)
            # Get value of local PETSc vector passing the global PETSc index
            value = x.getValue(local_petsc_index + index_shift) 
            if value != 0:
                u[block_index[0]]._data[psydac_index] = value if dtype is complex else value.real # PETSc always handles dtype specified in the installation configuration
        
    elif isinstance(Xh, StencilVectorSpace):

        if out is not None:
            assert isinstance(out, StencilVector)
            assert out.space is Xh
            u = out
        else:
            u = StencilVector(Xh)

        comm       = x.comm
        dtype      = Xh.dtype
        localsize, globalsize = x.getSizes()
        assert globalsize == u.shape[0], 'Sizes of global vectors do not match'

        # Find shift for process k:
        # ..get number of points for each process and each dimension:
        npts_local_per_block_per_process = xp.array(get_npts_per_block(Xh))[0] #indexed [k,d] for process k and dimension d
        # ..get local sizes for each process:
        local_sizes_per_block_per_process = xp.prod(npts_local_per_block_per_process, axis=-1) #indexed [k] for process k
        # ..sum the sizes over all the previous processes:
        index_shift = 0 + xp.sum(local_sizes_per_block_per_process[:comm.Get_rank()], dtype=int) #global variable

        for local_petsc_index in range(localsize):
            block_index, psydac_index = petsc_local_to_psydac(Xh, local_petsc_index) 
            # Get value of local PETSc vector passing the global PETSc index
            value = x.getValue(local_petsc_index + index_shift)
            if value != 0:
                u._data[psydac_index] = value if dtype is complex else value.real # PETSc always handles dtype specified in the installation configuration            

    else:
        raise ValueError('Xh must be a StencilVectorSpace or a BlockVectorSpace')

    u.update_ghost_regions()

    return u

#==============================================================================
def _sym_ortho(a, b):
    """
    Stable implementation of Givens rotation.
    This function was taken from the scipy repository
    https://github.com/scipy/scipy/blob/master/scipy/sparse/linalg/isolve/lsqr.py

    Notes
    -----
    The routine 'SymOrtho' was added for numerical stability. This is
    recommended by S.-C. Choi in [1]_.  It removes the unpleasant potential of
    ``1/eps`` in some important places (see, for example text following
    "Compute the next plane rotation Qk" in minres.py).

    References
    ----------
    .. [1] S.-C. Choi, "Iterative Methods for Singular Linear Equations
           and Least-Squares Problems", Dissertation,
           http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf
    """
    if b == 0:
        return _scalar_sign(a), 0, abs(a)
    elif a == 0:
        return 0, _scalar_sign(b), abs(b)
    elif abs(b) > abs(a):
        tau = a / b
        s = _scalar_sign(b) / sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = _scalar_sign(a) / sqrt(1+tau*tau)
        s = c * tau
        r = a / c
    return c, s, r

#==============================================================================
def _scalar_sign(x):
    """
    Sign of a real Python scalar. `xp.sign` (array_api_compat) requires its
    argument to expose a `.dtype` attribute, which plain Python floats don't have.
    """
    if x > 0:
        return 1.0
    elif x < 0:
        return -1.0
    return 0.0
