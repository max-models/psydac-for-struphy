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
    'parallel_tosparse',
    'FastAssemblyUnavailable',
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
                    aux = xp.to_numpy(tmp2.toarray())
                    for r in np.nonzero(aux)[0]:
                        data.append(aux[r])
                        col.append(c)
                        row.append(int(r))
                    if rank == currentrank:
                        v[h][i] = 0.0
                    v[h].update_ghost_regions()
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
                aux = xp.to_numpy(tmp2.toarray())
                for r in np.nonzero(aux)[0]:
                    data.append(aux[r])
                    col.append(c)
                    row.append(int(r))
                if rank == currentrank:
                    v[i] = 0.0
                v.update_ghost_regions()

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

    mat = sparse.coo_matrix((all_data, (all_rows, all_cols)), shape=(numrows, numcols), dtype=op.dtype)
    return mat.asformat(format)

#==============================================================================
class FastAssemblyUnavailable(Exception):
    """Raised by `parallel_tosparse` when the operator tree could not be assembled via
    the fast (O(1)-communication-round) path -- see its docstring. Callers should catch
    this and fall back to `tosparse_via_matvec`."""


def _local_flat_entries(V):
    """
    List of (setter, global_flat_index) pairs, one per DOF *owned* by this rank (no
    ghost/pad region), for a StencilVectorSpace or BlockVectorSpace V.

    `setter` is `('b', block_index, multi_index)` (usable as `vec[h][idx] = ...` for a
    BlockVector) or `('s', multi_index)` (usable as `vec[idx] = ...` otherwise) --
    tagged rather than inferred from shape, since a StencilVectorSpace's own
    `multi_index` can itself start with an int indistinguishable from a block index.
    `global_flat_index` uses the same block-major,
    `numpy.ravel_multi_index`-against-global-`npts` convention as
    `tosparse_via_matvec` and `StencilMatrix.tosparse()` (`_tocoo_no_pads`) -- the same
    one `Vector.toarray()` flattens to, which every caller of this module (e.g.
    `DirectSolver.solve`'s `b.toarray()`/`x_flat.reshape(...)` round trip) already
    relies on. All three MUST agree, since results from this function are combined
    with plain `StencilMatrix`/`BlockLinearOperator` sparse matrices in the same
    right-hand-side/solution vectors.
    """
    if isinstance(V, BlockVectorSpace):
        entries = []
        spoint = 0
        for h, sp in enumerate(V.spaces):
            npts = sp.npts
            iterables = [range(s, e + 1) for s, e in zip(sp.starts, sp.ends)]
            for idx in itertools.product(*iterables):
                flat = spoint + int(np.ravel_multi_index(idx, npts))
                entries.append((('b', h, idx), flat))
            spoint += int(np.prod(npts))
        return entries
    elif isinstance(V, StencilVectorSpace):
        npts = V.npts
        iterables = [range(s, e + 1) for s, e in zip(V.starts, V.ends)]
        return [(('s', idx), int(np.ravel_multi_index(idx, npts))) for idx in itertools.product(*iterables)]
    else:
        raise FastAssemblyUnavailable(
            f'_local_flat_entries only supports StencilVectorSpace/BlockVectorSpace, got {type(V)}',
        )


def _set_entry(vec, setter, value):
    if setter[0] == 'b':
        _, h, idx = setter
        vec[h][idx] = value
    else:
        _, idx = setter
        vec[idx] = value


def _get_entry(vec, setter):
    if setter[0] == 'b':
        _, h, idx = setter
        return vec[h][idx]
    else:
        _, idx = setter
        return vec[idx]


def _replicate_triples(rows, cols, vals, shape, comm, dtype):
    """Gather (rows, cols, vals) COO triples -- assumed *local* to this rank -- from
    every rank and sum-combine them (matching duplicates, e.g. periodic wraparound,
    exactly as `scipy.sparse.coo_matrix` does on `.tocsr()`) into one matrix identical
    on every rank. One collective round, regardless of `shape`.

    `rows`/`cols`/`vals` may be plain sequences or arrays; always gathered and
    concatenated as numpy arrays (`comm.allgather` pickles a numpy array through
    mpi4py's out-of-band buffer protocol, and `np.concatenate` is vectorized) rather
    than as Python lists -- for a leaf with real FEM bandwidth (tens to hundreds of
    thousands of local nonzeros, e.g. a 3D mass matrix), converting through
    element-by-element Python lists first was the dominant cost, dwarfing the O(1)
    round-count win this function exists for.
    """
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.asarray(vals, dtype=dtype)

    if comm is None or isinstance(comm, MockComm):
        all_rows, all_cols, all_vals = rows, cols, vals
    else:
        gathered = comm.allgather((rows, cols, vals))
        all_rows = np.concatenate([g[0] for g in gathered])
        all_cols = np.concatenate([g[1] for g in gathered])
        all_vals = np.concatenate([g[2] for g in gathered])
    return sparse.coo_matrix((all_vals, (all_rows, all_cols)), shape=shape, dtype=dtype).tocsr()


def _probe_vector(V, entries, offset):
    """A deterministic, reproducible-across-ranks probe Vector of space V: each owned
    DOF gets a distinct nonzero value derived from its global flat index (never 0, and
    never equal across two different `offset`s), so an operator's actual coupling
    structure is very unlikely to accidentally look diagonal/masking by coincidence."""
    v = V.zeros()
    for setter, flat in entries:
        _set_entry(v, setter, 1.0 + 0.618033988749895 * ((flat + offset) % 104729))
    v.update_ghost_regions()
    return v


def _validate_against_dot(node, candidate, comm, entries_domain, entries_codomain, seed):
    """Check `candidate @ p == node.dot(p)` (this rank's owned output entries only) for
    one probe vector `p`built from `_probe_vector`. `entries_domain` values already
    carry each entry's global flat column index; `entries_codomain` likewise for rows.
    Returns a local bool -- the caller combines these across ranks (see
    `parallel_tosparse`) before trusting `candidate`."""
    V_domain = node.domain
    p = _probe_vector(V_domain, entries_domain, seed)
    p_flat_local = {flat: _get_entry(p, setter) for setter, flat in entries_domain}

    if comm is None or isinstance(comm, MockComm):
        p_flat_full = dict(p_flat_local)
    else:
        gathered = comm.allgather(p_flat_local)
        p_flat_full = {}
        for d in gathered:
            p_flat_full.update(d)

    p_full = np.zeros(candidate.shape[1], dtype=candidate.dtype)
    for flat, val in p_flat_full.items():
        p_full[flat] = val

    q_candidate = candidate @ p_full
    q_true = node.dot(p)

    # An aggregate (L2-norm) check, not a per-entry one: a wide-bandwidth FEM operator
    # (e.g. a mass matrix with a degree-3 spline direction) sums many terms per row,
    # in a different order than `candidate`'s (scipy's own summation order for the
    # sparse matvec) -- individual output entries can then legitimately differ by much
    # more than a tight per-entry relative tolerance even when `candidate` is exactly
    # right, especially where terms partially cancel. Comparing the whole local output
    # vector's norm to the whole error vector's norm is robust to that per-entry
    # cancellation while still easily catching a genuinely wrong `candidate` (which
    # differs at O(1) relative scale, not at rounding-error scale).
    true_local = np.fromiter((_get_entry(q_true, setter) for setter, _ in entries_codomain), dtype=float)
    cand_local = np.fromiter((q_candidate[flat] for _, flat in entries_codomain), dtype=float)
    err = float(np.linalg.norm(true_local - cand_local))
    scale = float(np.linalg.norm(true_local))
    return err <= 1e-8 * scale + 1e-10


def _directional_derivative_triples(op):
    """Closed-form local (row, col, value) triples for a
    `feectools.feec.derivatives.DirectionalDerivativeOperator` -- `.tosparse()`'s
    default (no-pads) form isn't valid at nprocs > 1 for this operator (it asserts),
    and its `with_pads=True` form returns a small *local* matrix in a totally
    different (ghost-inclusive, non-globally-indexed) convention this module's
    block-major global indexing can't reuse -- so this reconstructs the same bidiagonal
    difference-operator matrix its serial `.tosparse()` builds, directly from the
    operator's own definition (`out[i] = sign * (in[i + e_d] - in[i])` along direction
    `d = op._diffdir`, `e_d` wrapped modulo `V.npts[d]` when periodic), one local
    (globally-indexed) row at a time -- no basis-vector sweep, no padding subtleties.
    Built in the "V -> W" (non-transposed) sense regardless of `op._transposed`;
    `parallel_tosparse` transposes the result back if needed, exactly as the operator's
    own serial `.tosparse()` does.
    """
    V, W, d = op._spaceV, op._spaceW, op._diffdir
    sign = -1.0 if op._negative else 1.0
    periodic = V.periods[d]

    if V.npts[d] == 1 and W.npts[d] == 1 and periodic:
        return [], [], []  # degenerate single-cell-periodic case: the zero matrix

    rows, cols, vals = [], [], []
    for idx, row_flat in _local_flat_entries(W):
        _, ii = idx
        jj = ii
        jj_next = list(ii)
        jj_next[d] = (ii[d] + 1) % V.npts[d] if periodic else ii[d] + 1
        col_flat = int(np.ravel_multi_index(jj, V.npts))
        rows.append(row_flat)
        cols.append(col_flat)
        vals.append(-sign)
        if periodic or jj_next[d] < V.npts[d]:
            col_next_flat = int(np.ravel_multi_index(jj_next, V.npts))
            rows.append(row_flat)
            cols.append(col_next_flat)
            vals.append(sign)
    return rows, cols, vals


def parallel_tosparse(op, comm, format="csr"):
    """
    Assemble the full global sparse matrix of a `LinearOperator` tree using O(1)
    collective-communication rounds (one per leaf node, roughly), instead of
    `tosparse_via_matvec`'s O(`op.domain.dimension`) rounds (one basis vector per
    global DOF) -- for the same replicated-on-every-rank result.

    Walks the operator tree using only types `feectools` itself defines
    (`SumLinearOperator`, `ScaledLinearOperator`, `ComposedLinearOperator`,
    `IdentityOperator`, `ZeroOperator`, `BlockLinearOperator`,
    `DirectionalDerivativeOperator`): the first five compose exactly the way
    `.tosparse()` already does in serial, just with the *leaves* below assembled
    without a per-DOF basis-vector sweep; `BlockLinearOperator` is recursed into
    block-by-block (not treated as one leaf) since a real Derham `grad`/`grad.T` can
    have `DirectionalDerivativeOperator` blocks, whose own `.tosparse()` is unusable
    in parallel (see `_directional_derivative_triples`, which reconstructs it in
    closed form instead). For anything else -- most of it defined outside feectools
    (`struphy.feec.mass.WeightedMassOperator`, `struphy.feec.linear_operators.
    BoundaryOperator`, ...), which this module must not import -- one of three
    O(1)-round strategies applies, tried in order:

      1. Duck-typed unwrap: if the node has a `._mat` plus the same
         `._V_extraction_op`/`._W_extraction_op`/`._V_boundary_op`/`._W_boundary_op`/
         `._transposed` attributes `struphy.feec.mass.WeightedMassOperator` has,
         rebuild the exact composition its own `.dot()` applies (boundary and
         extraction maps included, not just `._mat` alone -- an earlier version of
         this function assumed trivial extraction ops meant `._mat` alone was enough,
         which a real Struphy run showed is false whenever the boundary masks are
         non-trivial) from parts each recursed into via `build()` in turn.

      2. If the leaf has its own `.tosparse()` (true of `StencilMatrix` and
         `BlockLinearOperator`): call it *locally* (no communication -- the same call
         `A.tosparse()` already makes in serial, just once per rank instead of once
         globally) and `allgather`-sum the local fragments into the replicated global
         matrix.

      3. Otherwise, if `leaf.domain is leaf.codomain` (a necessary condition to act as
         a diagonal map): probe it with a value-tagged vector and check whether the
         output is consistent with a per-DOF diagonal scaling (this is exactly what
         essential-BC masking operators like `BoundaryOperator` are). Two `.dot()`
         calls plus one `allgather`.

    Every leaf's result is *always* cross-checked against the operator's own `.dot()`
    on a probe vector (strategies 1 and 2 both feed their candidate through the same
    `_validate_against_dot` check that strategy 3 uses to detect diagonality in the
    first place); an unrecognized leaf type (none of the three strategies applicable
    or valid) is treated the same as a failed check. Whether any of this happens is a
    pure function of operator *types*, identical on every rank by construction (same
    model, same run) -- so every rank always issues the same sequence of collective
    calls regardless of any individual check's pass/fail outcome; only *after* the
    full tree is walked does one final `allreduce(MPI.LAND)` combine every check
    across every rank into a single decision, so a data-dependent failure on one rank
    cannot leave another rank waiting on a collective call that rank never issues (no
    deadlock risk from divergent control flow). If that combined decision is False --
    or the tree contains a node type this function does not know how to handle at all
    (e.g. `MatrixFreeLinearOperator`, always a deterministic, type-only decision, so
    still consistent across ranks) -- `FastAssemblyUnavailable` is raised (on every
    rank, identically) and the caller should fall back to `tosparse_via_matvec`.

    Parameters
    ----------
    op : feectools.linalg.basic.LinearOperator
        Operator to assemble.

    comm : MPI.Comm | feectools.ddm.mpi.MockComm | None
        Communicator spanning every rank that owns a piece of `op`.

    format : str
        scipy.sparse matrix format of the result.

    Returns
    -------
    out : scipy.sparse matrix
        The full `(op.codomain.dimension, op.domain.dimension)` matrix, identical on
        every rank.
    """
    # Imported here, not at module scope: these are feectools types this function
    # checks via isinstance, kept local to make the "only feectools composite types
    # are special-cased" contract easy to audit at a glance.
    from feectools.linalg.basic import ComposedLinearOperator, IdentityOperator, ScaledLinearOperator, SumLinearOperator, ZeroOperator
    from feectools.linalg.block import BlockLinearOperator
    from feectools.feec.derivatives import DirectionalDerivativeOperator

    checks = []
    probe_seed = [1000003]  # mutable cell; a fresh seed per leaf keeps probes independent

    def build(node):
        if isinstance(node, ScaledLinearOperator):
            return node._scalar * build(node._operator)
        if isinstance(node, SumLinearOperator):
            mats = [build(a) for a in node._addends]
            out = mats[0]
            for m in mats[1:]:
                out = out + m
            return out
        if isinstance(node, ComposedLinearOperator):
            mats = [build(m) for m in node._multiplicants]
            out = mats[0]
            for m in mats[1:]:
                out = out @ m
            return out
        if isinstance(node, IdentityOperator):
            return sparse.identity(node.domain.dimension, format="csr", dtype=node.dtype or float)
        if isinstance(node, ZeroOperator):
            return sparse.csr_matrix(node.shape, dtype=node.dtype or float)
        if isinstance(node, BlockLinearOperator):
            # Recurse into each block individually rather than calling
            # `node.tosparse()` on the whole thing: a real Derham `grad`/`grad.T` is a
            # BlockLinearOperator whose blocks can themselves be
            # `DirectionalDerivativeOperator`s, whose *own* default `.tosparse()`
            # asserts outright at nprocs > 1 (see `_directional_derivative_triples`)
            # -- one such block would otherwise make the whole (possibly mostly
            # StencilMatrix) BlockLinearOperator's `.tosparse()` raise.
            nrows, ncols = node.n_block_rows, node.n_block_cols
            block_domain = (lambda j: node.domain[j]) if ncols > 1 else (lambda j: node.domain)
            block_codomain = (lambda i: node.codomain[i]) if nrows > 1 else (lambda i: node.codomain)
            grid = [[None for _ in range(ncols)] for _ in range(nrows)]
            for i in range(nrows):
                for j in range(ncols):
                    if (i, j) in node._blocks:
                        grid[i][j] = build(node._blocks[i, j])
                    else:
                        grid[i][j] = sparse.csr_matrix((block_codomain(i).dimension, block_domain(j).dimension))
            return sparse.bmat(grid, format="csr")
        if isinstance(node, DirectionalDerivativeOperator):
            # No generic strategy below applies (not diagonal-shaped in general, and
            # its own `.tosparse()` is unusable here -- see
            # `_directional_derivative_triples`); its structure is simple and fixed
            # enough to reconstruct in closed form directly, still validated below
            # like everything else.
            V, W = node._spaceV, node._spaceW
            rows, cols, vals = _directional_derivative_triples(node)
            mat_vw = _replicate_triples(rows, cols, vals, (W.dimension, V.dimension), comm, node.dtype or float)
            candidate = mat_vw.T.tocsr() if node._transposed else mat_vw
            entries_domain = _local_flat_entries(node.domain)
            entries_codomain = _local_flat_entries(node.codomain)
            probe_seed[0] += 97
            ok = _validate_against_dot(node, candidate, comm, entries_domain, entries_codomain, probe_seed[0])
            checks.append(ok)
            return candidate

        # Leaf: not one of the composite types above. Every strategy applicable to
        # this leaf's *type* is always attempted, on every rank, regardless of any
        # other rank's or strategy's data-dependent validation outcome -- see the
        # docstring's "same collective calls on every rank" invariant. Only the first
        # strategy that actually validates is kept.
        entries_domain = _local_flat_entries(node.domain)
        entries_codomain = _local_flat_entries(node.codomain)
        shape = (node.codomain.dimension, node.domain.dimension)
        dtype = node.dtype or float
        probe_seed[0] += 97

        # Duck-typed unwrap: struphy's `WeightedMassOperator` (M0, M1, ...) computes
        # `V_boundary_op @ V_extraction_op @ _mat @ W_extraction_op.T @ W_boundary_op.T`
        # (or the mirrored order when `._transposed`) on every `.dot()` call, by
        # default with the boundary masks actually applied (`apply_bc=True`) -- *not*
        # just `._mat` alone, even when both extraction ops are the identity (its own
        # boundary masks can still be non-trivial, e.g. Dirichlet-BC-adjacent DOFs on
        # a component of an Hcurl mass matrix -- discovered by this function's own
        # validation rejecting the naive "just `._mat`" shortcut on exactly such a
        # case, not by inspecting struphy's BC configuration). Rebuilding that same
        # composition from its parts -- each recursed into via `build()`, so a
        # boundary mask that itself needs the diagonal-probe strategy below still
        # gets it -- is exact when every part is present (duck-typed by attribute,
        # not `isinstance`, since none of these types live in feectools); still
        # validated below regardless, as insurance against this composition itself
        # being incomplete for some other struphy wrapper shaped differently.
        parts = [getattr(node, name, "missing") for name in (
            "_mat", "_V_extraction_op", "_W_extraction_op", "_V_boundary_op", "_W_boundary_op",
        )]
        if "missing" not in parts:
            inner_mat, v_ext, w_ext, v_bnd, w_bnd = parts
            transposed = bool(getattr(node, "_transposed", False))
            try:
                # Matches struphy.feec.mass.WeightedMassOperator.dot's own step
                # sequence exactly (apply_bc=True, its default): non-transposed
                # applies V_boundary_op.T, then V_extraction_op.T, then `._mat`, then
                # W_extraction_op, then W_boundary_op, in that order (v -> out); the
                # composed *matrix* is those same maps in reverse (rightmost applied
                # first). `._transposed` mirrors V and W throughout.
                if not transposed:
                    order = [w_bnd, w_ext, inner_mat, v_ext.transpose(), v_bnd.transpose()]
                else:
                    order = [v_bnd, v_ext, inner_mat, w_ext.transpose(), w_bnd.transpose()]
                mats = [build(m) for m in order]
                candidate_inner = mats[0]
                for m in mats[1:]:
                    candidate_inner = candidate_inner @ m
            except Exception:
                candidate_inner = None
            if candidate_inner is not None and _validate_against_dot(
                node, candidate_inner, comm, entries_domain, entries_codomain, probe_seed[0],
            ):
                checks.append(True)
                return candidate_inner

        candidate_tosparse = None
        try:
            local_coo = node.tosparse().tocoo()
            candidate_tosparse = _replicate_triples(
                local_coo.row, local_coo.col, local_coo.data,
                shape, comm, dtype,
            )
        except Exception:
            pass  # this leaf's .tosparse() -- if it has one -- doesn't work here (e.g.
            # raises, or -- as for struphy's BoundaryOperator -- succeeds but is
            # documented serial-only and produces locally- rather than
            # globally-indexed rows/cols at nprocs > 1); validation below (or, failing
            # that, the diagonal-probe strategy) is what actually decides trust, not
            # whether this call happened to raise.

        if candidate_tosparse is not None and _validate_against_dot(
            node, candidate_tosparse, comm, entries_domain, entries_codomain, probe_seed[0],
        ):
            checks.append(True)
            return candidate_tosparse

        if node.domain is not node.codomain:
            # Not diagonal-shaped, and the .tosparse() attempt above (if any) didn't
            # validate: nothing left to try for this leaf.
            checks.append(False)
            return candidate_tosparse if candidate_tosparse is not None else sparse.csr_matrix(shape, dtype=dtype)

        # Diagonal-probe strategy: two independent value-tagged probes; a true
        # diagonal map reproduces (scaled by a per-DOF constant) or zeroes each one,
        # consistently between the two -- exactly what an essential-BC mask does.
        p1 = _probe_vector(node.domain, entries_domain, probe_seed[0])
        p2 = _probe_vector(node.domain, entries_domain, probe_seed[0] + 50000)
        try:
            o1 = node.dot(p1)
            o2 = node.dot(p2)
        except Exception:
            checks.append(False)
            return candidate_tosparse if candidate_tosparse is not None else sparse.csr_matrix(shape, dtype=dtype)

        rows, cols, vals = [], [], []
        ok = True
        for setter, flat in entries_domain:
            v1 = _get_entry(p1, setter)
            v2 = _get_entry(p2, setter)
            a1 = _get_entry(o1, setter)
            a2 = _get_entry(o2, setter)
            is_zero = abs(a1) < 1e-300 and abs(a2) < 1e-300
            if is_zero:
                continue
            d1 = a1 / v1
            d2 = a2 / v2
            if abs(d1 - d2) > 1e-8 * max(1.0, abs(d1)):
                ok = False
                break
            rows.append(flat)
            cols.append(flat)
            vals.append(d1)
        checks.append(ok)
        return _replicate_triples(rows, cols, vals, shape, comm, dtype)

    result = build(op)

    all_ok = all(checks)
    if comm is not None and not isinstance(comm, MockComm):
        all_ok = comm.allreduce(all_ok, op=MPI.LAND)
    if not all_ok:
        raise FastAssemblyUnavailable('one or more leaf operators could not be verified')

    return result.asformat(format)

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
