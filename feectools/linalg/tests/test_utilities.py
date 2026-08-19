import cunumpy as xp
import numpy as np
import pytest
from scipy import sparse

from feectools.ddm.cart import CartDecomposition, DomainDecomposition
from feectools.ddm.mpi import mpi as MPI
from feectools.feec.derivatives import DirectionalDerivativeOperator
from feectools.linalg.basic import ComposedLinearOperator, IdentityOperator, LinearOperator, ScaledLinearOperator, SumLinearOperator
from feectools.linalg.block import BlockLinearOperator, BlockVectorSpace
from feectools.linalg.stencil import StencilMatrix, StencilVectorSpace
from feectools.linalg.utilities import (
    FastAssemblyUnavailable,
    _get_entry,
    _local_flat_entries,
    _set_entry,
    parallel_tosparse,
    tosparse_via_matvec,
)


def compute_global_starts_ends(domain_decomposition, npts):
    global_starts = [None] * len(npts)
    global_ends = [None] * len(npts)
    for axis in range(len(npts)):
        ee = domain_decomposition.global_element_ends[axis]
        global_ends[axis] = ee.copy()
        global_ends[axis][-1] = npts[axis] - 1
        global_starts[axis] = xp.array([0] + (global_ends[axis][:-1] + 1).tolist())
    return global_starts, global_ends


def make_space(n1, n2, p1, p2, comm, periodic=True):
    D = DomainDecomposition([n1, n2], periods=[periodic, False], comm=comm)
    npts = [n1, n2]
    gs, ge = compute_global_starts_ends(D, npts)
    cart = CartDecomposition(D, npts, gs, ge, pads=[p1, p2], shifts=[1, 1])
    return StencilVectorSpace(cart, dtype=float)


def make_stencil_matrix(V, p1, p2, scale=1.0):
    A = StencilMatrix(V, V)
    n_offdiag = (2 * p1 + 1) * (2 * p2 + 1) - 1
    for k1 in range(-p1, p1 + 1):
        for k2 in range(-p2, p2 + 1):
            A[:, :, k1, k2] = 0.0 if (k1 == 0 and k2 == 0) else -1.0 * scale
    A[:, :, 0, 0] = (n_offdiag + 1.0) * scale
    A.remove_spurious_entries()
    return A


class FakeBoundaryOperator(LinearOperator):
    """Mimics struphy.feec.linear_operators.BoundaryOperator well enough to exercise
    `parallel_tosparse`'s diagonal-probe fallback: a diagonal 0/1 mask (zeroes the
    first local DOF of each block on each rank), with a `.tosparse()` that is only
    valid in serial (returns a small *locally*-indexed diagonal, exactly as struphy's
    real implementation does -- see `struphy.feec.linear_operators.
    BoundaryOperator.tosparse`) -- so the fast path must detect the mismatch via
    validation and retry with the probe strategy instead of trusting `.tosparse()`
    blindly. Works for both a plain StencilVectorSpace (V) and a BlockVectorSpace
    (e.g. Hcurl, the codomain a real `grad` is wrapped in) domain, matching either
    shape BoundaryOperator actually appears in.
    """

    def __init__(self, V):
        self._V = V
        self._entries = _local_flat_entries(V)

    @property
    def domain(self):
        return self._V

    @property
    def codomain(self):
        return self._V

    @property
    def dtype(self):
        return float

    def tosparse(self):
        # Deliberately wrong at nprocs > 1: local indices, not global ones.
        n = len(self._entries)
        diag = np.ones(n)
        diag[0] = 0.0
        return sparse.diags(diag, format="csr")

    def toarray(self):
        return self.tosparse().toarray()

    def transpose(self, conjugate=False):
        return self

    def dot(self, v, out=None):
        if out is None:
            out = self.codomain.zeros()
        else:
            out *= 0.0
        for k, (setter, _) in enumerate(self._entries):
            _set_entry(out, setter, 0.0 if k == 0 else _get_entry(v, setter))
        out.update_ghost_regions()
        return out


class UnsupportedOperator(LinearOperator):
    """A leaf `parallel_tosparse` cannot possibly handle: no `.tosparse()`, and
    domain is not codomain (so the diagonal-probe strategy doesn't apply either)."""

    def __init__(self, V_in, V_out):
        self._Vin = V_in
        self._Vout = V_out

    @property
    def domain(self):
        return self._Vin

    @property
    def codomain(self):
        return self._Vout

    @property
    def dtype(self):
        return float

    def tosparse(self):
        raise NotImplementedError

    def toarray(self):
        raise NotImplementedError

    def transpose(self, conjugate=False):
        return UnsupportedOperator(self._Vout, self._Vin)

    def dot(self, v, out=None):
        if out is None:
            out = self.codomain.zeros()
        return out


class FakeWeightedMassOperator(LinearOperator):
    """Mimics struphy.feec.mass.WeightedMassOperator closely enough to exercise
    `parallel_tosparse`'s duck-typed `._mat`-composition unwrap: wraps an inner
    `._mat` behind identity extraction ops (trivial, as struphy's own serial
    `.tosparse()` requires) but *non-trivial* boundary ops (masks the first and last
    local DOF on each rank) -- i.e. `.dot()` is genuinely not the same as `._mat.dot()`
    alone, which is exactly the bug this class was written to catch (found via a real
    Struphy run, not anticipated up front -- see the git history of
    `parallel_tosparse`'s `._mat`-unwrap branch).
    """

    def __init__(self, V, mat, mask_first=True, mask_last=True):
        self._V = V
        self._mat = mat
        self._V_extraction_op = IdentityOperator(V)
        self._W_extraction_op = IdentityOperator(V)
        self._V_boundary_op = _MaskBoundaryOperator(V, mask_first, mask_last)
        self._W_boundary_op = _MaskBoundaryOperator(V, mask_first, mask_last)
        self._transposed = False

    @property
    def domain(self):
        return self._V

    @property
    def codomain(self):
        return self._V

    @property
    def dtype(self):
        return float

    def tosparse(self):
        # Deliberately unusable at nprocs > 1, exactly like struphy's real
        # WeightedMassOperator.tosparse() when boundary masking is actually active
        # (it asserts outright there); here it just raises, which
        # `parallel_tosparse` must also handle gracefully.
        raise NotImplementedError

    def toarray(self):
        raise NotImplementedError

    def transpose(self, conjugate=False):
        raise NotImplementedError

    def dot(self, v, out=None):
        tmp = self._V_boundary_op.transpose().dot(v)
        tmp = self._V_extraction_op.transpose().dot(tmp)
        tmp = self._mat.dot(tmp)
        tmp = self._W_extraction_op.dot(tmp)
        return self._W_boundary_op.dot(tmp, out=out)


class _MaskBoundaryOperator(LinearOperator):
    """A minimal BoundaryOperator stand-in: zeroes the first and/or last local DOF on
    each rank (self-adjoint, so `.transpose()` returns itself)."""

    def __init__(self, V, mask_first, mask_last):
        self._V = V
        self._entries = _local_flat_entries(V)
        self._mask_first = mask_first
        self._mask_last = mask_last

    @property
    def domain(self):
        return self._V

    @property
    def codomain(self):
        return self._V

    @property
    def dtype(self):
        return float

    def tosparse(self):
        raise NotImplementedError

    def toarray(self):
        raise NotImplementedError

    def transpose(self, conjugate=False):
        return self

    def dot(self, v, out=None):
        if out is None:
            out = self.codomain.zeros()
        else:
            out *= 0.0
        n = len(self._entries)
        for k, (setter, _) in enumerate(self._entries):
            masked = (self._mask_first and k == 0) or (self._mask_last and k == n - 1)
            _set_entry(out, setter, 0.0 if masked else _get_entry(v, setter))
        out.update_ghost_regions()
        return out


@pytest.mark.parametrize('n1', [8, 16])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parallel
def test_parallel_tosparse_matches_matvec_stencil_matrix(n1, p1, verbose=False):
    """A plain StencilMatrix (no wrapper) must assemble identically via the fast
    (`parallel_tosparse`) and slow (`tosparse_via_matvec`) paths."""
    n2, p2 = 8, 1
    comm = MPI.COMM_WORLD
    V = make_space(n1, n2, p1, p2, comm)
    A = make_stencil_matrix(V, p1, p2)

    fast = parallel_tosparse(A, comm)
    slow = tosparse_via_matvec(A, format="csr")

    diff = abs(fast - slow).max()
    if verbose:
        print(f"n1={n1} p1={p1} nprocs={comm.Get_size()} diff={diff:.2e}")
    assert diff < 1e-10


@pytest.mark.parallel
def test_parallel_tosparse_composed_and_sum(verbose=False):
    """A Sum-of-Scaled-and-Composed operator tree (the shape ImplicitDiffusion's
    left-hand side actually takes: sigma*M + G^T @ D @ G) must also match the slow
    reference path."""
    n1, n2, p1, p2 = 10, 8, 1, 1
    comm = MPI.COMM_WORLD
    V = make_space(n1, n2, p1, p2, comm)
    M = make_stencil_matrix(V, p1, p2, scale=1.0)
    G = make_stencil_matrix(V, p1, p2, scale=0.5)
    D = make_stencil_matrix(V, p1, p2, scale=2.0)

    composed = ComposedLinearOperator(V, V, G, D, G)
    scaled = ScaledLinearOperator(V, V, c=3.0, A=M)
    total = SumLinearOperator(V, V, scaled, composed)

    fast = parallel_tosparse(total, comm)
    slow = tosparse_via_matvec(total, format="csr")

    diff = abs(fast - slow).max()
    if verbose:
        print(f"nprocs={comm.Get_size()} diff={diff:.2e}")
    assert diff < 1e-8


@pytest.mark.parallel
def test_parallel_tosparse_block_vector_space(verbose=False):
    """A BlockLinearOperator (grad's actual shape, e.g. H1 -> Hcurl's 3 stacked
    components) must also assemble identically via the fast and slow paths."""
    n1, n2, p1, p2 = 8, 6, 1, 1
    comm = MPI.COMM_WORLD
    V = make_space(n1, n2, p1, p2, comm, periodic=False)
    W = BlockVectorSpace(V, V)
    B = BlockLinearOperator(W, W)
    B[0, 0] = make_stencil_matrix(V, p1, p2, scale=1.0)
    B[1, 1] = make_stencil_matrix(V, p1, p2, scale=2.0)
    B[0, 1] = make_stencil_matrix(V, p1, p2, scale=0.3)

    fast = parallel_tosparse(B, comm)
    slow = tosparse_via_matvec(B, format="csr")

    diff = abs(fast - slow).max()
    if verbose:
        print(f"nprocs={comm.Get_size()} diff={diff:.2e}")
    assert diff < 1e-8


@pytest.mark.parallel
def test_parallel_tosparse_diagonal_probe_fallback_block_vector_space(verbose=False):
    """The same diagonal-mask-on-both-sides shape as
    `test_parallel_tosparse_diagonal_probe_fallback`, but over a BlockVectorSpace --
    the actual shape `BoundaryOperator ∘ grad ∘ BoundaryOperator` takes in the real
    Poisson benchmark (grad's codomain, Hcurl, has 3 stacked components)."""
    n1, n2, p1, p2 = 8, 6, 1, 1
    comm = MPI.COMM_WORLD
    V = make_space(n1, n2, p1, p2, comm, periodic=False)
    W = BlockVectorSpace(V, V)
    B = BlockLinearOperator(W, W)
    B[0, 0] = make_stencil_matrix(V, p1, p2, scale=1.0)
    B[1, 1] = make_stencil_matrix(V, p1, p2, scale=2.0)
    mask = FakeBoundaryOperator(W)

    total = ComposedLinearOperator(W, W, mask, B, mask)

    fast = parallel_tosparse(total, comm)
    slow = tosparse_via_matvec(total, format="csr")

    diff = abs(fast - slow).max()
    if verbose:
        print(f"nprocs={comm.Get_size()} diff={diff:.2e}")
    assert diff < 1e-8


@pytest.mark.parallel
def test_parallel_tosparse_diagonal_probe_fallback(verbose=False):
    """A BoundaryOperator-like diagonal mask, wired in on both sides of a
    StencilMatrix (the actual shape struphy's BC-wrapped operators take), must be
    correctly recovered by the diagonal-probe fallback -- not silently misassembled
    from its serial-only `.tosparse()`."""
    n1, n2, p1, p2 = 8, 6, 1, 1
    comm = MPI.COMM_WORLD
    V = make_space(n1, n2, p1, p2, comm, periodic=False)
    A = make_stencil_matrix(V, p1, p2)
    mask = FakeBoundaryOperator(V)

    total = ComposedLinearOperator(V, V, mask, A, mask)

    fast = parallel_tosparse(total, comm)
    slow = tosparse_via_matvec(total, format="csr")

    diff = abs(fast - slow).max()
    if verbose:
        print(f"nprocs={comm.Get_size()} diff={diff:.2e}")
    assert diff < 1e-8


@pytest.mark.parallel
def test_parallel_tosparse_raises_for_unsupported_operator(verbose=False):
    """An operator this module genuinely cannot handle (no `.tosparse()`, not
    diagonal-shaped) must raise `FastAssemblyUnavailable` -- identically on every
    rank, so callers can fall back to `tosparse_via_matvec` without any risk of a
    partial/divergent collective-call sequence."""
    n1, n2, p1, p2 = 8, 6, 1, 1
    comm = MPI.COMM_WORLD
    Vin = make_space(n1, n2, p1, p2, comm, periodic=False)
    Vout = make_space(n1, n2, p1, p2, comm, periodic=False)
    op = UnsupportedOperator(Vin, Vout)

    with pytest.raises(FastAssemblyUnavailable):
        parallel_tosparse(op, comm)


def make_deriv_space(n1, n2, p1, p2, comm, periodic):
    # dim 0 always periodic (matches make_space's default); dim 1 (the
    # differentiation direction in test_parallel_tosparse_directional_derivative)
    # uses `periodic`, since that's the one whose periodicity actually changes the
    # matrix structure (wraparound coupling vs. a one-fewer-point boundary).
    D = DomainDecomposition([n1, n2], periods=[True, periodic], comm=comm)
    npts = [n1, n2]
    gs, ge = compute_global_starts_ends(D, npts)
    cart = CartDecomposition(D, npts, gs, ge, pads=[p1, p2], shifts=[1, 1])
    return StencilVectorSpace(cart, dtype=float)


@pytest.mark.parametrize('diffdir_periodic', [False, True])
@pytest.mark.parametrize('transposed', [False, True])
@pytest.mark.parallel
def test_parallel_tosparse_directional_derivative(diffdir_periodic, transposed, verbose=False):
    """`DirectionalDerivativeOperator`'s default `.tosparse()` asserts outright at
    nprocs > 1 (see `_directional_derivative_triples`'s docstring) -- the closed-form
    reconstruction it falls back to instead must match the slow reference path, for
    both a periodic and a non-periodic differentiation direction, transposed or not.
    """
    p1, p2 = 1, 1
    comm = MPI.COMM_WORLD
    n1, n2 = 8, 8
    V = make_deriv_space(n1, n2, p1, p2, comm, periodic=diffdir_periodic)
    W = make_deriv_space(n1, n2 if diffdir_periodic else n2 - 1, p1, p2, comm, periodic=diffdir_periodic)

    op = DirectionalDerivativeOperator(V, W, diffdir=1, negative=False, transposed=transposed)

    fast = parallel_tosparse(op, comm)
    slow = tosparse_via_matvec(op, format="csr")

    diff = abs(fast - slow).max()
    if verbose:
        print(f"periodic={diffdir_periodic} transposed={transposed} nprocs={comm.Get_size()} diff={diff:.2e}")
    assert diff < 1e-10


@pytest.mark.parallel
def test_parallel_tosparse_weighted_mass_operator_boundary_composition(verbose=False):
    """`FakeWeightedMassOperator` wraps a StencilMatrix behind trivial extraction ops
    but *non-trivial* boundary masking (`.dot()` != `._mat.dot()` alone) -- exactly
    the shape that caused a real, silent ~8% numeric mismatch against the naive
    "just unwrap `._mat`" shortcut on an actual Struphy run (before
    `parallel_tosparse`'s `._mat`-composition unwrap rebuilt the *whole*
    boundary/extraction chain instead). Must match the slow reference path.
    """
    n1, n2, p1, p2 = 8, 6, 1, 1
    comm = MPI.COMM_WORLD
    V = make_space(n1, n2, p1, p2, comm, periodic=False)
    inner = make_stencil_matrix(V, p1, p2)
    op = FakeWeightedMassOperator(V, inner)

    fast = parallel_tosparse(op, comm)
    slow = tosparse_via_matvec(op, format="csr")

    diff = abs(fast - slow).max()
    if verbose:
        print(f"nprocs={comm.Get_size()} diff={diff:.2e}")
    assert diff < 1e-8
