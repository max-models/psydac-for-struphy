
import cunumpy as xp
import pytest
from feectools.linalg.solvers import inverse, DirectSolver
from feectools.linalg.stencil import StencilVectorSpace, StencilMatrix, StencilVector
from feectools.linalg.basic import LinearSolver
from feectools.ddm.cart import DomainDecomposition, CartDecomposition
from feectools.ddm.mpi import mpi as MPI


def define_data_hermitian(n, p, dtype=float):
    domain_decomposition = DomainDecomposition([n - p], [False])
    cart = CartDecomposition(domain_decomposition, [n], [xp.array([0])], [xp.array([n - 1])], [p], [1])
    # ... Vector Spaces
    V = StencilVectorSpace(cart,dtype=dtype)
    e = V.ends[0]
    s = V.starts[0]

    # Build banded matrix with 2p+1 diagonals: must be symmetric and positive definite
    # Here we assign value 2*p on main diagonal and -1 on other diagonals
    if dtype==complex:
        factor=1+1j
    else:
        factor=1
    A = StencilMatrix(V, V)
    A[:, -p:0] = 1-1*factor
    A[:, 0:1] = 2 * p
    A[:, 1:p + 1] = 1-1*factor.conjugate()
    A.remove_spurious_entries()

    # Build exact solution
    xe = StencilVector(V)
    xe[s:e + 1] = factor*xp.random.random(e + 1 - s)
    return(V, A, xe)

def define_data(n, p, matrix_data, dtype=float):
    domain_decomposition = DomainDecomposition([n - p], [False])
    cart = CartDecomposition(domain_decomposition, [n], [xp.array([0])], [xp.array([n - 1])], [p], [1])
    # ... Vector Spaces
    V = StencilVectorSpace(cart, dtype=dtype)
    e = V.ends[0]
    s = V.starts[0]

    # Build banded matrix with 2p+1 diagonals: must be symmetric and positive definite
    # Here we assign value 2*p on main diagonal and -1 on other diagonals

    A = StencilMatrix(V, V)
    A[:, -p:0] = -matrix_data[0]
    A[:, 0:1] = matrix_data[1]
    A[:, 1:p + 1] = matrix_data[2]
    A.remove_spurious_entries()

    # Build exact solution
    xe = StencilVector(V)
    xe[s:e + 1] = xp.random.random(e + 1 - s)
    return(V, A, xe)


#===============================================================================
@pytest.mark.parametrize( 'n', [5, 10, 13] )
@pytest.mark.parametrize('p', [2, 3])
@pytest.mark.parametrize('dtype', [float])
@pytest.mark.parametrize('solver', ['cg', 'pcg', 'bicg', 'bicgstab', 'pbicgstab', 'minres', 'lsmr', 'gmres'])

def test_solver_tridiagonal(n, p, dtype, solver, verbose=False):

    #---------------------------------------------------------------------------
    # PARAMETERS
    #---------------------------------------------------------------------------

    if solver in ['bicg', 'bicgstab', 'pbicgstab', 'lsmr']:
        if dtype==complex:
            diagonals = [1-10j,6+9j,3+5j]
        else:
            diagonals = [1,6,3]
            
        if solver == 'pbicgstab' and dtype == complex:
            # pbicgstab only works for real matrices
            return
    elif solver == 'gmres':
        if dtype==complex:
            diagonals = [-7-2j,-6-2j,-1-10j]
        else:
            diagonals = [-7,-1,-3]

    if solver in ['cg', 'pcg', 'minres']:
        # pcg runs with Jacobi preconditioner
        V, A, xe = define_data_hermitian(n, p, dtype=dtype)
        if solver == 'minres' and dtype == complex:
            # minres only works for real matrices
            return
    else:
        V, A, xe = define_data(n, p, diagonals, dtype=dtype)

    # Tolerance for success: 2-norm of error in solution
    tol = 1e-8

    #---------------------------------------------------------------------------
    # TEST
    #---------------------------------------------------------------------------
    if verbose:
        # Title
        print()
        print( "="*80 )
        print( f"SERIAL TEST: solve linear system A*x = b using {solver}")
        print( "="*80 )
        print()

    #Create the solvers
    if solver in ['pcg', 'pbicgstab']:
        pc = A.diagonal(inverse=True)
        solv = inverse(A, solver, pc=pc, tol=1e-13, verbose=verbose, recycle=True)
    else:
        solv = inverse(A, solver, tol=1e-13, verbose=verbose, recycle=True)
    solvt = solv.transpose()
    solvh = solv.H
    solv2 = inverse(A@A, solver, tol=1e-13, verbose=verbose, recycle=True) # Test solver of composition of operators

    # Manufacture right-hand-side vector from exact solution
    be  = A @ xe
    be2 = A @ be # Test solver with consecutive solves
    bet = A.T @ xe
    beh = A.H @ xe

    # Solve linear system
    # Assert x0 got updated correctly and is not the same object as the previous solution, but just a copy
    x = solv @ be
    info = solv.get_info()
    solv_x0 = solv._options["x0"]
    assert xp.array_equal(x.toarray(), solv_x0.toarray())
    assert x is not solv_x0

    x2 = solv @ be2
    solv_x0 = solv._options["x0"]
    assert xp.array_equal(x2.toarray(), solv_x0.toarray())
    assert x2 is not solv_x0

    xt = solvt.solve(bet)
    solvt_x0 = solvt._options["x0"]
    assert xp.array_equal(xt.toarray(), solvt_x0.toarray())
    assert xt is not solvt_x0

    xh = solvh.dot(beh)
    solvh_x0 = solvh._options["x0"]
    assert xp.array_equal(xh.toarray(), solvh_x0.toarray())
    assert xh is not solvh_x0

    if solver != 'pcg':
        # PCG only works with operators with diagonal
        xc = solv2 @ be2
        solv2_x0 = solv2._options["x0"]
        assert xp.array_equal(xc.toarray(), solv2_x0.toarray())
        assert xc is not solv2_x0


    # Verify correctness of calculation: 2-norm of error
    b = A @ x
    b2 = A @ x2
    bt = A.T @ xt
    bh = A.H @ xh
    if solver != 'pcg':
        bc = A @ A @ xc

    err = b - be
    err_norm = xp.linalg.norm( err.toarray() )
    err2 = b2 - be2
    err2_norm = xp.linalg.norm( err2.toarray() )
    errt = bt - bet
    errt_norm = xp.linalg.norm( errt.toarray() )
    errh = bh - beh
    errh_norm = xp.linalg.norm( errh.toarray() )

    if solver != 'pcg': 
        errc = bc - be2
        errc_norm = xp.linalg.norm( errc.toarray() )

    #---------------------------------------------------------------------------
    # TERMINAL OUTPUT
    #---------------------------------------------------------------------------
    if verbose:
        print()
        print( 'A  =', A, sep='\n' )
        print( 'b  =', b )
        print( 'x  =', x )
        print( 'xe =', xe )
        print( 'info =', info )
        print()

        print( "-"*40 )
        print( f"2-norm of error in solution = {err_norm:.2e}" )
        if err_norm < tol:
            print( "PASSED" )
        else:
            print( "FAIL" )
        print( "-"*40 )

    #---------------------------------------------------------------------------
    # PYTEST
    #---------------------------------------------------------------------------
    # The lsmr solver does not consistently produce outputs x whose error ||Ax - b|| is less than tol.
    if solver != 'lsmr':
        assert err_norm < tol
        assert err2_norm < tol
        assert errt_norm < tol
        assert errh_norm < tol
        assert solver == 'pcg' or errc_norm < tol

#===============================================================================
def _compute_global_starts_ends(domain_decomposition, npts):
    # Same as feectools.linalg.tests.test_block.compute_global_starts_ends.
    global_starts = [None] * len(npts)
    global_ends = [None] * len(npts)
    for axis in range(len(npts)):
        ee = domain_decomposition.global_element_ends[axis]
        global_ends[axis] = ee.copy()
        global_ends[axis][-1] = npts[axis] - 1
        global_starts[axis] = xp.array([0] + (global_ends[axis][:-1] + 1).tolist())
    return global_starts, global_ends


@pytest.mark.parametrize('n1', [8, 16])
@pytest.mark.parametrize('n2', [8, 12])
@pytest.mark.parametrize('p1', [1, 2])
@pytest.mark.parallel
def test_direct_solver_parallel(n1, n2, p1, verbose=False):
    """`DirectSolver` at nprocs > 1 must recover the exact solution, same as serial."""
    p2 = 1

    comm = MPI.COMM_WORLD
    D = DomainDecomposition([n1, n2], periods=[False, False], comm=comm)
    npts = [n1, n2]
    global_starts, global_ends = _compute_global_starts_ends(D, npts)
    cart = CartDecomposition(D, npts, global_starts, global_ends, pads=[p1, p2], shifts=[1, 1])

    V = StencilVectorSpace(cart, dtype=float)
    A = StencilMatrix(V, V)

    # Diagonally dominant (hence nonsingular) stencil: -1 on every off-diagonal, enough
    # on the main diagonal to dominate the row sum -- same style as define_data_hermitian
    # above, extended to 2D.
    n_offdiag = (2 * p1 + 1) * (2 * p2 + 1) - 1
    for k1 in range(-p1, p1 + 1):
        for k2 in range(-p2, p2 + 1):
            A[:, :, k1, k2] = 0.0 if (k1 == 0 and k2 == 0) else -1.0
    A[:, :, 0, 0] = n_offdiag + 1.0
    A.remove_spurious_entries()

    s1, s2 = V.starts
    e1, e2 = V.ends
    xe = StencilVector(V)
    for i1 in range(s1, e1 + 1):
        for i2 in range(s2, e2 + 1):
            xe[i1, i2] = xp.random.random()
    xe.update_ghost_regions()

    be = A @ xe

    solv = DirectSolver(A)
    x = solv.solve(be)

    err_norm = xp.linalg.norm((x - xe).toarray())
    if verbose:
        print(f"n1={n1} n2={n2} p1={p1} p2={p2} nprocs={comm.Get_size()} err_norm={err_norm:.2e}")
    assert err_norm < 1e-9

# ===============================================================================
# SCRIPT FUNCTIONALITY
#===============================================================================

if __name__ == "__main__":
    import sys
    pytest.main( sys.argv )
