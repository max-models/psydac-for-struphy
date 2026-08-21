from dataclasses import dataclass
from time import time
from typing import TYPE_CHECKING


# Might not be needed
class MPICommWrapper:
    def __init__(self, use_mpi=True):
        self.use_mpi = use_mpi
        if use_mpi:
            from mpi4py import MPI

            self.comm = MPI.COMM_WORLD
        else:
            self.comm = MockComm()

    def __getattr__(self, name):
        return getattr(self.comm, name)


class MockComm:
    def __getattr__(self, name):
        # Return a function that does nothing and returns None
        def dummy(*args, **kwargs):
            return None

        return dummy

    # Override some functions
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def Barrier(self):
        return


class MPIwrapper:
    def __init__(
        self,
        use_mpi: bool = False,
        verbose: bool = False,
    ):
        self.use_mpi = use_mpi
        if use_mpi:
            from mpi4py import MPI

            self._MPI = MPI
            if verbose:
                print("MPI is enabled")
        else:
            self._MPI = MockMPI()
            if verbose:
                print("MPI is NOT enabled")

    @property
    def MPI(self):
        return self._MPI


class MockMPI:
    def __getattr__(self, name):
        # Return a function that does nothing and returns None
        def dummy(*args, **kwargs):
            return None

        return dummy

    # Override some functions
    @property
    def COMM_WORLD(self):
        return MockComm()

    # def comm_Get_rank(self):
    #     return 0

    # def comm_Get_size(self):
    #     return 1


import os

def _enabled(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ('', '0', 'false', 'no')


try:
    # CuPy arrays implement ``__cuda_array_interface__``, which mpi4py can
    # pass to CUDA-aware MPI implementations.  The DDM exchangers synchronize
    # the current CUDA stream before every MPI operation, and ranks bind to a
    # node-local GPU, so MPI is supported on the CuPy backend just as it is on
    # NumPy.  An MPI implementation must of course have CUDA support for
    # device-buffer communication.
    #
    # FEECTOOLS_DISABLE_MPI=1 remains available to force the serial path on
    # any backend.
    if _enabled('FEECTOOLS_DISABLE_MPI'):
        raise ImportError('MPI disabled by FEECTOOLS_DISABLE_MPI')

    from mpi4py import MPI

    _comm = MPI.COMM_WORLD
    # rank = _comm.Get_rank()
    # size = _comm.Get_size()
    mpi_enabled = True
except ImportError:
    # mpi4py not installed, or disabled on purpose
    mpi_enabled = False
except Exception:
    # mpi4py installed but not running under mpirun
    mpi_enabled = False

# TODO: add environment variable for mpi use
mpi_wrapper = MPIwrapper(
    use_mpi=mpi_enabled,
    verbose=False,
)

# TYPE_CHECKING is True when type checking (e.g., mypy), but False at runtime.
if TYPE_CHECKING:
    from mpi4py import MPI

    mpi = MPI
else:
    mpi = mpi_wrapper.MPI
