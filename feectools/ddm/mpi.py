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
    # MPI is off by default on the CuPy backend, and on by default otherwise.
    #
    # It is no longer *incorrect* to combine the two -- the reductions in
    # feectools.linalg stage their (tiny) buffers through the host, the ghost
    # exchangers synchronize the device before handing it a buffer, and each
    # rank binds to its own GPU. It is, however, still slow: a ghost exchange
    # of device memory through MPI derived datatypes costs milliseconds, so a
    # single-GPU run pays several times over for communication it does not
    # need. Until that is addressed, opt in explicitly:
    #
    #     FEECTOOLS_ENABLE_MPI=1     use MPI on the CuPy backend
    #     FEECTOOLS_DISABLE_MPI=1    force the serial path on any backend
    if _enabled('FEECTOOLS_DISABLE_MPI'):
        raise ImportError('MPI disabled by FEECTOOLS_DISABLE_MPI')

    if os.environ.get('ARRAY_BACKEND', '').lower() == 'cupy' \
            and not _enabled('FEECTOOLS_ENABLE_MPI'):
        raise ImportError('MPI off by default on the CuPy backend; '
                          'set FEECTOOLS_ENABLE_MPI=1 to use it')

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
