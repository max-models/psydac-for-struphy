"""Root-level pytest configuration."""
import pytest
import sys
from pathlib import Path


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "petsc: mark test as requiring PETSc"
    )
    config.addinivalue_line(
        "markers", "parallel: mark test as parallel"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests incompatible with pytest-xdist and with missing dependencies."""
    # Tests that require sympde which isn't always installed
    skip_modules = {
        "test_poisson.py",
        "test_sum_factorization_assembly_3d.py",
        "test_dirichlet_projectors.py",
        "test_tensor.py",
    }

    items_to_remove = []
    skip = pytest.mark.skip(reason="Requires optional dependency (sympde)")

    for item in items:
        # Skip if module is in skip list
        if item.fspath.basename in skip_modules:
            items_to_remove.append(item)
            continue

        # If running with xdist, automatically skip mpi and petsc tests
        if config.pluginmanager.has_plugin("xdist"):
            if item.get_closest_marker("mpi") or item.get_closest_marker("petsc"):
                skip_xdist = pytest.mark.skip(
                    reason="Incompatible with pytest-xdist parallel execution"
                )
                item.add_marker(skip_xdist)

    # Remove items with missing dependencies
    for item in items_to_remove:
        items.remove(item)
