"""Root-level pytest configuration."""

def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "petsc: mark test as requiring PETSc"
    )
    config.addinivalue_line(
        "markers", "parallel: mark test as parallel"
    )
