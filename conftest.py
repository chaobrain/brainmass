"""Pytest configuration and shared fixtures for the brainmass test suite.

Forces matplotlib onto the non-interactive ``Agg`` backend so the demo tests
(``test_demo_*``, which default to ``plot=True`` and call ``plt.show()``) never
open a blocking GUI window. Under ``Agg`` the ``plt.show()`` call is a
non-blocking no-op, so ``pytest brainmass/`` runs unattended in CI and headless
shells instead of hanging until each figure window is closed by hand.

Also provides reusable fixtures (goal-04 testing rigor):

``_isolate_environ_dt``
    Autouse. Snapshots and restores the global ``brainstate.environ`` ``dt`` so a
    test that calls ``environ.set(dt=...)`` cannot leak its time step into later,
    unrelated tests.
``dt``
    Sets a known integration time step for the duration of a test (parametrizable
    via ``indirect=True``) and yields it.
``seeded``
    Seeds ``brainstate.random`` and ``numpy.random`` for deterministic stochastic
    tests (parametrizable via ``indirect=True``) and yields the seed.
``connectome``
    A small structural-connectivity (``SC``) + distance/delay fixture for
    whole-brain coupling tests.
"""

import matplotlib

# Must run before any ``import matplotlib.pyplot`` performed by the test
# modules. pytest imports conftest.py before it collects/imports the test
# files, so selecting the backend here guarantees pyplot binds to ``Agg``.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import brainstate  # noqa: E402
import brainunit as u  # noqa: E402

#: Default integration time step used by the ``dt`` fixture.
DEFAULT_DT = 0.1 * u.ms

#: Default RNG seed used by the ``seeded`` fixture.
DEFAULT_SEED = 0

_UNSET = object()


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close every open figure after each test.

    Yields control to the test, then closes all figures so a long session does
    not trip matplotlib's "More than 20 figures have been opened" warning or
    leak figure memory across tests.
    """
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def _isolate_environ_dt():
    """Prevent the global ``environ`` time step from leaking across tests.

    Many tests call ``brainstate.environ.set(dt=...)`` at module or function
    scope. Because ``environ`` is process-global, a value set by one test would
    otherwise persist and silently change the behaviour of a later test that
    forgot to set its own ``dt``. This autouse fixture snapshots ``dt`` before
    the test and restores (or removes) it afterwards.
    """
    prev = brainstate.environ.get("dt", _UNSET)
    try:
        yield
    finally:
        if prev is _UNSET:
            # ``dt`` was unset before the test; drop anything the test set.
            brainstate.environ.pop("dt", None)
        else:
            brainstate.environ.set(dt=prev)


@pytest.fixture
def dt(request):
    """Set a known integration time step for the test and yield it.

    Parameters
    ----------
    request : pytest.FixtureRequest
        If parametrized with ``indirect=True``, ``request.param`` supplies the
        time step (a ``brainunit`` quantity with time units). Otherwise
        :data:`DEFAULT_DT` is used.

    Yields
    ------
    brainunit.Quantity
        The time step that was installed into ``brainstate.environ``.

    Examples
    --------
    .. code-block:: python

        >>> def test_uses_default_dt(dt):
        ...     assert dt is not None
    """
    value = getattr(request, "param", DEFAULT_DT)
    brainstate.environ.set(dt=value)
    yield value
    # Cleanup is handled by ``_isolate_environ_dt``.


@pytest.fixture
def seeded(request):
    """Seed the global RNGs for a deterministic stochastic test.

    Seeds both ``brainstate.random`` (the JAX-backed RNG used by the noise
    processes and initializers) and ``numpy.random`` (used by a few tests to
    build random connectivity).

    Parameters
    ----------
    request : pytest.FixtureRequest
        If parametrized with ``indirect=True``, ``request.param`` supplies the
        integer seed. Otherwise :data:`DEFAULT_SEED` is used.

    Yields
    ------
    int
        The seed that was applied.
    """
    seed = int(getattr(request, "param", DEFAULT_SEED))
    brainstate.random.seed(seed)
    np.random.seed(seed)
    yield seed


@pytest.fixture
def connectome():
    """Provide a small, deterministic connectome for coupling tests.

    Returns
    -------
    dict
        Dictionary with keys:

        ``n``
            Number of regions (``int``).
        ``SC``
            ``(n, n)`` non-negative structural-connectivity matrix with a zero
            diagonal, row-normalised so each row sums to one (``jax.Array``).
        ``dist``
            ``(n, n)`` symmetric inter-region distance matrix with a zero
            diagonal (``jax.Array``).
        ``delay``
            ``(n, n)`` conduction delay matrix (``dist`` scaled to milliseconds,
            a ``brainunit`` quantity).

    Notes
    -----
    Values are fixed (not random) so tests built on them are reproducible
    without needing a seed.
    """
    n = 4
    sc = np.array(
        [
            [0.0, 0.8, 0.2, 0.1],
            [0.8, 0.0, 0.5, 0.3],
            [0.2, 0.5, 0.0, 0.6],
            [0.1, 0.3, 0.6, 0.0],
        ],
        dtype=np.float32,
    )
    sc = sc / sc.sum(axis=1, keepdims=True)  # row-normalise
    dist = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.5, 2.0],
            [2.0, 1.5, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return {
        "n": n,
        "SC": u.math.asarray(sc),
        "dist": u.math.asarray(dist),
        "delay": u.math.asarray(dist) * u.ms,
    }
