"""Pytest configuration for the brainmass test suite.

Forces matplotlib onto the non-interactive ``Agg`` backend so the demo tests
(``test_demo_*``, which default to ``plot=True`` and call ``plt.show()``) never
open a blocking GUI window. Under ``Agg`` the ``plt.show()`` call is a
non-blocking no-op, so ``pytest brainmass/`` runs unattended in CI and headless
shells instead of hanging until each figure window is closed by hand.
"""

import matplotlib

# Must run before any ``import matplotlib.pyplot`` performed by the test
# modules. pytest imports conftest.py before it collects/imports the test
# files, so selecting the backend here guarantees pyplot binds to ``Agg``.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)
import pytest


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close every open figure after each test.

    Yields control to the test, then closes all figures so a long session does
    not trip matplotlib's "More than 20 figures have been opened" warning or
    leak figure memory across tests.
    """
    yield
    plt.close("all")
