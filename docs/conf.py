# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# a_list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys

sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'brainmass'
copyright = '2024-2026, brainmass'
author = 'BrainX Ecosystem'

# The full version, including alpha/beta/rc tags
import brainmass

release = brainmass.__version__

from highlight_lexer import fix_ipython2_lexer_in_notebooks

# The gallery of example notebooks now lives in-tree under ``docs/gallery/`` (goal-13a),
# split into ``model_zoo/`` and ``case_studies/``. The build no longer copies the flat root
# ``examples/`` tree in (goal-13h migrates the real example notebooks into the gallery). We
# still run the ipython2 -> ipython3 lexer fix-up over every gallery notebook so their
# embedded outputs highlight correctly. ``fix_ipython2_lexer_in_notebooks`` globs a single
# directory (non-recursive), so apply it to each gallery sub-directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _sub in ('gallery/model_zoo', 'gallery/case_studies'):
    _dir = os.path.join(_HERE, _sub)
    if os.path.isdir(_dir):
        fix_ipython2_lexer_in_notebooks(_dir)

import shutil

shutil.copy('../changelog.md', './changelog.md')

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'myst_nb',
    'matplotlib.sphinxext.plot_directive',
    'sphinx_thebe',
    'sphinx_design',
    'sphinx_math_dollar',
    'brainx_sphinx_header',
]


html_baseurl = 'https://brainx.chaobrain.com/brainmass/'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = ['.rst', '.ipynb', '.md']

# source_suffix = '.rst'
autosummary_generate = True
autosummary_generate_overwrite = False

# Autodoc settings for better API documentation
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__, default_rng',
}

# The master toctree document.
master_doc = 'index'

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.13", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
}
nitpick_ignore = [
    ("py:class", "docutils.nodes.document"),
    ("py:class", "docutils.parsers.rst.directives.body.Sidebar"),
]

suppress_warnings = ["myst.domains", "ref.ref"]

numfig = True

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
]
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# ``roadmap.md`` is the internal improvement-program roadmap (an engineering artifact, not
# part of the published navigation); excluding it keeps it out of the build and avoids an
# "isn't included in any toctree" warning. The data-driven *pillar* roadmap lives at
# ``data_driven/roadmap.md`` and IS published.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'roadmap.md', '**.ipynb_checkpoints']

html_theme = "sphinx_book_theme"
html_logo = "https://brainx.chaobrain.com/images/brainmass.webp"
html_title = "brainmass"
html_copy_source = True
html_sourcelink_suffix = ""
html_favicon = "https://brainx.chaobrain.com/images/brainmass.webp"
html_last_updated_fmt = ""

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Notebooks (the ``tutorials/``, ``howto/``, ``concepts/``, ``developer/`` guides plus the
# ``gallery/`` model-zoo and case-study demos) are NOT executed at build time, on purpose: each
# carries embedded outputs and several are large. The self-contained guide notebooks are executed
# once at authoring time (so their outputs are real); documentation is kept honest by
# ``sphinx.ext.doctest`` (docstring ``>>>`` examples, run in CI by goal-03). (The legacy flat
# ``examples/`` tree was retired in goal-13m; its demos now live under ``docs/gallery/``.)
nb_execution_mode = "off"          # myst-nb >= 0.13 (current)
jupyter_execute_notebooks = "off"  # legacy alias, kept for older myst-nb
thebe_config = {
    "repository_url": "https://github.com/binder-examples/jupyter-stacks-datascience",
    "repository_branch": "master",
}

html_theme_options = {
    'show_toc_level': 2,
}

# -- Options for doctest (sphinx.ext.doctest) ----------------------
#
# ``make doctest`` (and goal-03's CI job) executes every docstring ``>>>`` example and any
# remaining ``.. testcode::`` block, so documented API usage cannot drift unnoticed. (The narrative
# tutorials are now executed Jupyter notebooks rather than ``.. testcode::`` pages.) The setup below
# is prepended to each doctest group, giving examples the common imports and a default integration
# step without per-example boilerplate. Examples needing a different ``dt`` (e.g. the dimensionless
# ``BOLDSignal``) reset it locally.
doctest_global_setup = """
import matplotlib
matplotlib.use("Agg")  # headless: example ``plt.show()`` calls must not block the build
import brainmass
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
brainstate.environ.set(dt=0.1 * u.ms)
"""

# Test bare ``>>>`` doctest blocks (the NumPy-doc ``Examples`` sections) in addition to the
# explicit ``.. testcode::`` / ``.. doctest::`` directives.
doctest_test_doctest_blocks = "default"
