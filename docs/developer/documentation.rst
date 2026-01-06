Documentation
=============

Guidelines for writing and building documentation.


Documentation Structure
-----------------------

.. code-block:: text

   docs/
   ├── index.rst               # Landing page
   ├── api/                    # API reference
   │   ├── index.rst
   │   ├── models.rst
   │   └── ...
   ├── tutorials/              # User tutorials
   │   ├── index.rst
   │   ├── quickstart.rst
   │   └── ...
   ├── examples/               # Example notebooks
   │   ├── index.rst
   │   └── ...
   ├── developer/              # Developer guide
   └── conf.py                 # Sphinx configuration


Building Documentation
----------------------

Local build:

.. code-block:: bash

   cd docs
   make html

   # Open in browser
   open _build/html/index.html  # macOS
   # or
   start _build/html/index.html  # Windows


Clean rebuild:

.. code-block:: bash

   make clean
   make html


Docstring Format
----------------

Use Google-style docstrings:

.. code-block:: python

   def my_function(param1, param2, param3=None):
       \"\"\"Brief one-line description.

       Longer description with more details if needed.
       Can span multiple paragraphs.

       Args:
           param1: Description of param1
           param2: Description of param2 with units (Hz)
           param3: Optional parameter, defaults to None

       Returns:
           Description of return value with shape and units

       Raises:
           ValueError: When param1 is negative

       Examples:

           >>> result = my_function(1.0, 2.0)
           >>> print(result)
           3.0

       References:
           Author et al. (2020). Paper Title. Journal.
       \"\"\"


Writing Tutorials
-----------------

Tutorial structure:

1. Clear objectives
2. Prerequisites
3. Step-by-step instructions
4. Complete code examples
5. Exercises or next steps


Example Tutorial Template
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: rst

   Tutorial Title
   ==============

   Learn how to [objective].

   Prerequisites
   -------------
   - Knowledge of [topic]
   - Completed [previous tutorial]

   Overview
   --------
   This tutorial covers:
   1. Topic 1
   2. Topic 2
   3. Topic 3

   Step 1: [Title]
   ---------------

   Explanation...

   .. code-block:: python

      # Code example
      import brainmass
      model = brainmass.HopfOscillator(in_size=10)

   [More steps...]

   Next Steps
   ----------
   - Try [exercise]
   - Read [related topic]


Adding Examples
---------------

Create Jupyter notebook in ``examples/``:

1. Name format: ``###-descriptive-name.ipynb``
2. Include clear markdown cells
3. Add to appropriate gallery index
4. Ensure it runs without errors


Documentation Style
-------------------

**Voice:**

- Use second person ("you")
- Active voice preferred
- Clear and concise

**Formatting:**

- Code blocks for all code
- Inline code for variables/functions
- Bold for UI elements
- Italics for emphasis (sparingly)

**Links:**

- Cross-reference related docs
- Link to API documentation
- External links where helpful


See Also
--------

- Sphinx documentation
- MyST Markdown guide
- :doc:`contributing` - Contribution process
