Contributing Guide
==================

Thank you for contributing to ``brainmass``!


Ways to Contribute
-------------------

- **Code:** New models, features, bug fixes
- **Documentation:** Tutorials, API docs, examples
- **Examples:** Jupyter notebooks demonstrating use cases
- **Bug reports:** Issue reports with reproducible examples
- **Feature requests:** Proposals for new functionality


Development Setup
-----------------

1. Fork the repository on GitHub
2. Clone your fork:

   .. code-block:: bash

      git clone https://github.com/YOUR_USERNAME/brainmass.git
      cd brainmass

3. Install in development mode:

   .. code-block:: bash

      pip install -e .[dev,doc]

4. Create a branch:

   .. code-block:: bash

      git checkout -b feature/my-feature


Code Guidelines
---------------

**Style:**

- Follow PEP 8
- Use type hints
- Write descriptive variable names
- Keep functions focused and small

**Documentation:**

- Add docstrings to all public functions/classes
- Use Google-style docstrings
- Include examples in docstrings
- Update relevant tutorial/API docs

**Testing:**

- Write tests for new features
- Ensure existing tests pass
- Aim for >80% code coverage


Pull Request Process
---------------------

1. **Before submitting:**
   - Run tests: ``pytest tests/``
   - Check code style: ``ruff check brainmass/``
   - Build docs: ``make -C docs html``
   - Update CHANGELOG

2. **PR description:**
   - Describe what changed and why
   - Reference related issues
   - Include examples if applicable

3. **Review process:**
   - Maintainers will review within a week
   - Address feedback
   - Once approved, PR will be merged


Testing
-------

Run all tests:

.. code-block:: bash

   pytest tests/

Run specific test:

.. code-block:: bash

   pytest tests/test_models.py::test_hopf_oscillator


Documentation
-------------

Build documentation locally:

.. code-block:: bash

   cd docs
   make html
   # Open _build/html/index.html

Add new documentation to appropriate section (API, tutorials, examples).


Commit Messages
---------------

Use clear, descriptive commit messages:

- Start with verb (Add, Fix, Update, etc.)
- Keep first line under 72 characters
- Add details in body if needed

Good examples:

.. code-block:: text

   Add QIF model implementation

   Fix coupling bug in DiffusiveCoupling

   Update installation docs for Windows


Reporting Issues
----------------

Good bug reports include:

1. Minimal reproducible example
2. Expected vs actual behavior
3. Environment info (OS, Python version, brainmass version)
4. Error messages/tracebacks


Feature Requests
----------------

Feature requests should:

1. Describe the use case
2. Explain why it's valuable
3. Suggest implementation approach (optional)
4. Consider backward compatibility


Code Review
-----------

When reviewing:

- Be constructive and respectful
- Focus on code, not person
- Suggest improvements, don't demand
- Approve when ready


Release Process
---------------

(For maintainers)

1. Update version in ``pyproject.toml``
2. Update ``CHANGELOG.md``
3. Create release tag
4. Build and publish to PyPI


Getting Help
------------

- **Questions:** GitHub Discussions
- **Stuck:** Ask in your PR
- **Other:** Email maintainers


Thank You!
----------

Every contribution helps make ``brainmass`` better for everyone.
