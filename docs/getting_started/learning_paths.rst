Learning Paths
==============

``brainmass`` serves three kinds of users. Each persona below is a signposted route
through the documentation — follow the one that matches your goal. The paths overlap by
design: a beginner who collects data becomes a researcher; a researcher who extends a
model becomes a modeler.


Beginner
--------

*New to neural mass models and/or brainmass. You want to run a first simulation and build
the mental model.*

1. :doc:`installation` — install brainmass for CPU, GPU, or TPU.
2. :doc:`/getting_started/quickstart` — a five-minute first simulation with the ``Simulator`` and a
   one-line ``Fitter`` teaser.
3. :doc:`key_concepts` — the mental model: Step-model, ``Simulator``, ``Network``,
   ``Fitter``, and units.
4. :doc:`../tutorials/01_first_simulation`, :doc:`../tutorials/02_models_and_dynamics`,
   :doc:`../tutorials/03_noise`, :doc:`../tutorials/04_building_a_network` — the core
   tutorial sequence.
5. :doc:`../gallery/index` — browse the **model zoo** to see each model family in action.


Researcher
----------

*You have empirical data (EEG / MEG / fMRI) and want to fit models to it, map activity to
signals, and analyze the results.*

1. :doc:`/getting_started/quickstart` — get oriented with the orchestration layer.
2. :doc:`../tutorials/05_forward_models` — turn neural activity into BOLD / EEG / MEG
   signals.
3. :doc:`../tutorials/06_fitting_with_gradients` and
   :doc:`../tutorials/07_gradient_free_fitting` — fit models to data with gradient-based
   and gradient-free backends.
4. :doc:`../howto/analyze_results` — functional connectivity, FCD, and power spectra.
5. :doc:`../howto/parameter_sweeps` — sweep parameters efficiently with ``vmap``.
6. :doc:`../gallery/index` — work through the **case studies** for end-to-end examples.


Modeler
-------

*You build and extend models, couplings, and objectives, and run differentiable /
data-driven workflows.*

1. :doc:`../concepts/why_differentiable` and :doc:`../concepts/architecture_overview` —
   the narrative and the package architecture.
2. :doc:`../howto/custom_coupling`, :doc:`../howto/custom_objective`, and
   :doc:`../howto/batch_and_accelerate` — extend the building blocks and make them fast.
3. :doc:`../developer/index` — the developer guides, including the data-driven workflow
   extension playbook.
