Observation Models
==================

.. currentmodule:: brainmass

Observation models map region-level neural activity to a *measurable* signal by
post-processing a simulated trajectory. ``brainmass`` ships two complementary
observation paths:

1. **Convolution BOLD** -- :class:`HRFBold` convolves (temporally-averaged) neural
   activity with a closed-form hemodynamic response function (HRF) kernel and
   decimates to the fMRI repetition time (TR). It is a single linear operator that
   is **differentiable in its few scalar parameters**, so it is the BOLD path of
   choice for *fitting*.
2. **Temporal averaging** -- :class:`TemporalAverage` is a standalone, anti-aliased
   downsampler (block mean over non-overlapping windows). It is the averaging
   complement to the point-decimation ``Simulator(sample_every=k)`` subsampling and
   the downsampler :class:`HRFBold` uses internally.

The biophysical Balloon-Windkessel :class:`BOLDSignal` ODE and the EEG/MEG
lead-field forwards live on :doc:`forward`; pick :class:`BOLDSignal` over
:class:`HRFBold` when biophysical realism (flow, volume, deoxyhemoglobin) matters
more than speed or differentiability.


HRF kernels
-----------

A *hemodynamic response function* (HRF) is the impulse response of the
neural-to-BOLD transform. Each kernel below is a callable instance --
``h = kernel(t)`` returns a **dimensionless** :math:`h(t)` (``t`` is a time
:class:`~brainunit.Quantity`, or a plain array interpreted as milliseconds, the TVB
convention). They differ only in their closed form:

.. list-table::
   :widths: 32 68
   :header-rows: 1

   * - Kernel
     - Closed form / reference
   * - :class:`FirstOrderVolterraHRFKernel`
     - TVB canonical first-order Volterra (underdamped damped oscillator;
       Friston et al. 2000)
   * - :class:`GammaHRFKernel`
     - Peak-normalised gamma probability density (Boynton et al. 1996)
   * - :class:`DoubleExponentialHRFKernel`
     - Difference of damped oscillations (Polonsky et al. 2000)
   * - :class:`MixtureOfGammasHRFKernel`
     - Difference of two gammas / SPM canonical HRF (Glover 1999)

:class:`HRFKernel` is the shared abstract base; subclass it to add a new kernel.


API Reference
-------------

HRF kernel family
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   HRFKernel
   FirstOrderVolterraHRFKernel
   GammaHRFKernel
   DoubleExponentialHRFKernel
   MixtureOfGammasHRFKernel


Convolution BOLD
^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   HRFBold

:class:`HRFBold` downsamples the neural trajectory (via :class:`TemporalAverage`),
convolves it with the chosen HRF kernel, and decimates to the TR:

.. math::

   \text{BOLD} = k_1 V_0 \,(\,h * y_{\text{ds}} - 1\,),

where :math:`y_{\text{ds}}` is the temporally-averaged neural activity, :math:`h`
the HRF kernel and :math:`*` 1-D convolution along time.


Temporal averaging
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   TemporalAverage

:class:`TemporalAverage` downsamples a trajectory by averaging over non-overlapping
windows of ``w = round(period / dt)`` samples
(``y[k] = mean(signal[k*w : (k+1)*w])``; the trailing partial window is dropped). It
preserves units and is smoother / anti-aliased relative to point decimation.


Examples
--------

A closed-form HRF kernel is a callable returning a dimensionless impulse response:

.. doctest::

   >>> import brainmass
   >>> import brainunit as u
   >>> import jax.numpy as jnp
   >>> kernel = brainmass.MixtureOfGammasHRFKernel()
   >>> t = jnp.arange(0., 20000., 100.)        # 20 s grid in ms
   >>> h = kernel(t)
   >>> h.shape
   (200,)
   >>> bool(jnp.all(jnp.isfinite(h)))
   True

Convolve a slow neural drive to a BOLD trace, then average a signal down a second
way:

.. doctest::

   >>> t = jnp.arange(2000.)
   >>> z = 1.0 + 0.5 * jnp.sin(2 * jnp.pi * t[:, None] / 800.0)   # (2000, 1) drive
   >>> bold = brainmass.HRFBold(
   ...     period=200. * u.ms, downsample_period=4. * u.ms,
   ...     kernel=brainmass.FirstOrderVolterraHRFKernel(duration=400. * u.ms),
   ... )
   >>> y = bold(z, dt=1. * u.ms)
   >>> y.shape[1]
   1
   >>> signal = jnp.arange(20.).reshape(20, 1)
   >>> brainmass.TemporalAverage(period=5. * u.ms)(signal, dt=1. * u.ms).shape
   (4, 1)


See Also
--------

- :doc:`forward` -- the Balloon-Windkessel :class:`BOLDSignal` ODE and the EEG/MEG
  lead-field forwards.
- :doc:`orchestration` -- ``Simulator(sample_every=k)`` point-decimation subsampling
  and the FCD-distribution objectives that consume BOLD traces.
- :doc:`models` -- the neural mass models whose activity these observers transform.
