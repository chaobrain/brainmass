# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

from typing import Optional, Union, Callable

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp

from ._typing import Initializer

Array = jax.Array
Quantity = u.Quantity

__all__ = [
    'BOLDSignal',
    'LeadFieldModel',
    'EEGLeadFieldModel',
    'MEGLeadFieldModel',
]


class BOLDSignal(brainstate.nn.Dynamics):
    r"""
    Balloon-Windkessel hemodynamic model of Friston et al. (2003) [1]_.

    The Balloon-Windkessel model describes the coupling of perfusion to BOLD signal, with a
    dynamical model of the transduction of neuronal activity into perfusion changes. The
    model assumes that the BOLD signal is a static nonlinear function of the normalized total
    deoxyhemoglobin voxel content, normalized venous volume, resting net oxygen extraction
    fraction by the capillary bed, and resting blood volume fraction. The BOLD-signal estimation
    for each brain area is computed by the level of synaptic activity in that particular cortical
    area, noted $z_i$ for a given cortical are $i$.

    For the i-th region, synaptic activity $z_i$ causes an increase in a vasodilatory signal $x_i$
    that is subject to autoregulatory feedback. Inflow $f_i$ responds in proportion to this signal
    with concomitant changes in blood volume $v_i$ and deoxyhemoglobin content $q_i$. The equations
    relating these biophysical processes are as follows:

    $$
    \begin{gathered}
    \dot{x}_i=z_i-k_i x_i-\gamma_i\left(f_i-1\right) \\
    \dot{f}_i=x_i \\
    \tau_i \dot{v}_i=f_i-v_i^{1 / \alpha} \\
    \tau_i \dot{q}_i=\frac{f_i}{\rho}\left[1-(1-\rho)^{1 / f_i}\right]-q_i v_i^{1 / \alpha-1},
    \end{gathered}
    $$

    where $\rho$ is the resting oxygen extraction fraction. The BOLD signal is given by the following:

    $$
    \mathrm{BOLD}_i=V_0\left[k_1\left(1-q_i\right)+k_2\left(1-q_i / v_i\right)+k_3\left(1-v_i\right)\right],
    $$

    where $V_0 = 0.02, k1 = 7\rho, k2 = 2$, and $k3 = 2\rho − 0.2$. All biophysical parameters were taken
    as in Friston et al. (2003) [1]_. The BOLD model converts the local synaptic activity of a given cortical
    area into an observable BOLD signal and does not actively couple the signals from other cortical areas.

    Parameters
    ----------
    in_size : int
        Size of the input vector (number of brain regions).
    gamma : float or callable, optional
        Rate of signal decay (default is 0.41).
    k : float or callable, optional
        Rate of flow-dependent elimination (default is 0.65).
    alpha : float or callable, optional
        Grubb's exponent (default is 0.32).
    tau : float or callable, optional
        Hemodynamic transit time (default is 0.98).
    rho : float or callable, optional
        Resting oxygen extraction fraction (default is 0.34).
    V0 : float, optional
        Resting blood volume fraction (default is 0.02).

    References
    ----------
    .. [1] Friston KJ, Harrison L, Penny W (2003) Dynamic causal modelling. Neuroimage 19:1273–1302,
           doi:10.1016/S1053-8119(03)00202-7
    """

    def __init__(
        self,
        in_size,
        gamma: Initializer = 0.41,
        k: Initializer = 0.65,
        alpha: Initializer = 0.32,
        tau: Initializer = 0.98,
        rho: Initializer = 0.34,
        V0: float = 0.02,
        init: Callable = braintools.init.Constant(1.),
    ):
        super().__init__(in_size)

        self.gamma = braintools.init.param(gamma, self.varshape)
        self.k = braintools.init.param(k, self.varshape)
        self.alpha = braintools.init.param(alpha, self.varshape)
        self.tau = braintools.init.param(tau, self.varshape)
        self.rho = braintools.init.param(rho, self.varshape)

        self.V0 = V0
        self.k1 = 7 * self.rho
        self.k2 = 2.
        self.k3 = 2 * self.rho - 0.2

        self.init = init

    def init_state(self, batch_size=None, **kwargs):
        self.x = brainstate.HiddenState(braintools.init.param(self.init, self.varshape, batch_size))
        self.f = brainstate.HiddenState(braintools.init.param(self.init, self.varshape, batch_size))
        self.v = brainstate.HiddenState(braintools.init.param(self.init, self.varshape, batch_size))
        self.q = brainstate.HiddenState(braintools.init.param(self.init, self.varshape, batch_size))

    def reset_state(self, batch_size=None, **kwargs):
        self.x.value = braintools.init.param(self.init, self.varshape, batch_size)
        self.f.value = braintools.init.param(self.init, self.varshape, batch_size)
        self.v.value = braintools.init.param(self.init, self.varshape, batch_size)
        self.q.value = braintools.init.param(self.init, self.varshape, batch_size)

    def derivative(self, y, t, z):
        x, f, v, q = y
        dx = z - self.k * x - self.gamma * (f - 1)
        df = x
        dv = (f - u.math.power(v, 1 / self.alpha)) / self.tau
        E = 1 - u.math.power(1 - self.rho, 1 / f)
        dq = (f * E / self.rho - u.math.power(v, 1 / self.alpha) * q / v) / self.tau
        return dx, df, dv, dq

    def update(self, z):
        x, f, v, q = braintools.quad.ode_rk2_step(
            self.derivative, (self.x.value, self.f.value, self.v.value, self.q.value), 0., z
        )
        self.x.value = x
        self.f.value = f
        self.v.value = v
        self.q.value = q

    def bold(self):
        return self.V0 * (self.k1 * (1 - self.q.value) +
                          self.k2 * (1 - self.q.value / self.rho) +
                          self.k3 * (1 - self.v.value))


class LeadFieldModel(brainstate.nn.Module):
    r"""
    Lead-field model.

    The lead-field matrix is a mathematical representation that describes the relationship
    between the electrical activity generated by neural sources in the brain and the resulting
    electrical measurements observed on the scalp's surface during EEG recordings.

    A **differentiable, unit-safe forward (lead-field) model** that maps Neural Mass Model (NMM)
    region-level outputs to EEG/MEG sensor space:

    .. math::

        \mathbf{y}(t) \;=\; \mathbf{s}(t) \, \mathbf{L} \;+\; \boldsymbol{\varepsilon}(t),

    where

    - :math:`\mathbf{s}(t)\in\mathbb{R}^{R}` is the **equivalent current dipole moment** (ECD) per region
      (typically in **nA·m**), obtained from NMM observables via a scale/biophysical mapping.
    - :math:`\mathbf{L}\in\mathbb{R}^{R\times M}` is the (region-aggregated) **lead field** matrix
      (EEG unit: **V/(nA·m)**, MEG unit: **T/(nA·m)**).
    - :math:`\mathbf{y}(t)\in\mathbb{R}^{M}` are the sensor measurements (EEG in **V**, MEG in **T**).
    - :math:`\boldsymbol{\varepsilon}(t)\sim\mathcal{N}(\mathbf{0},\boldsymbol{\Sigma})` is sensor noise.

    This implementation is **unit-aware** by using `brainunit <https://github.com/chaobrain/brainunit>`_:

    - If you pass unitless arrays, you can **attach/declare units** via constructor arguments.
    - If you pass Quantities (with units), the module will **validate and convert** to consistent units.


    Notes on Units
    --------------

    - **Consistency**: The class internally converts to a consistent set of units so that
      :math:`\mathbf{L} [\text{sensor}/(\text{nA·m})] \times \mathbf{s} [\text{nA·m}] \Rightarrow \mathbf{y} [\text{sensor}]`.
    - **Two common workflows**:

      1) *Unitless workflow*: provide unitless `L`, `scale`, and inputs. Specify `leadfield_unit`,
         `sensor_unit`, `dipole_unit` in the constructor to attach units. The model ensures consistency.
      2) *Quantity workflow*: provide `L` (with proper units), and pass NMM observables/scale as
         `brainunit` Quantities. The model converts as needed and outputs a `Quantity` in `sensor_unit`.
    - Sensor unit: ``"V"`` for EEG, ``"T"`` for MEG
    - Dipole unit: ``"nA*m"``,  ECD unit


    Shape Conventions
    -----------------

    - Let ``M`` = #sensors, ``R`` = #regions, ``T`` = #time steps.
    - `L`: ``(R, M)``.
    - `nmm_obs_or_dipoles`: ``(T, R)``.
    - Output `y`: ``(T, M)``.


    Biophysical Guidance
    --------------------

    - For EEG, :math:`\mathbf{L}` is usually computed with a BEM/forward solver in units of **V/(nA·m)** under a specific reference (e.g., average reference).
    - For MEG, :math:`\mathbf{L}` is typically in **T/(nA·m)**, accounting for sensor type (magnetometer/gradiometer).
    - Mapping NMM observables to ECD:
        *If your NMM provides (population) membrane potentials in mV*, use a scale such as
        ``scale = alpha * u("nA*m")/u("mV")`` so that ``s = scale * V_m`` produces dipole moments.
        `alpha` encodes column geometry, synchrony, etc.


    Examples
    --------
    >>> import brainunit as u
    >>> import jax.numpy as jnp

    >>> # constants
    >>> M, R, T = 64, 68, 2000

    >>> # EEG example: lead field in V/(nA*m)
    >>> L = jnp.ones((R, M)) * u.volt / (u.nA * u.meter)
    >>> # Suppose NMM observable is in mV; choose scale to map mV -> nA*m
    >>> scale = 0.1 * u.nA * u.meter / u.volt     # toy number
    >>> model = LeadFieldModel(L=L, scale=scale, sensor_unit=u.volt, dipole_unit=u.nA * u.meter)
    >>> nmm_obs = 50.0 * jnp.ones((T, R)) * u.mV # (T, R) in mV
    >>> y = model(nmm_obs)  # Quantity with unit "V", shape (T, M)

    >>> # MEG example: lead field in T/(nA*m)
    >>> L_u = jnp.ones((R, M)) * u.tesla * (u.nA * u.meter)
    >>> model_u = LeadFieldModel(L=L_u,  sensor_unit=u.volt, dipole_unit=u.nA*u.meter, scale=u.nA*u.meter/u.tesla)
    >>> y2 = model_u(50.0 * jnp.ones((T, R)) * u.tesla)


    Parameters
    ----------
    L : ArrayLike
        Lead field matrix. Accepts either:

        - A `brainunit.Quantity` with unit **sensor_unit / dipole_unit** and shape ``(R, M)``; or
        - A unitless ``(R, M)`` array, in which case `leadfield_unit` must be provided.
    sensor_unit : Unit, optional
        Unit of the sensor signals, e.g. ``"V"`` for EEG or ``"T"`` for MEG.
    dipole_unit : Unit, optional
        Unit of dipole moment (ECD), typically ``"nA*m"``.
    scale : Quantity, Unit, float, optional
        Optional mapping from the **NMM observable** to **dipole moment**.
        This can be:

        - A unit-aware scalar `Quantity`, e.g. ``10.0 * "nA*m" / u.mV``, meaning:
          ``s = scale * nmm_obs`` will convert mV to nA·m;
        - A dimensionless float if your NMM output is **already** in dipole units.
        If `scale` is not provided, the model expects inputs to `update` to already be in `dipole_unit`.
    noise_cov : ArrayLike
        Optional sensor noise covariance (``(M, M)``). If provided, it can be a Quantity with
        unit **sensor_unit^2**, or a unitless array in which case its unit is assumed compatible
        with ``sensor_unit**2``. Noise is sampled i.i.d. across time (same covariance at each step).

    """

    def __init__(
        self,
        in_size,
        out_size,
        L: Union[Array, Quantity, Callable],
        sensor_unit: u.Unit = u.mV,  # "V" for EEG, "T" for MEG
        dipole_unit: u.Unit = u.nA * u.meter,  # ECD unit
        scale: Optional[Union[float, Quantity]] = None,
        noise_cov: Optional[Union[Array, Quantity]] = None,
    ):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.L = braintools.init.param(L, (self.in_size[-1], self.out_size[-1]))
        self.noise_cov = braintools.init.param(noise_cov, (self.out_size[-1], self.out_size[-1]), allow_none=True)
        self.sensor_unit = sensor_unit
        self.dipole_unit = dipole_unit
        if scale is None:
            scale = dipole_unit / sensor_unit
        self.scale = scale
        if self.noise_cov is not None:
            u.fail_for_dimension_mismatch(self.noise_cov, self.sensor_unit ** 2)
            cov = self.noise_cov.to_decimal(self.sensor_unit ** 2)  # (M,M) unitless magnitude of sensor_unit^2
            self._noise_conv_Lc = jnp.linalg.cholesky(cov + 1e-32 * jnp.eye(cov.shape[0], dtype=cov.dtype))
        u.fail_for_dimension_mismatch(self.L, self.sensor_unit / self.dipole_unit)  # expected unit of L

    def _sample_noise(self, T: int) -> Quantity:
        """
        Sample Gaussian sensor noise E (M,T) with covariance self._noise_cov_q per time step.
        Returns a Quantity in sensor_unit.
        """
        z = brainstate.random.randn(T, self.M)
        e = z @ self._noise_conv_Lc  # (T,M) @ (M,M) -> (T,M)
        return e * self.sensor_unit

    # Inferred sizes
    @property
    def M(self) -> int:
        return self.out_size[-1]

    @property
    def R(self) -> int:
        return self.in_size[-1]

    def update(self, nmm_obs_or_dipoles: Union[Array, Quantity]) -> Quantity:
        r"""
        Forward mapping from region signals to sensors.

        Parameters
        ----------
        nmm_obs_or_dipoles : ArrayLike
            ``(R, T)`` region time series. If `already_in_dipole_unit=False`, this is the
            **NMM observable** (e.g., membrane potential in mV) which will be mapped to dipole
            moment via `scale`. If `already_in_dipole_unit=True`, it is interpreted directly as
            dipole moment (ECD) in `dipole_unit`.
            Accepts unitless arrays (units are attached via `scale`/constructor) or `Quantity`.

        Returns
        -------
        y :
            Sensor-space time series **Quantity** of shape ``(M, T)`` with unit `sensor_unit`.

        Notes
        -----
        The mapping uses the instantaneous linear model:

        .. math::
            \mathbf{Y} = \mathbf{S}\,\mathbf{L} + \mathbf{E}.

        where :math:`\mathbf{S}` stacks dipoles over time and :math:`\mathbf{E}` is sampled per
        time step from :math:`\mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma})` if enabled.

        """
        # 1) Ensure dipoles S (T, R) as Quantity in dipole_unit
        S_q = self.scale * nmm_obs_or_dipoles

        # 2) Lead field L as Quantity in sensor/dipole
        Lq = self.L

        # 3) Linear mapping
        Y_q = S_q @ Lq  # -> Quantity in sensor_unit

        # 4) Optional noise
        if self.noise_cov is not None:
            E_q = self._sample_noise(Y_q.shape[0])
            Y_q = Y_q + E_q

        # 5) Return Quantity in sensor_unit
        return Y_q

    # ----- Utilities for region aggregation from vertex-level leadfields -----

    @staticmethod
    def compress_vertex_leadfield(
        L_vertex_3V: Union[Array, Quantity],
        W_vertex_to_region_3VxR: Union[Array, Quantity],
    ) -> Union[Array, Quantity]:
        r"""
        Compress a vertex-level lead field to region-level:

        .. math::
            \mathbf{L}_{\text{region}} \;=\; \mathbf{L}_{\text{vertex}} \, \mathbf{W},

        where `L_vertex_3V` has shape ``(M, 3V)`` (x/y/z dipole components per vertex) and
        `W_vertex_to_region_3VxR` has shape ``(3V, R)`` (direction/area/parcel weights).

        Unit behavior
        -------------
        - If either input is a `Quantity`, unit algebra is applied automatically.
        - Typically, `L_vertex_3V` is in **sensor_unit/(nA·m)** and `W` is **dimensionless**,
          yielding region-level **sensor_unit/(nA·m)**.

        Returns
        -------
        L_region :
            Region-level lead field of shape ``(R, M)`` (Quantity if any input had units).
        """
        if isinstance(L_vertex_3V, u.Quantity) or isinstance(W_vertex_to_region_3VxR, u.Quantity):
            return L_vertex_3V @ W_vertex_to_region_3VxR
        return L_vertex_3V @ W_vertex_to_region_3VxR

    @staticmethod
    def build_fixed_orientation_weights(
        normals_V3: Array,
        parcel_VxR: Array,
        area_weights_V: Optional[Array] = None,
    ) -> Array:
        """
        Build a fixed-orientation vertex->region weight matrix with per-vertex surface normals.

        Parameters
        ----------
        normals_V3 :
            Array of shape ``(V, 3)`` with **unit surface normals** (x, y, z) at each cortical vertex.
        parcel_VxR :
            Array of shape ``(V, R)`` giving vertex-to-region assignment (one-hot or soft).
        area_weights_V :
            Optional array of shape ``(V,)`` with vertex area (or voxel volume) weights.
            If None, uses ones.

        Returns
        -------
        W_3VxR :
            Array of shape ``(3V, R)``. For each vertex ``v`` and region ``r``,
            the (x,y,z) blocks receive: ``area[v] * parcel[v,r] * normals[v,k]``.

        Notes
        -----
        This constructs a **direction-weighted, area-aware** aggregation matrix suitable for
        compressing a vertex-level lead field ``(M,3V)`` into region-level ``(M,R)`` with:

        .. code-block:: python

            W = build_fixed_orientation_weights(normals_V3, parcel_VxR, area_weights_V)
            L_region = L_vertex_3V @ W

        """
        V = normals_V3.shape[0]
        if normals_V3.shape != (V, 3):
            raise ValueError(f"`normals_V3` must be (V,3), got {normals_V3.shape}.")
        if parcel_VxR.shape[0] != V:
            raise ValueError("`parcel_VxR` first dim must match V.")
        if area_weights_V is None:
            area_weights_V = jnp.ones((V,), dtype=normals_V3.dtype)

        # (V,R)
        ar = area_weights_V[:, None] * parcel_VxR
        Wx = ar * normals_V3[:, 0:1]  # (V,R)
        Wy = ar * normals_V3[:, 1:2]
        Wz = ar * normals_V3[:, 2:3]
        # (3V,R)
        return jnp.vstack([Wx, Wy, Wz])


class EEGLeadFieldModel(LeadFieldModel):
    def __init__(
        self,
        in_size,
        out_size,
        L: Union[Array, Quantity, Callable],
        sensor_unit: u.Unit = u.mV,
        noise_cov: Optional[Union[Array, Quantity]] = None,
    ):
        super().__init__(
            in_size,
            out_size,
            L=L,
            sensor_unit=sensor_unit,
            dipole_unit=u.nA * u.meter,
            scale=u.nA * u.meter / u.mV,
            noise_cov=noise_cov,
        )


class MEGLeadFieldModel(LeadFieldModel):
    def __init__(
        self,
        in_size,
        out_size,
        L: Union[Array, Quantity, Callable],
        sensor_unit: u.Unit = u.tesla,
        noise_cov: Optional[Union[Array, Quantity]] = None,
    ):
        super().__init__(
            in_size,
            out_size,
            L=L,
            sensor_unit=sensor_unit,
            dipole_unit=u.nA * u.meter,
            scale=u.nA * u.meter / u.tesla,
            noise_cov=noise_cov,
        )
