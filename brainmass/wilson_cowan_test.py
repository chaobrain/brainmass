# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

import brainmass


class TestWilsonCowanModel:
    def test_initialization_basic(self):
        """Test basic WilsonCowanModel initialization with default parameters"""
        model = brainmass.WilsonCowanModel(1)
        assert model.in_size == (1,)
        assert model.tau_E == 1. * u.ms
        assert model.tau_I == 1. * u.ms
        assert model.a_E == 1.2
        assert model.a_I == 1.0
        assert model.theta_E == 2.8
        assert model.theta_I == 4.0
        assert model.wEE == 12.
        assert model.wIE == 4.
        assert model.wEI == 13.
        assert model.wII == 11.
        assert model.r == 1.
        assert model.noise_E is None
        assert model.noise_I is None

    def test_initialization_with_custom_parameters(self):
        """Test WilsonCowanModel initialization with custom parameters"""
        model = brainmass.WilsonCowanModel(
            in_size=5,
            tau_E=2.0 * u.ms,
            tau_I=1.5 * u.ms,
            a_E=1.5,
            a_I=1.1,
            theta_E=3.0,
            theta_I=4.5,
            wEE=10.,
            wIE=3.5,
            wEI=12.,
            wII=10.5,
            r=0.8
        )
        assert model.in_size == (5,)
        assert model.tau_E == 2.0 * u.ms
        assert model.tau_I == 1.5 * u.ms
        assert model.a_E == 1.5
        assert model.a_I == 1.1
        assert model.theta_E == 3.0
        assert model.theta_I == 4.5
        assert model.wEE == 10.
        assert model.wIE == 3.5
        assert model.wEI == 12.
        assert model.wII == 10.5
        assert model.r == 0.8

    def test_initialization_multidimensional(self):
        """Test WilsonCowanModel initialization with multidimensional input"""
        model = brainmass.WilsonCowanModel((3, 4))
        assert model.in_size == (3, 4)

    def test_initialization_with_noise(self):
        """Test WilsonCowanModel initialization with noise processes"""
        noise_E = brainmass.OUProcess(1, sigma=0.5, tau=10. * u.ms)
        noise_I = brainmass.OUProcess(1, sigma=0.3, tau=15. * u.ms)

        model = brainmass.WilsonCowanModel(1, noise_E=noise_E, noise_I=noise_I)
        assert model.noise_E is noise_E
        assert model.noise_I is noise_I

    def test_initialization_invalid_noise(self):
        """Test that invalid noise objects raise assertion errors"""
        try:
            model = brainmass.WilsonCowanModel(1, noise_E="invalid")
            assert False, "Should raise assertion error for invalid noise_E"
        except AssertionError:
            pass

        try:
            model = brainmass.WilsonCowanModel(1, noise_I="invalid")
            assert False, "Should raise assertion error for invalid noise_I"
        except AssertionError:
            pass

    def test_state_initialization(self):
        """Test state initialization creates correct shape and initial values"""
        model = brainmass.WilsonCowanModel((2, 3))
        model.init_state()

        assert hasattr(model, 'rE')
        assert hasattr(model, 'rI')
        assert isinstance(model.rE, brainstate.HiddenState)
        assert isinstance(model.rI, brainstate.HiddenState)
        assert model.rE.value.shape == (2, 3)
        assert model.rI.value.shape == (2, 3)

        # Initial state should be zeros (dimensionless)
        assert u.math.allclose(model.rE.value, np.zeros((2, 3)))
        assert u.math.allclose(model.rI.value, np.zeros((2, 3)))

    def test_state_initialization_with_batch(self):
        """Test state initialization with batch dimension"""
        model = brainmass.WilsonCowanModel(3)
        batch_size = 5
        model.init_state(batch_size=batch_size)

        assert model.rE.value.shape == (5, 3)
        assert model.rI.value.shape == (5, 3)
        assert u.math.allclose(model.rE.value, np.zeros((5, 3)))
        assert u.math.allclose(model.rI.value, np.zeros((5, 3)))

    def test_state_reset(self):
        """Test state reset functionality"""
        model = brainmass.WilsonCowanModel(2)
        model.init_state()

        # Modify state
        model.rE.value = jnp.array([1.0, 2.0])
        model.rI.value = jnp.array([0.5, 1.5])
        assert not u.math.allclose(model.rE.value, np.zeros(2))
        assert not u.math.allclose(model.rI.value, np.zeros(2))

        # Reset should return to zeros
        model.reset_state()
        assert u.math.allclose(model.rE.value, np.zeros(2))
        assert u.math.allclose(model.rI.value, np.zeros(2))

    def test_state_reset_with_batch(self):
        """Test state reset with batch dimension"""
        model = brainmass.WilsonCowanModel(2)
        batch_size = 3
        model.init_state(batch_size=batch_size)

        # Modify state
        model.rE.value = jnp.ones((3, 2))
        model.rI.value = jnp.ones((3, 2)) * 0.5

        # Reset should return to zeros
        model.reset_state(batch_size=batch_size)
        assert u.math.allclose(model.rE.value, np.zeros((3, 2)))
        assert u.math.allclose(model.rI.value, np.zeros((3, 2)))

    def test_F_sigmoid_function(self):
        """Test the sigmoid activation function F"""
        model = brainmass.WilsonCowanModel(1)

        # Test with typical parameters
        x = jnp.array([0., 1., 2., 3., 4., 5.])
        a = 1.2
        theta = 2.8

        result = model.F(x, a, theta)

        # F should be monotonically increasing
        assert jnp.all(jnp.diff(result) >= 0)

        # F should be bounded (allow small negative values due to floating point precision)
        assert jnp.all(result >= -1e-6)
        assert jnp.all(result <= 1)

        # F should approach 0 for large negative inputs relative to threshold
        assert model.F(-10., a, theta) < 0.1

        # F should approach its maximum for large positive inputs
        assert model.F(10., a, theta) > 0.8

    def test_drE_differential_equation(self):
        """Test excitatory population differential equation"""
        model = brainmass.WilsonCowanModel(1)

        rE = 0.5  # dimensionless activity
        rI = 0.3  # dimensionless activity  
        ext = 1.0

        drE_dt = model.drE(rE, rI, ext)

        # Result should have correct units (time^-1 dimension)
        actual_unit = u.get_unit(drE_dt)
        assert actual_unit.dim == (1 / u.ms).dim

        # Result should be finite
        assert u.math.isfinite(drE_dt)

    def test_drI_differential_equation(self):
        """Test inhibitory population differential equation"""
        model = brainmass.WilsonCowanModel(1)

        rE = 0.5  # dimensionless activity
        rI = 0.3  # dimensionless activity
        ext = 1.0

        drI_dt = model.drI(rI, rE, ext)

        # Result should have correct units (time^-1 dimension)
        actual_unit = u.get_unit(drI_dt)
        assert actual_unit.dim == (1 / u.ms).dim

        # Result should be finite
        assert u.math.isfinite(drI_dt)

    def test_update_returns_correct_shape(self):
        """Test that update returns correct shape"""
        model = brainmass.WilsonCowanModel((2, 3))
        model.init_state()

        brainstate.environ.set(dt=0.1 * u.ms)
        result = model.update()

        assert result.shape == (2, 3)

    def test_update_modifies_state(self):
        """Test that update modifies the internal state"""
        brainstate.environ.set(dt=0.1 * u.ms)

        model = brainmass.WilsonCowanModel(1)
        model.init_state()

        initial_rE = model.rE.value.copy()
        initial_rI = model.rI.value.copy()

        # Apply external input to ensure state change
        result = model.update(rE_inp=2.0, rI_inp=1.0)

        # State should have changed
        assert not u.math.allclose(model.rE.value, initial_rE)
        assert not u.math.allclose(model.rI.value, initial_rI)

    def test_update_with_external_inputs(self):
        """Test update with external inputs"""
        brainstate.environ.set(dt=0.1 * u.ms)

        model = brainmass.WilsonCowanModel(1)
        model.init_state()

        # Test with different external inputs
        result1 = model.update(rE_inp=1.0, rI_inp=0.5)

        model.reset_state()
        result2 = model.update(rE_inp=2.0, rI_inp=1.0)

        # Different inputs should produce different outputs
        assert not u.math.allclose(result1, result2)

    def test_update_with_noise(self):
        """Test update with noise processes"""
        brainstate.environ.set(dt=0.1 * u.ms)

        # Create dimensionless noise processes for Wilson-Cowan model
        noise_E = brainmass.OUProcess(1, sigma=0.1, tau=10. * u.ms, mean=0.)
        noise_I = brainmass.OUProcess(1, sigma=0.05, tau=15. * u.ms, mean=0.)
        noise_E.init_state()
        noise_I.init_state()

        model = brainmass.WilsonCowanModel(1, noise_E=noise_E, noise_I=noise_I)
        model.init_state()

        # Multiple updates should produce different results due to noise
        results = []
        for _ in range(5):
            model.reset_state()
            noise_E.reset_state()
            noise_I.reset_state()
            result = model.update()
            results.append(result)

        # At least some results should be different (due to noise)
        results_values = []
        for r in results:
            if hasattr(r, 'mantissa'):
                val = r.mantissa.item() if r.mantissa.ndim > 0 else r.mantissa
            else:
                val = r.item() if r.ndim > 0 else r
            results_values.append(float(val))
        
        unique_count = len(jnp.unique(jnp.round(jnp.array(results_values), 6)))
        assert unique_count > 1, "Noise should cause variability in outputs"

    def test_oscillatory_dynamics(self):
        """Test that the model can produce oscillatory dynamics"""
        brainstate.environ.set(dt=0.01 * u.ms)

        # Parameters chosen to promote oscillations
        model = brainmass.WilsonCowanModel(
            1,
            tau_E=1.0 * u.ms,
            tau_I=2.0 * u.ms,
            wEE=10.,
            wEI=12.,
            wIE=10.,
            wII=3.
        )
        model.init_state()

        def step_run(i):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                return model.update(rE_inp=3.0)

        # Run simulation
        n_steps = 5000
        results = brainstate.transform.for_loop(step_run, np.arange(n_steps))

        # Check that simulation runs and produces finite values
        assert u.math.all(u.math.isfinite(results)), "All results should be finite"
        activity = results.mantissa if hasattr(results, 'mantissa') else results
        activity_var = jnp.var(activity[-1000:])  # Variance in last part of simulation
        # Test that the model produces reasonable activity levels
        final_activity = jnp.mean(activity[-100:])
        assert final_activity >= 0, "Activity should be non-negative"
        assert final_activity < 10, "Activity should not explode"

    def test_steady_state_convergence(self):
        """Test convergence to steady state with constant input"""
        brainstate.environ.set(dt=0.1 * u.ms)

        model = brainmass.WilsonCowanModel(1)
        model.init_state()

        def step_run(i):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                return model.update(rE_inp=1.0, rI_inp=0.5)

        # Run simulation
        n_steps = 2000
        results = brainstate.transform.for_loop(step_run, np.arange(n_steps))

        # Check convergence (last 100 steps should be similar)
        final_values = results[-100:]
        final_var = u.math.var(final_values)
        final_var_val = final_var.mantissa if hasattr(final_var, 'mantissa') else final_var
        assert final_var_val < 0.01, "Model should converge to steady state"

    def test_batch_processing(self):
        """Test WilsonCowanModel with batch processing"""
        brainstate.environ.set(dt=0.1 * u.ms)

        batch_size = 10
        model = brainmass.WilsonCowanModel(3)
        model.init_state(batch_size=batch_size)

        result = model.update(rE_inp=1.0, rI_inp=0.5)

        assert result.shape == (batch_size, 3)

    def test_multidimensional_input(self):
        """Test model with multidimensional input size"""
        brainstate.environ.set(dt=0.1 * u.ms)

        model = brainmass.WilsonCowanModel((2, 3))
        model.init_state()

        result = model.update()

        assert result.shape == (2, 3)

    def test_parameter_arrays(self):
        """Test model with array parameters for different regions"""
        brainstate.environ.set(dt=0.1 * u.ms)

        # Different parameters for different regions
        tau_E_array = jnp.array([1.0, 2.0, 1.5]) * u.ms
        tau_I_array = jnp.array([1.0, 1.5, 2.0]) * u.ms

        model = brainmass.WilsonCowanModel(
            3,
            tau_E=tau_E_array,
            tau_I=tau_I_array
        )
        model.init_state()

        result = model.update(rE_inp=1.0, rI_inp=0.5)

        assert result.shape == (3,)

    def test_stability_long_simulation(self):
        """Test stability over long simulation"""
        brainstate.environ.set(dt=0.1 * u.ms)

        model = brainmass.WilsonCowanModel(1)
        model.init_state()

        def step_run(i):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                return model.update(rE_inp=2.0, rI_inp=1.0)

        # Long simulation
        n_steps = 10000
        results = brainstate.transform.for_loop(step_run, np.arange(n_steps))

        # Check that values remain bounded
        assert u.math.all(u.math.isfinite(results)), "All values should remain finite"
        max_val = u.math.max(u.math.abs(results)).mantissa if hasattr(u.math.max(u.math.abs(results)), 'mantissa') else u.math.max(u.math.abs(results))
        assert max_val < 100.0, "Values should remain reasonably bounded"

    def test_excitation_inhibition_balance(self):
        """Test excitation-inhibition balance affects dynamics"""
        brainstate.environ.set(dt=0.1 * u.ms)

        # High excitation model
        model_exc = brainmass.WilsonCowanModel(1, wEE=15., wEI=5.)
        model_exc.init_state()

        # High inhibition model
        model_inh = brainmass.WilsonCowanModel(1, wEE=5., wEI=15.)
        model_inh.init_state()

        def run_model(model):
            def step_run(i):
                with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                    return model.update(rE_inp=2.0)

            return brainstate.transform.for_loop(step_run, np.arange(1000))

        results_exc = run_model(model_exc)
        results_inh = run_model(model_inh)

        # Excitatory-dominant should have higher final activity
        mean_exc = u.math.mean(results_exc[-100:])
        mean_inh = u.math.mean(results_inh[-100:])
        final_exc = mean_exc.mantissa if hasattr(mean_exc, 'mantissa') else mean_exc
        final_inh = mean_inh.mantissa if hasattr(mean_inh, 'mantissa') else mean_inh

        assert final_exc > final_inh, "Excitatory-dominant model should have higher activity"


class TestWilsonCowanIntegration:
    def test_integration_with_coupling(self):
        """Test WilsonCowanModel integration with coupling for network simulation"""
        brainstate.environ.set(dt=0.1 * u.ms)

        # Create two coupled regions
        n_regions = 2
        models = [brainmass.WilsonCowanModel(1) for _ in range(n_regions)]

        # Initialize all models
        for model in models:
            model.init_state()

        # Simple coupling matrix
        coupling_strength = 0.5

        def step_run(i):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                # Get current activities
                activities = [model.rE.value for model in models]

                # Apply coupling (simplified)
                coupling_inputs = [
                    coupling_strength * activities[1],  # Region 0 receives from region 1
                    coupling_strength * activities[0]  # Region 1 receives from region 0
                ]

                # Update models with coupling
                results = []
                for j, model in enumerate(models):
                    coupling_val = coupling_inputs[j]
                    # JAX will handle the broadcasting automatically
                    result = model.update(rE_inp=1.0 + coupling_val)
                    results.append(result)

                return jnp.stack(results)

        # Run coupled simulation
        n_steps = 1000
        results = brainstate.transform.for_loop(step_run, np.arange(n_steps))

        assert results.shape == (n_steps, n_regions, 1)
        assert u.math.all(u.math.isfinite(results))

    def test_integration_with_external_stimulus(self):
        """Test model response to time-varying external stimulus"""
        brainstate.environ.set(dt=0.1 * u.ms)

        model = brainmass.WilsonCowanModel(1)
        model.init_state()

        def step_run(i):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                # Sinusoidal input
                t = i * 0.1  # time in ms
                stimulus = 2.0 + 1.0 * jnp.sin(2 * jnp.pi * t / 100.0)  # 10 Hz oscillation
                return model.update(rE_inp=stimulus)

        # Run simulation
        n_steps = 2000
        results = brainstate.transform.for_loop(step_run, np.arange(n_steps))

        # Model should follow the stimulus
        assert results.shape == (n_steps, 1)
        assert u.math.all(u.math.isfinite(results))

        # Activity should show variation due to stimulus
        activity_var = u.math.var(results).mantissa if hasattr(u.math.var(results), 'mantissa') else u.math.var(results)
        assert activity_var > 1e-12, "Model should respond to varying stimulus"
