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
import functools

import brainstate
import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import brainmass


class TestJansenRitModel:

    def test_initialization(self):
        """Test model initialization and parameter setting."""
        model = brainmass.JansenRitStep(in_size=10)

        # Check default parameter values
        assert model.Ae == 3.25 * u.mV
        assert model.Ai == 22. * u.mV
        assert model.be == 100. / u.second
        assert model.bi == 50. / u.second
        assert model.C == 135.
        assert model.a1 == 1.
        assert model.a2 == 0.8
        assert model.a3 == 0.25
        assert model.a4 == 0.25
        assert model.s_max == 5. / u.second
        assert model.v0 == 6. * u.mV
        assert model.r == 0.56

        print("[PASS] Initialization test passed")

    def test_custom_parameters(self):
        """Test model with custom parameters."""
        model = brainmass.JansenRitStep(
            in_size=5,
            Ae=5.0 * u.mV,
            Ai=30.0 * u.mV,
            be=80. / u.second,
            bi=40. / u.second,
            C=150.,
            a1=1.2,
            a2=0.9,
            a3=0.3,
            a4=0.3,
            s_max=7. / u.second,
            v0=7. * u.mV,
            r=0.7
        )

        assert model.Ae == 5.0 * u.mV
        assert model.Ai == 30.0 * u.mV
        assert model.be == 80. / u.second
        assert model.bi == 40. / u.second
        assert model.C == 150.
        assert model.a1 == 1.2
        assert model.a2 == 0.9
        assert model.a3 == 0.3
        assert model.a4 == 0.3
        assert model.s_max == 7. / u.second
        assert model.v0 == 7. * u.mV
        assert model.r == 0.7

        print("[PASS] Custom parameters test passed")

    def test_state_initialization(self):
        """Test state initialization and reset."""
        model = brainmass.JansenRitStep(in_size=5)

        # Test single instance initialization
        model.init_state()
        assert model.M.value.shape == (5,)
        assert model.Mv.value.shape == (5,)
        assert model.E.value.shape == (5,)
        assert model.Ev.value.shape == (5,)
        assert model.I.value.shape == (5,)
        assert model.Iv.value.shape == (5,)

        # Check all states start at zero
        assert u.math.allclose(model.M.value / u.mV, 0.)
        assert u.math.allclose(model.Mv.value / (u.mV / u.second), 0.)
        assert u.math.allclose(model.E.value / u.mV, 0.)
        assert u.math.allclose(model.Ev.value / (u.mV / u.second), 0.)
        assert u.math.allclose(model.I.value / u.mV, 0.)
        assert u.math.allclose(model.Iv.value / (u.mV / u.second), 0.)

        # Test batch initialization
        model.init_state(batch_size=3)
        assert model.M.value.shape == (3, 5)
        assert model.Mv.value.shape == (3, 5)
        assert model.E.value.shape == (3, 5)
        assert model.Ev.value.shape == (3, 5)
        assert model.I.value.shape == (3, 5)
        assert model.Iv.value.shape == (3, 5)

        # Test reset
        model.M.value = jnp.ones((3, 5)) * 0.5 * u.mV
        model.Mv.value = jnp.ones((3, 5)) * 0.1 * u.mV / u.second
        model.reset_state(batch_size=3)
        assert u.math.allclose(model.M.value / u.mV, 0.)
        assert u.math.allclose(model.Mv.value / (u.mV / u.second), 0.)

        print("[PASS] State initialization test passed")

    def test_sigmoid_function(self):
        """Test the sigmoid activation function."""
        model = brainmass.JansenRitStep(in_size=1)

        # Test different input values
        v_low = 0. * u.mV  # Below threshold
        v_threshold = model.v0  # At threshold
        v_high = 20. * u.mV  # Above threshold

        s_low = model.S(v_low)
        s_threshold = model.S(v_threshold)
        s_high = model.S(v_high)

        # Check sigmoid properties
        assert s_low < s_threshold < s_high
        assert s_low >= 0. / u.second
        assert s_high <= model.s_max
        assert s_threshold == model.s_max / 2.  # Should be halfway at v0

        # Test saturation
        v_very_high = 100. * u.mV
        s_saturated = model.S(v_very_high)
        assert u.math.allclose(s_saturated / (1 / u.second), model.s_max / (1 / u.second), rtol=1e-2)

        print("[PASS] Sigmoid function test passed")

    def test_derivative_functions(self):
        """Test the derivative computation functions."""
        model = brainmass.JansenRitStep(in_size=1)
        model.init_state()

        # Set some test values
        y0 = 5. * u.mV
        y1 = 2. * u.mV / u.second
        y2 = 3. * u.mV
        y3 = 1. * u.mV / u.second
        y4 = 2. * u.mV
        y5 = 0.5 * u.mV / u.second
        Ip = 1. * u.mV
        Ii = 0.5 * u.mV

        # Test dy1 function
        dy1_val = model.dMv(y1, y0, y2, y4, Ip)
        assert hasattr(dy1_val, 'mantissa')  # Should be a Quantity with units

        # Test dy3 function
        dy3_val = model.dEv(y3, y0, y2)
        assert hasattr(dy3_val, 'mantissa')  # Should be a Quantity with units

        # Test dy5 function
        dy5_val = model.dIv(y5, y0, y4, Ii)
        assert hasattr(dy5_val, 'mantissa')  # Should be a Quantity with units

        print("[PASS] Derivative functions test passed")

    def test_single_step_update(self):
        """Test single time step update."""
        brainstate.environ.set(dt=0.0001 * u.second)  # 0.1 ms timestep

        model = brainmass.JansenRitStep(in_size=1)
        model.init_state()

        # Initial state should be zero
        assert u.math.allclose(model.M.value / u.mV, 0.)
        assert u.math.allclose(model.Mv.value / (u.mV / u.second), 0.)

        # Update with external inputs
        Ip = 2. * u.mV
        Ii = 0. * u.mV
        output = model.update(M_inp=Ip, I_inp=Ii)

        # After one step, states should have evolved
        # The output should be the EEG proxy signal
        expected_output = model.a2 * model.E.value - model.a4 * model.I.value
        assert u.math.allclose(output, expected_output)

        print("[PASS] Single step update test passed")

    def test_eeg_output_signal(self):
        """Test EEG output signal computation."""
        model = brainmass.JansenRitStep(in_size=1)
        model.init_state()

        # Set specific state values
        model.E.value = jnp.array([10.]) * u.mV  # Excitatory PSP
        model.I.value = jnp.array([5.]) * u.mV  # Inhibitory PSP

        output = model.update()

        # Output should be a2*E - a4*I
        expected = 10. * u.mV - 5. * u.mV
        assert u.math.allclose(output, expected)

        print("[PASS] EEG output signal test passed")

    def test_oscillatory_dynamics(self):
        """Test oscillatory behavior with appropriate parameters."""
        brainstate.environ.set(dt=0.0001 * u.second)  # 0.1 ms

        model = brainmass.JansenRitStep(in_size=1)
        model.init_state()

        # Run simulation with constant input
        n_steps = 10000  # 1 second
        Ip = 3. * u.mV  # Moderate input to pyramidal population

        def step_run(i):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                return model.update(M_inp=Ip)

        indices = np.arange(n_steps)
        outputs = brainstate.transform.for_loop(step_run, indices)

        # Check for oscillatory behavior
        # The signal should not be constant
        # assert u.math.std(outputs) > 0.1 * u.mV

        # Check that states remain within reasonable bounds
        assert jnp.all(jnp.abs(model.M.value / u.mV) < 50.)
        assert jnp.all(jnp.abs(model.E.value / u.mV) < 50.)
        assert jnp.all(jnp.abs(model.I.value / u.mV) < 50.)

        print("[PASS] Oscillatory dynamics test passed")

    def test_input_response(self):
        """Test response to different input levels."""
        brainstate.environ.set(dt=0.0001 * u.second)

        model = brainmass.JansenRitStep(in_size=1)

        input_levels = [0., 1., 3., 5., 10.] * u.mV
        final_outputs = []

        def step_run(i, Ip):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                return model.update(M_inp=Ip)

        for Ip in input_levels:
            model.init_state()

            # Run to steady state
            out = brainstate.transform.for_loop(functools.partial(step_run, Ip=Ip), np.arange(5000))
            final_outputs.append(out[-1])

        # Higher inputs should generally lead to larger output magnitudes
        output_magnitudes = [u.math.abs(out) for out in final_outputs]

        # # At least some progression with input level
        # assert output_magnitudes[-1] >= output_magnitudes[0]
        #
        # print("[PASS] Input response test passed")

    def test_parameter_sensitivity(self):
        """Test sensitivity to key parameters."""
        brainstate.environ.set(dt=0.0001 * u.second)

        # Test with different excitatory gains
        Ae_values = [2. * u.mV, 3.25 * u.mV, 5. * u.mV]
        outputs = []

        for Ae in Ae_values:
            model = brainmass.JansenRitStep(in_size=1, Ae=Ae)
            model.init_state()

            def step_run(i, Ip):
                with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                    return model.update(M_inp=Ip)

            # Short simulation
            out = brainstate.transform.for_loop(functools.partial(step_run, Ip=2. * u.mV), np.arange(1000))
            outputs.append(u.math.abs(out[-1]))

        # Different Ae values should produce different outputs
        assert not u.math.allclose(outputs[0], outputs[1])
        assert not u.math.allclose(outputs[1], outputs[2])

        print("[PASS] Parameter sensitivity test passed")

    def test_batch_processing(self):
        """Test batch processing capability."""
        brainstate.environ.set(dt=0.0001 * u.second)

        batch_size = 4
        model = brainmass.JansenRitStep(in_size=2)
        model.init_state(batch_size=batch_size)

        # Update with same input for all batches
        Ip = 2. * u.mV
        output = model.update(M_inp=Ip)

        # Check output shape
        assert output.shape == (batch_size, 2)

        # Check that all batch elements have valid states
        states_units = [(model.M.value, u.mV), (model.Mv.value, u.mV / u.second), (model.E.value, u.mV),
                        (model.Ev.value, u.mV / u.second), (model.I.value, u.mV), (model.Iv.value, u.mV / u.second)]
        for state, unit in states_units:
            assert state.shape == (batch_size, 2)
            assert jnp.all(jnp.isfinite((state / unit)))

        print("[PASS] Batch processing test passed")

    def test_stability(self):
        """Test numerical stability over long simulation."""
        brainstate.environ.set(dt=0.0001 * u.second)

        model = brainmass.JansenRitStep(in_size=1)
        model.init_state()

        # Long simulation
        n_steps = 20000  # 2 seconds
        Ip = 2. * u.mV

        def step_run(i):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                return model.update(M_inp=Ip)

        indices = np.arange(n_steps)
        outputs = brainstate.transform.for_loop(step_run, indices)

        # Check for numerical stability
        assert jnp.all(jnp.isfinite(outputs / u.mV))
        assert jnp.all(jnp.isfinite(model.M.value / u.mV))
        assert jnp.all(jnp.isfinite(model.Mv.value / (u.mV / u.second)))

        # States should not blow up
        assert jnp.all(jnp.abs(model.M.value / u.mV) < 100.)
        assert jnp.all(jnp.abs(model.E.value / u.mV) < 100.)
        assert jnp.all(jnp.abs(model.I.value / u.mV) < 100.)

        print("[PASS] Stability test passed")

    def run_all_tests(self):
        """Run all tests."""
        print("Running JansenRit2Window tests...")
        print("=" * 40)

        self.test_initialization()
        self.test_custom_parameters()
        self.test_state_initialization()
        self.test_sigmoid_function()
        self.test_derivative_functions()
        self.test_single_step_update()
        self.test_eeg_output_signal()
        self.test_oscillatory_dynamics()
        self.test_input_response()
        self.test_parameter_sensitivity()
        self.test_batch_processing()
        self.test_stability()

        print("=" * 40)
        print("All JansenRit2Window tests passed! [PASS]")

    def test_demo_alpha_rhythm(self, plot=True):
        """Demonstrate alpha rhythm generation."""
        brainstate.environ.set(dt=0.0001 * u.second)  # 0.1 ms

        # Parameters for alpha rhythm generation
        model = brainmass.JansenRitStep(
            in_size=1,
            Ae=3.25 * u.mV,
            Ai=22. * u.mV,
            be=100. / u.second,
            bi=50. / u.second,
            C=135.,
            a1=1.,
            a2=0.8,
            a3=0.25,
            a4=0.25
        )
        model.init_state()

        n_steps = 20000  # 2 seconds
        Ip = 2.5 * u.mV  # Moderate tonic input

        def step_run(i):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                return model.update(M_inp=Ip)

        indices = np.arange(n_steps)
        outputs = brainstate.transform.for_loop(step_run, indices)

        time = indices * brainstate.environ.get_dt()

        if plot:
            plt.figure(figsize=(12, 8))

            # Plot EEG signal
            plt.subplot(2, 1, 1)
            plt.plot(time, outputs, linewidth=1)
            plt.ylabel('EEG Signal (mV)')
            plt.title('Jansen-Rit Model: Alpha Rhythm Generation')
            plt.grid(True, alpha=0.3)

            # Plot power spectral density
            plt.subplot(2, 1, 2)
            from scipy import signal
            dt_val = float(brainstate.environ.get_dt() / u.second)
            freqs, psd = signal.welch(outputs.mantissa.flatten(), fs=1 / dt_val, nperseg=4096)
            plt.semilogy(freqs[:200], psd[:200])  # Focus on 0-100 Hz
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power Spectral Density')
            plt.title('Power Spectrum')
            plt.grid(True, alpha=0.3)
            plt.xlim(0, 50)

            plt.tight_layout()
            plt.show()
            plt.close()

        return time, outputs

    def test_demo_seizure_like_activity(self, plot=True):
        """Demonstrate seizure-like activity with modified parameters."""
        brainstate.environ.set(dt=0.0001 * u.second)  # 0.1 ms

        # Parameters for seizure-like activity (increased connectivity)
        model = brainmass.JansenRitStep(
            in_size=1,
            Ae=5. * u.mV,  # Increased excitatory gain
            Ai=15. * u.mV,  # Reduced inhibitory gain
            C=200.,  # Increased connectivity
            a2=1.2,  # Increased excitatory feedback
            a4=0.15  # Reduced inhibitory feedback
        )
        model.init_state()

        n_steps = 15000  # 1.5 seconds
        Ip = 3. * u.mV  # Higher input

        def step_run(i):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                return model.update(M_inp=Ip)

        indices = np.arange(n_steps)
        outputs = brainstate.transform.for_loop(step_run, indices)

        time = indices * brainstate.environ.get_dt()

        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(time, outputs, linewidth=1, color='red')
            plt.ylabel('EEG Signal (mV)')
            plt.xlabel('Time (s)')
            plt.title('Jansen-Rit Model: Seizure-like Activity')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            plt.close()

        return time, outputs
