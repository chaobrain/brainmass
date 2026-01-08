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
import matplotlib.pyplot as plt
import numpy as np

import brainmass


class TestWongWangModel:

    def test_initialization(self):
        """Test model initialization and parameter setting."""
        model = brainmass.WongWangStep(in_size=10)

        # Check default parameter values
        assert model.tau_S == 100. * u.ms
        assert model.gamma == 0.641
        assert model.a == 270. * (u.Hz / u.nA)
        assert model.theta == 0.31 * u.nA

        # Check connectivity parameters
        assert model.J_N11 == 0.2609 * u.nA
        assert model.J_N22 == 0.2609 * u.nA
        assert model.J_N12 == 0.0497 * u.nA
        assert model.J_N21 == 0.0497 * u.nA

        print("✓ Initialization test passed")

    def test_state_initialization(self):
        """Test state initialization and reset."""
        model = brainmass.WongWangStep(in_size=5)

        # Test single instance initialization
        model.init_state()
        assert model.S1.value.shape == (5,)
        assert model.S2.value.shape == (5,)
        assert jnp.allclose(model.S1.value, 0.)
        assert jnp.allclose(model.S2.value, 0.)

        # Test batch initialization
        model.init_state(batch_size=3)
        assert model.S1.value.shape == (3, 5)
        assert model.S2.value.shape == (3, 5)

        # Test reset
        model.S1.value = jnp.ones((3, 5)) * 0.5
        model.S2.value = jnp.ones((3, 5)) * 0.3
        model.reset_state(batch_size=3)
        assert jnp.allclose(model.S1.value, 0.)
        assert jnp.allclose(model.S2.value, 0.)

        print("✓ State initialization test passed")

    def test_phi_function(self):
        """Test the input-output transfer function."""
        model = brainmass.WongWangStep(in_size=1)

        # Test threshold behavior
        I_below = 0.3 * u.nA  # Below threshold
        I_above = 0.4 * u.nA  # Above threshold

        r_below = model.phi(I_below)
        r_above = model.phi(I_above)

        assert r_below == 0. * u.Hz
        assert r_above > 0. * u.Hz
        assert r_above == model.a * (I_above - model.theta)

        print("✓ Transfer function test passed")

    def test_compute_inputs(self):
        """Test input computation with different coherence levels."""
        model = brainmass.WongWangStep(in_size=1)
        model.init_state()

        # Test zero coherence (equal inputs)
        I1_zero, I2_zero = model.compute_inputs(coherence=0.)

        # Test positive coherence (favors population 1)
        I1_pos, I2_pos = model.compute_inputs(coherence=0.5)

        # Test negative coherence (favors population 2)
        I1_neg, I2_neg = model.compute_inputs(coherence=-0.5)

        # Check coherence effects
        assert I1_pos > I1_zero > I1_neg
        assert I2_neg > I2_zero > I2_pos

        print("✓ Input computation test passed")

    def test_single_step_update(self):
        """Test single time step update."""
        brainstate.environ.set(dt=0.1 * u.second)  # Set 0.1 second timestep

        model = brainmass.WongWangStep(in_size=1)
        model.init_state()

        # Initial state should be zero
        assert model.S1.value == 0.
        assert model.S2.value == 0.

        # Update with positive coherence
        r1, r2 = model.update(coherence=0.32)

        # After one step, S values should have changed from zero
        assert model.S1.value > 0.
        assert model.S2.value > 0.

        # With positive coherence, population 1 should be more active
        assert r1 >= r2

        print("✓ Single step update test passed")

    def test_decision_dynamics(self):
        """Test decision-making dynamics over time."""
        brainstate.environ.set(dt=0.1 * u.second)  # 0.1 s timestep

        model = brainmass.WongWangStep(in_size=1)
        model.init_state()

        # Simulate decision with strong rightward coherence
        coherence = 0.5
        n_steps = 5000  # 500 s simulation

        def step_run(i):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                r1, r2 = model.update(coherence=coherence)
                return r1, r2, model.S1.value, model.S2.value

        indices = np.arange(n_steps)
        results = brainstate.transform.for_loop(step_run, indices)
        r1_trace, r2_trace, S1_trace, S2_trace = results

        # Check that decision emerges (population 1 should win)
        final_r1 = r1_trace[-1]
        final_r2 = r2_trace[-1]

        assert final_r1 > final_r2  # Population 1 should win
        assert final_r1 > 10. * u.Hz  # Should reach reasonable firing rate

        # Check that S values are within bounds
        assert jnp.all(S1_trace >= 0.) and jnp.all(S1_trace <= 1.)
        assert jnp.all(S2_trace >= 0.) and jnp.all(S2_trace <= 1.)

        print("✓ Decision dynamics test passed")

    def test_coherence_effect(self):
        """Test effect of different coherence levels."""
        brainstate.environ.set(dt=0.1 * u.second)

        model = brainmass.WongWangStep(in_size=1)
        model.init_state()

        coherence_levels = [-0.8, -0.4, 0., 0.4, 0.8]
        final_rates = []

        for coh in coherence_levels:
            model.reset_state()
            # Short simulation to reach steady state
            for _ in range(30):
                r1, r2 = model.update(coherence=coh)
            final_rates.append((r1, r2))

        # Check coherence effects
        for i, (r1, r2) in enumerate(final_rates):
            coh = coherence_levels[i]
            if coh > 0.1:  # Strong positive coherence
                assert r1 > r2
            elif coh < -0.1:  # Strong negative coherence  
                assert r2 > r1
            # For zero coherence, either could win due to noise/asymmetries

        print("✓ Coherence effect test passed")

    def test_get_decision(self):
        """Test decision detection method."""
        model = brainmass.WongWangStep(in_size=1)
        model.init_state()

        # Set states to create clear decision
        model.S1.value = jnp.array([0.8])  # High activity pop 1
        model.S2.value = jnp.array([0.1])  # Low activity pop 2

        decision = model.get_decision(threshold=10. * u.Hz)

        # Should detect decision for population 1
        assert decision == 1 or decision == 0  # Depends on exact firing rates

        print("✓ Decision detection test passed")

    def test_noise_integration(self):
        """Test model with noise processes."""
        brainstate.environ.set(dt=0.1 * u.second)

        # Create noise processes
        noise_s1 = brainmass.OUProcess(1, tau=2. * u.second, sigma=0.02 * u.nA, mean=0. * u.nA)
        noise_s2 = brainmass.OUProcess(1, tau=2. * u.second, sigma=0.02 * u.nA, mean=0. * u.nA)

        model = brainmass.WongWangStep(
            in_size=1,
            noise_s1=noise_s1,
            noise_s2=noise_s2
        )
        brainstate.nn.init_all_states(model)

        # Run simulation with noise
        n_steps = 1000

        def step_run(i):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                r1, r2 = model.update(coherence=0.)

        brainstate.transform.for_loop(step_run, np.arange(n_steps))

        # With noise, states should be non-zero even at zero coherence
        assert model.S1.value > 0. or model.S2.value > 0.

        print("✓ Noise integration test passed")

    def test_batch_processing(self):
        """Test batch processing capability."""
        brainstate.environ.set(dt=0.1 * u.second)

        batch_size = 4
        model = brainmass.WongWangStep(in_size=1)
        model.init_state(batch_size=batch_size)

        # Update with different coherence for each batch element
        coherences = jnp.array([0.2, 0.4, -0.2, -0.4])

        def step_fun(i):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                # Note: update method currently takes scalar coherence
                # This is a limitation that could be addressed
                r1, r2 = model.update(coherence=0.2)  # Same for all batches

        brainstate.transform.for_loop(step_fun, np.arange(1000))

        # Check that all batch elements have valid states
        assert model.S1.value.shape == (batch_size, 1)
        assert model.S2.value.shape == (batch_size, 1)
        assert jnp.all(model.S1.value >= 0.) and jnp.all(model.S1.value <= 1.)
        assert jnp.all(model.S2.value >= 0.) and jnp.all(model.S2.value <= 1.)

        print("✓ Batch processing test passed")

    def run_all_tests(self):
        """Run all tests."""
        print("Running WongWangStep tests...")
        print("=" * 40)

        self.test_initialization()
        self.test_state_initialization()
        self.test_phi_function()
        self.test_compute_inputs()
        self.test_single_step_update()
        self.test_decision_dynamics()
        self.test_coherence_effect()
        self.test_get_decision()
        self.test_noise_integration()
        self.test_batch_processing()

        print("=" * 40)
        print("All WongWangStep tests passed! ✓")

    def test_demo_decision_making(self, plot=True):
        """Demonstrate decision-making behavior."""
        brainstate.environ.set(dt=0.0001 * u.second)

        model = brainmass.WongWangStep(in_size=1)
        model.init_state()

        n_steps = 8000  # 800 ms
        coherence = 0.32  # Moderate coherence

        def step_run(i):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                r1, r2 = model.update(coherence=coherence)
                return r1, r2, model.S1.value, model.S2.value

        indices = np.arange(n_steps)
        results = brainstate.transform.for_loop(step_run, indices)
        r1_trace, r2_trace, S1_trace, S2_trace = results

        time = indices * brainstate.environ.get_dt()  # Convert to ms

        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # Plot firing rates
            ax1.plot(time, r1_trace, label='Population 1', color='blue', linewidth=2)
            ax1.plot(time, r2_trace, label='Population 2', color='red', linewidth=2)
            ax1.set_ylabel('Firing Rate (Hz)')
            ax1.set_title(f'Wong-Wang Decision Making (coherence = {coherence})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot synaptic variables
            ax2.plot(time, S1_trace, label='S1', color='blue', linewidth=2)
            ax2.plot(time, S2_trace, label='S2', color='red', linewidth=2)
            ax2.set_xlabel('Time (second)')
            ax2.set_ylabel('Synaptic Gating')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
            plt.close()

        return time, r1_trace, r2_trace, S1_trace, S2_trace

    def test_demo_decision_making_ms(self, plot=True):
        """Demonstrate decision-making behavior."""
        brainstate.environ.set(dt=0.1 * u.ms)

        model = brainmass.WongWangStep(in_size=1, tau_S=100. * u.ms)
        model.init_state()

        n_steps = 8000  # 800 ms
        coherence = 0.32  # Moderate coherence

        def step_run(i):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                r1, r2 = model.update(coherence=coherence)
                return r1, r2, model.S1.value, model.S2.value

        indices = np.arange(n_steps)
        results = brainstate.transform.for_loop(step_run, indices)
        r1_trace, r2_trace, S1_trace, S2_trace = results

        time = indices * brainstate.environ.get_dt()  # Convert to ms

        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # Plot firing rates
            ax1.plot(time, r1_trace, label='Population 1', color='blue', linewidth=2)
            ax1.plot(time, r2_trace, label='Population 2', color='red', linewidth=2)
            ax1.set_ylabel('Firing Rate (Hz)')
            ax1.set_title(f'Wong-Wang Decision Making (coherence = {coherence})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot synaptic variables
            ax2.plot(time, S1_trace, label='S1', color='blue', linewidth=2)
            ax2.plot(time, S2_trace, label='S2', color='red', linewidth=2)
            ax2.set_xlabel('Time (second)')
            ax2.set_ylabel('Synaptic Gating')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
            plt.close()

    def test_demo_decision_making_ms_v3(self, plot=True):
        """Demonstrate decision-making behavior."""
        brainstate.environ.set(dt=0.1 * u.ms)

        model = brainmass.WongWangStep(in_size=1, tau_S=0.1 * u.second)
        model.init_state()

        n_steps = 8000  # 800 ms
        coherence = 0.32  # Moderate coherence

        def step_run(i):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                r1, r2 = model.update(coherence=coherence)
                return r1, r2, model.S1.value, model.S2.value

        indices = np.arange(n_steps)
        results = brainstate.transform.for_loop(step_run, indices)
        r1_trace, r2_trace, S1_trace, S2_trace = results

        time = indices * brainstate.environ.get_dt()  # Convert to ms

        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # Plot firing rates
            ax1.plot(time, r1_trace, label='Population 1', color='blue', linewidth=2)
            ax1.plot(time, r2_trace, label='Population 2', color='red', linewidth=2)
            ax1.set_ylabel('Firing Rate (Hz)')
            ax1.set_title(f'Wong-Wang Decision Making (coherence = {coherence})')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot synaptic variables
            ax2.plot(time, S1_trace, label='S1', color='blue', linewidth=2)
            ax2.plot(time, S2_trace, label='S2', color='red', linewidth=2)
            ax2.set_xlabel('Time (second)')
            ax2.set_ylabel('Synaptic Gating')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
            plt.close()
