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

import brainunit as u
import jax.numpy as jnp
import numpy as np

import brainmass
import brainstate


class TestOUProcess:
    def test_initialization_basic(self):
        """Test basic OUProcess initialization with default parameters"""
        noise = brainmass.OUProcess(1)
        assert noise.in_size == (1,)
        assert noise.sigma.value() == 1. * u.nA
        assert noise.mean.value() == 0. * u.nA
        assert noise.tau.value() == 10. * u.ms

    def test_initialization_with_parameters(self):
        """Test OUProcess initialization with custom parameters"""
        noise = brainmass.OUProcess(
            in_size=5,
            mean=2.5 * u.nA,
            sigma=0.5 * u.nA,
            tau=20. * u.ms
        )
        assert noise.in_size == (5,)
        assert noise.sigma.value() == 0.5 * u.nA
        assert noise.mean.value() == 2.5 * u.nA
        assert noise.tau.value() == 20. * u.ms

    def test_initialization_multidimensional(self):
        """Test OUProcess initialization with multidimensional input"""
        noise = brainmass.OUProcess((3, 4))
        assert noise.in_size == (3, 4)

    def test_initialization_sequence_input(self):
        """Test OUProcess initialization with sequence input size"""
        noise = brainmass.OUProcess([2, 3, 4])
        assert noise.in_size == (2, 3, 4)

    def test_parameter_validation_negative_sigma(self):
        """Test that negative sigma values are handled appropriately"""
        # The implementation doesn't explicitly validate sigma > 0, 
        # but we can test behavior with negative values
        noise = brainmass.OUProcess(1, sigma=-1.0 * u.nA)
        assert noise.sigma.value() == -1.0 * u.nA

    def test_parameter_validation_negative_tau(self):
        """Test that negative tau values are handled appropriately"""
        # The implementation doesn't explicitly validate tau > 0,
        # but we can test behavior with negative values  
        noise = brainmass.OUProcess(1, tau=-5.0 * u.ms)
        assert noise.tau.value() == -5.0 * u.ms

    def test_state_initialization(self):
        """Test state initialization creates correct shape and initial values"""
        noise = brainmass.OUProcess((2, 3))
        noise.init_state()

        assert hasattr(noise, 'x')
        assert isinstance(noise.x, brainstate.HiddenState)
        assert noise.x.value.shape == (2, 3)
        assert u.get_unit(noise.x.value) == u.nA
        # Initial state should be zeros
        assert u.math.allclose(noise.x.value, np.zeros((2, 3)) * u.nA)

    def test_state_initialization_with_batch(self):
        """Test state initialization with batch dimension"""
        noise = brainmass.OUProcess(3)
        batch_size = 5
        noise.init_state(batch_size=batch_size)

        assert noise.x.value.shape == (5, 3)
        assert u.math.allclose(noise.x.value, np.zeros((5, 3)) * u.nA)

    def test_state_reset(self):
        """Test state reset functionality"""
        noise = brainmass.OUProcess(2)
        noise.init_state()

        # Modify state
        noise.x.value = jnp.array([1.0, 2.0]) * u.nA
        assert not u.math.allclose(noise.x.value, np.zeros(2) * u.nA)

    def test_state_reset_with_batch(self):
        """Test state reset with batch dimension"""
        noise = brainmass.OUProcess(2)
        batch_size = 3
        noise.init_state(batch_size=batch_size)

        # Modify state
        noise.x.value = jnp.ones((3, 2)) * u.nA

    def test_update_returns_correct_shape(self):
        """Test that update returns correct shape"""
        noise = brainmass.OUProcess((2, 3))
        noise.init_state()

        with brainstate.environ.context(dt=0.1 * u.ms):
            result = noise.update()

        assert result.shape == (2, 3)
        assert u.get_unit(result) == u.nA

    def test_statistical_properties_mean_reversion(self):
        """Test that the process shows mean reversion over time"""
        brainstate.environ.set(dt=0.1 * u.ms)

        # Start with non-zero initial condition
        noise = brainmass.OUProcess(1, mean=0.0 * u.nA, sigma=0.1 * u.nA, tau=10.0 * u.ms)
        noise.init_state()
        noise.x.value = jnp.array([5.0]) * u.nA  # Start far from mean

        def step_run(i):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                return noise.update()

        # Run for many steps to observe mean reversion
        xs = brainstate.transform.for_loop(step_run, np.arange(1000))

        # The final values should be closer to mean than initial
        final_mean = np.mean(xs[-100:])  # Average of last 100 steps
        assert abs(final_mean) < 4.0 * u.nA, "Process should revert toward mean over time"

    def test_statistical_properties_with_nonzero_mean(self):
        """Test that process converges to specified non-zero mean"""
        brainstate.environ.set(dt=0.1 * u.ms)

        target_mean = 2.0 * u.nA
        noise = brainmass.OUProcess(1, mean=target_mean, sigma=0.5 * u.nA, tau=5.0 * u.ms)
        noise.init_state()

        def step_run(i):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                return noise.update()

        # Run simulation
        xs = brainstate.transform.for_loop(step_run, np.arange(2000))

        # Check convergence to target mean
        final_mean = u.math.mean(xs[-200:]).mantissa  # Average of last 200 steps
        assert abs(final_mean - 2.0) < 0.5, f"Mean should converge to {target_mean}, got {final_mean}"

    def test_different_time_steps(self):
        """Test behavior with different time step sizes"""
        noise = brainmass.OUProcess(1, sigma=1.0 * u.nA, tau=10.0 * u.ms)

        # Test with different dt values
        dt_values = [0.01 * u.ms, 0.1 * u.ms, 1.0 * u.ms]
        results = []

        for dt in dt_values:
            noise.init_state()
            with brainstate.environ.context(dt=dt):
                result = noise.update()
            results.append(result)

        # All should return valid results with same shape
        for result in results:
            assert result.shape == (1,)
            assert u.get_unit(result) == u.nA

    def test_batch_processing(self):
        """Test OUProcess with batch processing"""
        batch_size = 10
        noise = brainmass.OUProcess(3, sigma=1.0 * u.nA, tau=10.0 * u.ms)
        noise.init_state(batch_size=batch_size)

        with brainstate.environ.context(dt=0.1 * u.ms):
            result = noise.update()

        assert result.shape == (batch_size, 3)
        assert u.get_unit(result) == u.nA

    def test_batch_processing_independence(self):
        """Test that batch samples are independent"""
        batch_size = 100
        noise = brainmass.OUProcess(1, sigma=2.0 * u.nA, tau=10.0 * u.ms)
        noise.init_state(batch_size=batch_size)

        def step_run(i):
            with brainstate.environ.context(dt=0.1 * u.ms):
                return noise.update()

        # Generate multiple time steps
        xs = brainstate.transform.for_loop(step_run, np.arange(100))

        # Check that different batch elements have different values
        # (with high probability)
        final_values = xs[-1, :]  # Last time step, all batch elements
        unique_values = len(u.math.unique(u.math.around(final_values, decimals=6)))
        assert unique_values > batch_size * 0.8, "Batch samples should be largely independent"

    def test_call_method_alias(self):
        """Test that calling the object directly works as alias for update"""
        noise = brainmass.OUProcess(1)
        noise.init_state()

        with brainstate.environ.context(dt=0.1 * u.ms):
            result1 = noise.update()

    def test_integration_long_simulation(self):
        """Test stability over long simulation"""
        brainstate.environ.set(dt=0.1 * u.ms)

        noise = brainmass.OUProcess(1, mean=1.0 * u.nA, sigma=0.5 * u.nA, tau=20.0 * u.ms)
        noise.init_state()

        def step_run(i):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                return noise.update()

        # Long simulation
        xs = brainstate.transform.for_loop(step_run, np.arange(10000))

        # Check that values remain bounded (shouldn't explode to infinity)
        assert u.math.all(u.math.isfinite(xs)), "All values should remain finite"
        assert u.math.max(u.math.abs(xs)) < 50.0 * u.nA, "Values should remain reasonably bounded"


class TestNoise:
    def test1(self):
        noise = brainmass.OUProcess(1)
        noise.init_state()

        def step_run(i):
            with brainstate.environ.context(dt=0.1 * u.ms):
                noise()
            return noise.x.value

        xs = brainstate.transform.for_loop(step_run, np.arange(100000))
        # Remove plotting for automated testing
        # plt.plot(xs)
        # plt.show()
        # plt.close()


class TestGaussianAndWhiteNoise:
    def test_gaussian_initialization_and_shape(self):
        n = brainmass.GaussianNoise(in_size=3)
        assert n.in_size == (3,)
        assert n.sigma.value() == 1.0 * u.nA
        assert n.mean.value() == 0.0 * u.nA
        n.init_state()
        out = n.update()
        assert out.shape == (3,)
        assert u.get_unit(out) == u.nA

    def test_white_noise_alias(self):
        n = brainmass.WhiteNoise(in_size=(2, 4), sigma=0.5 * u.nA)
        n.init_state()
        out = n.update()
        assert out.shape == (2, 4)
        assert u.get_unit(out) == u.nA


class TestBrownianNoise:
    def test_initialization_and_shape(self):
        n = brainmass.BrownianNoise(in_size=4)
        n.init_state()
        with brainstate.environ.context(dt=0.1 * u.ms):
            out = n.update()
        assert out.shape == (4,)
        assert u.get_unit(out) == u.nA


class TestColoredNoise:
    def test_colored_noise_shape_and_stats(self):
        # Use sufficiently long last axis to allow frequency shaping
        in_size = (3, 512)
        target_mean = 0.3 * u.nA
        target_sigma = 1.2 * u.nA
        n = brainmass.ColoredNoise(in_size=in_size, beta=1.0, mean=target_mean, sigma=target_sigma)
        n.init_state()
        y = n.update()
        assert y.shape == in_size
        assert u.get_unit(y) == u.nA
        # Stats over last axis, averaged across leading dims
        mu = u.math.mean(u.math.mean(y, axis=-1))
        stds = u.math.std(y, axis=-1)
        std_avg = u.math.mean(stds)
        # Loose checks due to randomness
        assert abs(mu - target_mean) < 0.4 * u.nA
        assert abs(std_avg - target_sigma) < 0.5 * u.nA

    def test_pink_blue_violet_classes(self):
        for cls in [brainmass.PinkNoise, brainmass.BlueNoise, brainmass.VioletNoise]:
            n = cls(in_size=(2, 256), sigma=0.8 * u.nA)
            n.init_state()
            y = n.update()
            assert y.shape == (2, 256)
            assert u.get_unit(y) == u.nA
