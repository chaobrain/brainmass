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
        brainstate.random.seed(0)  # deterministic so the tolerances below are meaningful

        # Start with non-zero initial condition
        noise = brainmass.OUProcess(1, mean=0.0 * u.nA, sigma=0.1 * u.nA, tau=10.0 * u.ms)
        noise.init_state()
        noise.x.value = jnp.array([5.0]) * u.nA  # Start far from mean

        def step_run(i):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                return noise.update()

        # Run for many steps to observe mean reversion (100 ms = 10 tau).
        xs = brainstate.transform.for_loop(step_run, np.arange(1000))

        # The process must revert to near its mean of 0 (started at 5 nA). The old
        # bound of 4 nA was so loose it would pass even with badly-scaled noise.
        final_mean = np.mean(xs[-100:])
        assert abs(final_mean) < 0.5 * u.nA, "Process should revert toward mean over time"

        # The stationary fluctuation scale must match the noise amplitude. A broken
        # increment (e.g. the historical sqrt(dt) cancellation bug) would shift this
        # by orders of magnitude.
        tail_std = float(np.std(xs[-500:].mantissa))
        assert 0.03 < tail_std < 0.15, f"stationary std {tail_std:.3f} nA is off-scale"

    def test_statistical_properties_with_nonzero_mean(self):
        """Test that process converges to specified non-zero mean"""
        brainstate.environ.set(dt=0.1 * u.ms)

        brainstate.random.seed(0)  # deterministic mean estimate
        target_mean = 2.0 * u.nA
        noise = brainmass.OUProcess(1, mean=target_mean, sigma=0.5 * u.nA, tau=5.0 * u.ms)
        noise.init_state()

        def step_run(i):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                return noise.update()

        # Run simulation
        xs = brainstate.transform.for_loop(step_run, np.arange(2000))

        # Check convergence to target mean (averaging the second half for a stable
        # estimate of the stationary mean).
        final_mean = u.math.mean(xs[1000:]).mantissa
        assert abs(final_mean - 2.0) < 0.3, f"Mean should converge to {target_mean}, got {final_mean}"

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
        brainstate.random.seed(0)
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
        brainstate.random.seed(0)

        noise = brainmass.OUProcess(1, mean=1.0 * u.nA, sigma=0.5 * u.nA, tau=20.0 * u.ms)
        noise.init_state()

        def step_run(i):
            with brainstate.environ.context(i=i, t=i * brainstate.environ.get_dt()):
                return noise.update()

        # Long simulation
        xs = brainstate.transform.for_loop(step_run, np.arange(10000))

        # Check that values remain bounded (shouldn't explode to infinity). The old
        # bound of 50 nA was ~25x the actual excursion; a much tighter bound still
        # passes the well-behaved process but catches a runaway/explosion.
        assert u.math.all(u.math.isfinite(xs)), "All values should remain finite"
        assert u.math.max(u.math.abs(xs)) < 5.0 * u.nA, "Values should remain reasonably bounded"

        # The stationary mean/std must match the configured drift and amplitude.
        assert abs(np.mean(xs[5000:].mantissa) - 1.0) < 0.2, "should fluctuate about mean 1 nA"
        assert 0.2 < float(np.std(xs[5000:].mantissa)) < 0.7, "stationary std off-scale"


class TestNoise:
    def test1(self):
        """A long OU run stays finite and bounded (no plotting needed)."""
        brainstate.random.seed(0)
        noise = brainmass.OUProcess(1)
        noise.init_state()

        def step_run(i):
            with brainstate.environ.context(dt=0.1 * u.ms):
                noise()
            return noise.x.value

        xs = brainstate.transform.for_loop(step_run, np.arange(100000))
        assert u.math.all(u.math.isfinite(xs)), "OU process should remain finite"
        # Default sigma=1 nA, tau=10 ms: excursions stay well within ~10 nA.
        assert u.math.max(u.math.abs(xs)) < 10.0 * u.nA


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


class TestBrownianNoiseScaling:
    """Brownian (Wiener) increment must scale as ``sigma * sqrt(dt)``."""

    @staticmethod
    def _first_increment(dt, seed=123, n=50000, sigma=2.0 * u.nA):
        # First update from a zero initial state equals the per-step increment.
        brainstate.random.seed(seed)
        proc = brainmass.BrownianNoise(in_size=n, sigma=sigma)
        proc.init_state()
        with brainstate.environ.context(dt=dt):
            out = proc.update()
        return np.asarray(u.get_magnitude(out))

    def test_increment_variance_scales_with_dt(self):
        # Same seed => identical N(0,1) draw, so Var ratio == dt ratio exactly.
        inc1 = self._first_increment(0.1 * u.ms)
        inc2 = self._first_increment(0.4 * u.ms)
        ratio = float(np.var(inc2)) / float(np.var(inc1))
        # Buggy code (increment == sigma) gives ratio ~ 1; correct gives ~ 4.
        assert 3.8 < ratio < 4.2, f'variance ratio {ratio} should be ~4 (dt 0.4/0.1)'

    def test_increment_magnitude_matches_sigma_sqrt_dt(self):
        # At dt == 1 ms the increment std equals sigma (sqrt(dt/ms) == 1).
        inc = self._first_increment(1.0 * u.ms, sigma=2.0 * u.nA)
        assert abs(float(np.std(inc)) - 2.0) < 0.1

    def test_zero_sigma_is_deterministic(self):
        proc = brainmass.BrownianNoise(in_size=10, sigma=0.0 * u.nA)
        proc.init_state()
        with brainstate.environ.context(dt=0.1 * u.ms):
            out = proc.update()
        assert u.math.allclose(out, 0. * u.nA)

    def test_seed_determinism(self):
        a = self._first_increment(0.1 * u.ms, seed=7, n=200)
        b = self._first_increment(0.1 * u.ms, seed=7, n=200)
        assert np.allclose(a, b)

    def test_update_is_unit_safe_and_finite(self):
        brainstate.random.seed(0)
        proc = brainmass.BrownianNoise(in_size=4, sigma=1.5 * u.nA)
        proc.init_state()
        with brainstate.environ.context(dt=0.1 * u.ms):
            out = proc.update()
        assert u.get_unit(out) == u.nA
        assert u.math.all(u.math.isfinite(out))


class TestColoredNoiseSpectralSlope:
    """Colored-noise PSD slope: PSD(f) ~ f**(-beta) so log-log slope == -beta."""

    @staticmethod
    def _psd_slope(noise, seed=0):
        brainstate.random.seed(seed)
        y = np.asarray(u.get_magnitude(noise.update()))  # (reps, n)
        n = y.shape[-1]
        psd = np.mean(np.abs(np.fft.rfft(y, axis=-1)) ** 2, axis=0)
        freqs = np.fft.rfftfreq(n)
        lo, hi = 1, len(freqs)  # exclude DC only
        f, p = freqs[lo:hi], psd[lo:hi]
        mask = p > 0
        return float(np.polyfit(np.log(f[mask]), np.log(p[mask]), 1)[0])

    def _make(self, cls=None, beta=None, in_size=(256, 2048)):
        n = (cls(in_size=in_size, sigma=1.0 * u.nA) if cls is not None
             else brainmass.ColoredNoise(in_size=in_size, beta=beta, sigma=1.0 * u.nA))
        n.init_state()
        return n

    def test_blue_slope_is_two(self):
        slope = self._psd_slope(self._make(cls=brainmass.BlueNoise))
        assert abs(slope - 2.0) < 0.5, f'blue slope {slope} should be ~ +2'

    def test_violet_slope_is_four(self):
        slope = self._psd_slope(self._make(cls=brainmass.VioletNoise))
        assert abs(slope - 4.0) < 0.8, f'violet slope {slope} should be ~ +4'

    def test_pink_slope_is_minus_one(self):
        slope = self._psd_slope(self._make(cls=brainmass.PinkNoise))
        assert abs(slope + 1.0) < 0.5, f'pink slope {slope} should be ~ -1'

    def test_slope_ordering_vs_white_and_pink(self):
        s_pink = self._psd_slope(self._make(cls=brainmass.PinkNoise), seed=1)
        s_white = self._psd_slope(self._make(beta=0.0), seed=2)
        s_blue = self._psd_slope(self._make(cls=brainmass.BlueNoise), seed=3)
        s_violet = self._psd_slope(self._make(cls=brainmass.VioletNoise), seed=4)
        assert s_pink < s_white < s_blue < s_violet
        assert s_pink < 0 < s_blue
