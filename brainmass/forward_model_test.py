"""
Tests for forward_model.py

Covers:
- LeadFieldModel forward mapping with units and shapes
- EEG/MEG subclasses default units
- Vertex leadfield compression and fixed-orientation weights
- Basic BOLDSignal derivative shape check
"""

import brainunit as u
import jax.numpy as jnp
import pytest

import brainmass


class TestLeadFieldModel:
    def test_forward_shapes_and_units(self):
        # R regions, M sensors, T time points
        R, M, T = 3, 4, 5

        # Leadfield with units: sensor/(nA*m)
        L = (jnp.arange(R * M, dtype=jnp.float32).reshape(R, M) + 1.0) * (u.mV / (u.nA * u.meter))

        # Input observable in sensor units (mV), so default scale (nA*m/mV) maps to dipole units
        nmm_obs = jnp.ones((T, R), dtype=jnp.float32) * (2.0 * u.mV)

        model = brainmass.LeadFieldModel(
            in_size=(R,),
            out_size=(M,),
            L=L,
            sensor_unit=u.mV,
            dipole_unit=u.nA * u.meter,
        )

        y = model.update(nmm_obs)
        assert y.shape == (T, M)
        assert u.get_unit(y).has_same_dim(u.mV)

        # Expected computation: S = 2 mV * (nA*m/mV) -> 2 nA*m; Y = S @ L -> mV
        S = (2.0 * u.nA * u.meter) * jnp.ones((T, R), dtype=jnp.float32)
        expected = S @ L
        assert u.math.allclose(y, expected)

    def test_scale_quantity_mapping(self):
        R, M, T = 2, 3, 4

        L = jnp.ones((R, M), dtype=jnp.float32) * (u.mV / (u.nA * u.meter))
        # NMM observable in mV but we apply a custom scale 0.5 nA*m/mV
        nmm_obs = jnp.full((T, R), 10.0, dtype=jnp.float32) * u.mV
        scale = 0.5 * u.nA * u.meter / u.mV

        model = brainmass.LeadFieldModel(
            in_size=(R,),
            out_size=(M,),
            L=L,
            sensor_unit=u.mV,
            dipole_unit=u.nA * u.meter,
            scale=scale,
        )

        y = model.update(nmm_obs)
        # Each time row: S = 10 * 0.5 = 5 nA*m per region; expected = S @ L
        S = (5.0 * u.nA * u.meter) * jnp.ones((T, R), dtype=jnp.float32)
        expected = S @ L
        assert u.math.allclose(y, expected)


class TestEEGMEGModels:
    def test_eeg_defaults(self):
        R, M, T = 2, 3, 5
        L = jnp.ones((R, M), dtype=jnp.float32) * (u.mV / (u.nA * u.meter))
        nmm_obs = jnp.ones((T, R), dtype=jnp.float32) * (3.0 * u.mV)

        model = brainmass.EEGLeadFieldModel(
            in_size=(R,),
            out_size=(M,),
            L=L,
        )
        y = model.update(nmm_obs)
        assert y.shape == (T, M)
        assert u.get_unit(y).has_same_dim(u.mV)

    def test_meg_defaults(self):
        R, M, T = 2, 3, 5
        L = jnp.ones((R, M), dtype=jnp.float32) * (u.tesla / (u.nA * u.meter))
        nmm_obs = jnp.ones((T, R), dtype=jnp.float32) * (4.0 * u.tesla)

        model = brainmass.MEGLeadFieldModel(
            in_size=(R,),
            out_size=(M,),
            L=L,
        )
        y = model.update(nmm_obs)
        assert y.shape == (T, M)
        assert u.get_unit(y).has_same_dim(u.tesla)


class TestLeadfieldUtilities:
    def test_compress_vertex_leadfield_shapes_and_units(self):
        # M sensors, V vertices, R regions
        M, V, R = 3, 2, 4
        # L_vertex has shape (M, 3V)
        L_vertex = jnp.arange(M * 3 * V, dtype=jnp.float32).reshape(M, 3 * V) * (u.mV / (u.nA * u.meter))
        # W has shape (3V, R)
        W = jnp.ones((3 * V, R), dtype=jnp.float32)

        L_region = brainmass.LeadFieldModel.compress_vertex_leadfield(L_vertex, W)
        # As implemented: (M,3V) @ (3V,R) -> (M,R)
        assert L_region.shape == (M, R)
        assert u.get_unit(L_region).has_same_dim(u.mV / (u.nA * u.meter))

        # Numeric sanity: with W all-ones, each (m,r) is sum over 3V of L_vertex[m,:]
        expected = (L_vertex @ jnp.ones((3 * V, R), dtype=jnp.float32))
        assert u.math.allclose(L_region, expected)

    def test_build_fixed_orientation_weights(self):
        V, R = 3, 2
        normals = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=jnp.float32
        )
        parcel = jnp.array(
            [
                [1.0, 0.0],
                [0.5, 0.5],
                [0.0, 1.0],
            ],
            dtype=jnp.float32
        )
        area = jnp.array([2.0, 3.0, 4.0], dtype=jnp.float32)

        W = brainmass.LeadFieldModel.build_fixed_orientation_weights(normals, parcel, area)
        assert W.shape == (3 * V, R)

        # Check a few entries: block structure [V:x; V:y; V:z]
        # Vertex 0, region 0, x-component: area[0] * parcel[0,0] * normals[0,0] = 2 * 1 * 1 = 2
        assert jnp.isclose(W[0, 0], 2.0)
        # Vertex 1, region 1, y-component: 3 * 0.5 * 1 = 1.5 at row V + 1
        assert jnp.isclose(W[V + 1, 1], 1.5)
        # Vertex 2, region 1, z-component: 4 * 1 * 1 = 4 at row 2V + 2
        assert jnp.isclose(W[2 * V + 2, 1], 4.0)

    def test_build_fixed_orientation_weights_validation(self):
        V, R = 2, 2
        normals_bad = jnp.ones((V, 2), dtype=jnp.float32)  # wrong last dim
        parcel = jnp.ones((V, R), dtype=jnp.float32)
        with pytest.raises(ValueError):
            brainmass.LeadFieldModel.build_fixed_orientation_weights(normals_bad, parcel)

        normals = jnp.ones((V, 3), dtype=jnp.float32)
        parcel_bad = jnp.ones((V + 1, R), dtype=jnp.float32)  # wrong first dim
        with pytest.raises(ValueError):
            brainmass.LeadFieldModel.build_fixed_orientation_weights(normals, parcel_bad)


class TestBOLDSignal:
    def test_derivative_shapes(self):
        R = 4
        bold = brainmass.BOLDSignal(in_size=(R,))

        # Provide unitless arrays for derivative; only shape consistency is checked here
        x = jnp.ones((R,), dtype=jnp.float32)
        f = jnp.ones((R,), dtype=jnp.float32)
        v = jnp.ones((R,), dtype=jnp.float32)
        q = jnp.ones((R,), dtype=jnp.float32)
        z = jnp.zeros((R,), dtype=jnp.float32)

        dx, df, dv, dq = bold.derivative((x, f, v, q), 0.0, z)
        for arr in (dx, df, dv, dq):
            assert arr.shape == (R,)
