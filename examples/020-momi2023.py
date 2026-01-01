"""
Jansen-Rit Neural Mass Model - Momi 2023 Implementation

Rewritten using nmm/ modular APIs.
"""

import os
import time
os.environ['JAX_TRACEBACK_FILTERING'] = 'off'

import brainstate
import brainunit as u
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from sklearn.metrics.pairwise import cosine_similarity

import braintools
from brainmass.jansen_rit_v3 import JansenRitModel
from braintools.param import Param, Const, ReluT, GaussianReg


class ModelFitting:
    """Model fitting class for training and testing."""

    def __init__(self, model: JansenRitModel, ts, num_epoches):
        """
        Initialize model fitting.

        Args:
            model: JansenRitWindow instance.
            ts: (num_windows, batch_size, output_size) empirical EEG data.
            num_epoches: Number of training epochs.
        """
        self.model = model
        self.num_epoches = num_epoches
        self.ts = ts
        self.w_cost = 10.

        # Initialize ADAM optimizer
        self.optimizer = braintools.optim.Adam(lr=5e-2)
        self.weights = self.model.states(brainstate.ParamState)
        self.optimizer.register_trainable_weights(self.weights)

    def f_loss(self, model_state, inputs, targets):
        # Forward pass
        params = self.model.get_params()
        model_state, (eeg_output, _) = self.model.update(model_state, params, inputs)

        # Calculate prior reg loss using Param API
        loss_prior = []
        for module in self.model.nodes(Param).values():
            loss_prior.append(module.reg_loss())

        # Calculate total loss
        loss = self.w_cost * u.math.mean((eeg_output - targets) ** 2) + sum(loss_prior)

        return loss, (model_state, eeg_output)

    @brainstate.transform.jit(static_argnums=0)
    def f_train(self, model_state, inputs, targets):
        f_grad = brainstate.transform.grad(self.f_loss, self.weights,
                                           has_aux=True, return_value=True, check_states=False)
        grads, loss, (model_state, eeg_output) = f_grad(model_state, inputs, targets)
        self.optimizer.step(grads)
        return loss, model_state, eeg_output

    @brainstate.transform.jit(static_argnums=0)
    def f_predict(self, model_state, inputs):
        # Forward pass (no gradient computation)
        params = self.model.get_params()
        model_state, (eeg_output, state_output) = self.model.update(model_state, params, inputs)
        return model_state, eeg_output, state_output

    def train(self, u):
        """
        Train the model.

        Args:
            u: (time_dim, steps_per_TR, node_size) external input.

        Returns:
            Loss history array.
        """
        # Define constants
        w_cost = 10  # Weight for data fitting cost

        # Initialize states - ModelData contains dynamics_state and delay_state
        model_state = self.model.get_states()

        # Define masks for lower triangle matrices
        mask_e = np.tril_indices(self.model.output_size, -1)

        loss_his = []

        # ts: (num_windows, batch_size, output_size)
        num_windows = self.ts.shape[0]
        batch_size = self.ts.shape[1]

        # Training loop
        for i_epoch in range(self.num_epoches):
            batch_losses = []
            batch_eeg = []

            for i_window in range(num_windows):
                # external input: (time_dim, steps_per_TR, node_size)
                # Slice for this window: (batch_size, steps_per_TR, node_size)
                external = u[i_window * batch_size:(i_window + 1) * batch_size]
                loss, model_state, eeg_output = self.f_train(model_state, external, self.ts[i_window])

                # Record loss and eeg
                loss_np = np.asarray(loss)
                loss_his.append(loss_np)
                batch_losses.append(loss_np)
                batch_eeg.append(np.asarray(eeg_output))

            # Calculate FC correlation for monitoring
            ts_emp = self.ts.reshape(-1, self.ts.shape[-1])  # (total_time, output_size)
            fc = np.corrcoef(ts_emp.T)
            ts_sim = np.concatenate(batch_eeg, axis=0)  # (total_time, output_size)
            fc_sim = np.corrcoef(ts_sim[10:, :].T)

            # Print progress
            print(
                f'epoch: {i_epoch}, '
                f'loss: {np.asarray(batch_losses).mean():.4f}, '
                f'FC corr: {np.corrcoef(fc_sim[mask_e], fc[mask_e])[0, 1]:.4f}, '
                f'cos_sim: {np.diag(cosine_similarity(ts_sim.T, ts_emp.T)).mean():.4f}'
            )

        return np.array(loss_his)

    def test(self, base_batch_num: int, u):
        model_state = self.model.get_states()

        # ts: (num_windows, batch_size, output_size)
        batch_size = self.ts.shape[1]
        total_time = self.ts.shape[0] * batch_size
        num_windows = self.ts.shape[0] + base_batch_num

        # Prepare external input with warmup period
        u_hat = np.zeros(
            (base_batch_num * batch_size + total_time,
             self.model.steps_per_TR,
             self.model.node_size)
        )
        u_hat[base_batch_num * batch_size:] = u

        E_sim = []
        I_sim = []
        eeg_sim = []

        # Testing loop
        for i_batch in range(num_windows):
            # external: (batch_size, steps_per_TR, node_size)
            external = u_hat[i_batch * batch_size:(i_batch + 1) * batch_size]
            model_state, eeg_output, state_output = self.f_predict(model_state, external)

            # Store outputs after warmup period
            if i_batch > base_batch_num - 1:
                # Extract E and I from states: (batch_size, node_size, 6)
                E_sim.append(np.asarray(state_output['E']))  # E is index 1
                I_sim.append(np.asarray(state_output['I']))  # I is index 2
                eeg_sim.append(np.asarray(eeg_output))

        # Calculate final FC correlation
        ts_emp = self.ts.reshape(-1, self.ts.shape[-1])
        fc = np.corrcoef(ts_emp.T)
        E_result = np.concatenate(E_sim, axis=0)
        I_result = np.concatenate(I_sim, axis=0)
        eeg_result = np.concatenate(eeg_sim, axis=0)

        print(
            f'Test, FC corr: {fc.mean():.4f}, '
            f'cos_sim: {np.diag(cosine_similarity(eeg_result.T, ts_emp.T)).mean():.4f}'
        )
        return E_result, I_result, eeg_result


def main(subject_idx=0, visualize=True, output_dir='./results'):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Jansen-Rit Model Training - Using nmm/ APIs")
    print("=" * 60)

    # =============================================================================
    # 1. CONFIGURE DATA PATHS
    # =============================================================================
    files_dir = 'D:/codes/githubs/computational_neuroscience/PyTepFit/reproduce_Momi_et_al_2022'
    sc_file = files_dir + '/Schaefer2018_200Parcels_7Networks_count.csv'
    dist_file = files_dir + '/Schaefer2018_200Parcels_7Networks_distance.csv'
    eeg_file = files_dir + '/only_high_trial.mat'
    stim_weights_file = files_dir + '/stim_weights.npy'
    print("\n1. Loading data files...")

    # =============================================================================
    # 2. LOAD DATA
    # =============================================================================

    # Load structural connectivity
    sc_df = pd.read_csv(sc_file, header=None, sep=' ')
    sc = sc_df.values
    sc = 0.5 * (sc + sc.T)  # Symmetrize
    sc = np.log1p(sc) / np.linalg.norm(np.log1p(sc))  # Normalize
    print(f"   Structural connectivity: {sc.shape}")

    # Load distance matrix
    dist_df = pd.read_csv(dist_file, header=None, sep=' ')
    dist = dist_df.values
    print(f"   Distance matrix: {dist.shape}")

    # Load empirical EEG data
    data_high = scipy.io.loadmat(eeg_file)
    eeg_data = data_high['only_high_trial']
    print(f"   EEG data: {eeg_data.shape}")
    print(f"   (subjects x channels x timepoints)")

    # Load stimulation weights
    stim_weights = np.load(stim_weights_file)
    ki0 = stim_weights  # (node_size,) 1D vector
    print(f"   Stimulation weights: {stim_weights.shape}")

    # =============================================================================
    # 3. CONFIGURE MODEL PARAMETERS
    # =============================================================================
    print("\n2. Configuring model parameters...")

    node_size = stim_weights.shape[0]  # 200 brain regions
    output_size = 62  # EEG channels
    batch_size = 50  # Batch size for training
    step_size = 0.0001  # Integration step (0.1 ms)
    num_epochs = 40  # Training epochs
    tr = 0.001  # Time resolution (1 ms)
    base_batch_num = 20  # Warmup batches for testing

    print(f"   Nodes: {node_size}, Channels: {output_size}")
    print(f"   Epochs: {num_epochs}, Batch size: {batch_size}")
    print(f"   Integration step: {step_size} s, TR: {tr} s")

    # =============================================================================
    # 4. INITIALIZE MODEL
    # =============================================================================
    print("\n3. Initializing Jansen-Rit model...")

    # Select subject
    data_mean = eeg_data[subject_idx]

    sub_file_leadfield = files_dir + f'/leadfield_from_mne/sub{str(subject_idx + 1).zfill(3)}/leadfield.npy'
    print('loading leadfield file: %s' % sub_file_leadfield)
    lm = np.load(sub_file_leadfield, allow_pickle=True)

    out1 = np.ones(output_size)

    # Create model instance with Param objects - use SingleGainJRTRModel for single-pathway
    model = JansenRitModel(
        node_size=node_size,
        output_size=output_size,
        TRs_per_window=batch_size,
        tr=tr,
        step_size=step_size,
        sc=sc,
        dist=dist,
        # Fixed parameters
        A=Const(3.25),
        B=Const(22),
        vmax=Const(5),
        v0=Const(6),
        r=Const(0.56),
        cy0=Const(0.0005),
        ki=Const(ki0),
        # Learnable parameters with ReLU transform and Gaussian reg
        a=Param(101, t=ReluT(1.), reg=GaussianReg(101, 2.0, True)),
        b=Param(51, t=ReluT(1.), reg=GaussianReg(51, 1.0, True)),
        g=Param(1000, t=ReluT(0.01), reg=GaussianReg(1000, 10., True)),
        c1=Param(135, t=ReluT(0.01), reg=GaussianReg(135, 5., True)),
        c2=Param(135 * 0.8, t=ReluT(0.01), reg=GaussianReg(135 * 0.8, 2.5, True)),
        c3=Param(135 * 0.25, t=ReluT(0.01), reg=GaussianReg(135 * 0.25, 1.25, True)),
        c4=Param(135 * 0.25, t=ReluT(0.01), reg=GaussianReg(135 * 0.25, 1.25, True)),
        std_in=Param(250, t=ReluT(150.0)),
        y0=Param(np.full(output_size, 2.), reg=GaussianReg(2. * out1, 0.5 * out1, True)),
        mu=Param(2.5, t=ReluT(1.5), reg=GaussianReg(2.5, 0.4, True)),
        k=Param(10, t=ReluT(0.5), reg=GaussianReg(10, 3.33, True)),
        # Connectivity and readout
        lm=Param(lm),
        w_bb=Param(sc + np.random.randn(node_size, node_size) * 0.02),
        state_init=lambda s, **kwargs: jnp.asarray(np.random.uniform(0.5, 2., s), dtype=brainstate.environ.dftype()),
        delay_init=lambda s, **kwargs: jnp.asarray(np.random.uniform(0.5, 2., s), dtype=brainstate.environ.dftype()),
    )

    print("   Model initialized successfully")

    # =============================================================================
    # 5. TRAIN MODEL
    # =============================================================================
    print("\n4. Training model...")
    print("   (This may take several minutes depending on hardware)")

    # Prepare training data
    ts_raw = data_mean[:, 900:1300]  # (output_size, 400)
    num_windows = ts_raw.shape[1] // batch_size
    ts = ts_raw.T.reshape(num_windows, batch_size, -1)  # (num_windows, batch_size, output_size)

    F = ModelFitting(
        model=model,
        ts=ts,
        num_epoches=num_epochs,
    )

    # Create external stimulus (TMS pulse)
    hidden_size = int(tr / step_size)
    u = np.zeros((400, hidden_size, node_size))  # (time_dim, steps_per_TR, node_size)
    u[110:120] = 1000  # Pulse at 110-120ms

    # Train model
    start_time = time.time()
    train_loss = F.train(u=u)
    training_time = time.time() - start_time

    print(f"\n   Training completed in {training_time:.1f} seconds")
    print(f"   Final loss: {train_loss[-1]:.4f}")

    # =============================================================================
    # 6. TEST MODEL
    # =============================================================================
    print("\n5. Testing trained model...")

    # Test model
    E_test, I_test, eeg_test = F.test(base_batch_num, u=u)

    print("   Testing completed")
    print(f"   Simulated EEG shape: {eeg_test.shape}")

    # =============================================================================
    # 7. SAVE RESULTS
    # =============================================================================
    print("\n6. Saving results...")

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)

    # =============================================================================
    # 8. VISUALIZE RESULTS (Optional)
    # =============================================================================
    if visualize:
        print("\n7. Creating visualizations...")

        # EEG Comparison
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))

        ax[0].plot(E_test - I_test, alpha=0.5)
        ax[0].set_title('Test: Source-level EEG (E - I)')
        ax[0].set_xlabel('Time (ms)')
        ax[0].set_ylabel('Activity')

        ax[1].plot(eeg_test, alpha=0.5)
        ax[1].set_title('Test: Simulated Channel EEG')
        ax[1].set_xlabel('Time (ms)')
        ax[1].set_ylabel('Amplitude')

        ax[2].plot(eeg_data[subject_idx].T[900:1300, :], alpha=0.5)
        ax[2].set_title('Empirical EEG')
        ax[2].set_xlabel('Time (ms)')
        ax[2].set_ylabel('Amplitude')

        plt.tight_layout()
        plt.show()

        # Fitted Connectivity Comparison
        w_bb = model.w_bb.value()
        sc_fitted = sc * np.exp(w_bb)
        sc_fitted = np.log1p(0.5 * (sc_fitted + sc_fitted.T))
        sc_fitted = sc_fitted / (np.linalg.norm(sc_fitted) + 1e-8)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        ax[0].imshow(np.log1p(sc), cmap='hot')
        ax[0].set_title('Original Structural Connectivity')
        ax[0].set_xlabel('ROI')
        ax[0].set_ylabel('ROI')

        ax[1].imshow(sc_fitted, cmap='hot')
        ax[1].set_title('Fitted Structural Connectivity')
        ax[1].set_xlabel('ROI')
        ax[1].set_ylabel('ROI')

        plt.tight_layout()
        plt.show()

        # Training Loss History
        plt.figure(figsize=(10, 4))
        plt.plot(train_loss)
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.title('Training Loss History')
        plt.grid(True, alpha=0.3)
        plt.show()

        # MNE Visualization
        try:
            import mne
            print("   - MNE topographic plots...")

            epochs = mne.read_epochs(files_dir + '/all_avg.mat_avg_high_epoched', verbose=False)
            evoked = epochs.average()

            empirical_data = epochs.average()
            empirical_data.data = epochs._data[subject_idx, :, :]

            simulated_data = epochs.average()
            simulated_data.data[:, 900:1300] = eeg_test.T

            ts_args = dict(xlim=[-0.025, 0.3])
            ch, peak_locs1 = evoked.get_peak(ch_type='eeg', tmin=-0.05, tmax=0.04)
            ch, peak_locs2 = evoked.get_peak(ch_type='eeg', tmin=0.02, tmax=0.1)
            ch, peak_locs4 = evoked.get_peak(ch_type='eeg', tmin=0.12, tmax=0.15)
            ch, peak_locs5 = evoked.get_peak(ch_type='eeg', tmin=0.15, tmax=0.20)
            times = [peak_locs1, peak_locs2, peak_locs4, peak_locs5]

            empirical_data.plot_joint(ts_args=ts_args, times=times,
                                      title=f'Empirical TEPs (Subject {subject_idx})', show=False)
            simulated_data.plot_joint(ts_args=ts_args, times=times,
                                      title=f'Simulated TEPs (Subject {subject_idx})')
        except Exception as e:
            print(f"   MNE visualization skipped: {e}")

        print("   Visualizations complete!")


if __name__ == '__main__':
    main()
