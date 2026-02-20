# ABOUTME: Tests for Section 1.5 - Noisy Simulation covering noise channels, density matrices, and noisy vs ideal comparisons

import pytest
import numpy as np
import cirq


class TestNoiseChannels:
    """Test different noise channel implementations."""

    def test_bit_flip_channel(self):
        """Bit flip channel applies random X errors."""
        q = cirq.LineQubit(0)
        p = 0.3

        # Create bit flip channel
        bit_flip = cirq.bit_flip(p)
        circuit = cirq.Circuit(cirq.H(q), bit_flip(q))

        # Bit flip should be a valid operation
        assert isinstance(bit_flip(q), cirq.Operation)

        # Verify Kraus operators sum correctly: sum(A_k† A_k) = I
        kraus_ops = cirq.kraus(bit_flip)
        sum_kraus = sum(K.conj().T @ K for K in kraus_ops)
        np.testing.assert_allclose(sum_kraus, np.eye(2), atol=1e-8)

    def test_depolarize_channel(self):
        """Depolarizing channel applies symmetric white noise."""
        q = cirq.LineQubit(0)
        p = 0.1

        # Create depolarizing channel
        depolarize = cirq.depolarize(p)
        circuit = cirq.Circuit(cirq.H(q), depolarize(q))

        # Verify it's a valid operation
        assert isinstance(depolarize(q), cirq.Operation)

        # Verify Kraus operators
        kraus_ops = cirq.kraus(depolarize)
        sum_kraus = sum(K.conj().T @ K for K in kraus_ops)
        np.testing.assert_allclose(sum_kraus, np.eye(2), atol=1e-8)

    def test_amplitude_damping_channel(self):
        """Amplitude damping models energy relaxation (T1 decay)."""
        q = cirq.LineQubit(0)
        gamma = 0.2

        # Create amplitude damping channel
        amp_damp = cirq.amplitude_damp(gamma)
        circuit = cirq.Circuit(cirq.X(q), amp_damp(q))

        # Verify it's a valid operation
        assert isinstance(amp_damp(q), cirq.Operation)

        # Verify Kraus operators
        kraus_ops = cirq.kraus(amp_damp)
        sum_kraus = sum(K.conj().T @ K for K in kraus_ops)
        np.testing.assert_allclose(sum_kraus, np.eye(2), atol=1e-8)

    def test_phase_damping_channel(self):
        """Phase damping models phase relaxation (T2 decay)."""
        q = cirq.LineQubit(0)
        gamma = 0.15

        # Create phase damping channel
        phase_damp = cirq.phase_damp(gamma)
        circuit = cirq.Circuit(cirq.H(q), phase_damp(q))

        # Verify it's a valid operation
        assert isinstance(phase_damp(q), cirq.Operation)

        # Verify Kraus operators
        kraus_ops = cirq.kraus(phase_damp)
        sum_kraus = sum(K.conj().T @ K for K in kraus_ops)
        np.testing.assert_allclose(sum_kraus, np.eye(2), atol=1e-8)

    def test_noise_strength_affects_output(self):
        """Higher noise parameter should cause more degradation."""
        q = cirq.LineQubit(0)

        # Create circuit with superposition
        base_circuit = cirq.Circuit(cirq.H(q))

        # Simulate with different noise levels
        simulator = cirq.DensityMatrixSimulator()

        # Low noise
        low_noise_circuit = base_circuit + cirq.Circuit(cirq.depolarize(0.01)(q))
        low_result = simulator.simulate(low_noise_circuit)

        # High noise
        high_noise_circuit = base_circuit + cirq.Circuit(cirq.depolarize(0.5)(q))
        high_result = simulator.simulate(high_noise_circuit)

        # Calculate purity: Tr(ρ²) - pure states have purity 1
        low_purity = np.trace(low_result.final_density_matrix @ low_result.final_density_matrix).real
        high_purity = np.trace(high_result.final_density_matrix @ high_result.final_density_matrix).real

        # High noise should reduce purity more
        assert low_purity > high_purity


class TestDensityMatrixSimulation:
    """Test density matrix simulation functionality."""

    def test_density_matrix_simulator_exists(self):
        """DensityMatrixSimulator can be instantiated."""
        simulator = cirq.DensityMatrixSimulator()
        assert simulator is not None

    def test_pure_state_density_matrix(self):
        """Pure states have density matrix ρ = |ψ⟩⟨ψ|."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.H(q))

        # Simulate with state vector
        sv_simulator = cirq.Simulator()
        sv_result = sv_simulator.simulate(circuit)
        psi = sv_result.final_state_vector

        # Simulate with density matrix
        dm_simulator = cirq.DensityMatrixSimulator()
        dm_result = dm_simulator.simulate(circuit)
        rho = dm_result.final_density_matrix

        # Verify ρ = |ψ⟩⟨ψ|
        expected_rho = np.outer(psi, psi.conj())
        np.testing.assert_allclose(rho, expected_rho, atol=1e-8)

    def test_density_matrix_properties(self):
        """Density matrices must be Hermitian, PSD, and have trace 1."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.H(q),
            cirq.depolarize(0.1)(q)
        )

        simulator = cirq.DensityMatrixSimulator()
        result = simulator.simulate(circuit)
        rho = result.final_density_matrix

        # Test Hermiticity: ρ = ρ†
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-8)

        # Test trace: Tr(ρ) = 1
        trace = np.trace(rho)
        np.testing.assert_allclose(trace, 1.0, atol=1e-6)

        # Test positive semi-definite: all eigenvalues >= 0
        eigenvalues = np.linalg.eigvalsh(rho)
        assert np.all(eigenvalues >= -1e-10)

    def test_mixed_state_has_reduced_purity(self):
        """Noisy evolution creates mixed states with purity < 1."""
        q = cirq.LineQubit(0)

        # Pure state
        pure_circuit = cirq.Circuit(cirq.H(q))

        # Mixed state (with noise)
        mixed_circuit = cirq.Circuit(
            cirq.H(q),
            cirq.depolarize(0.3)(q)
        )

        simulator = cirq.DensityMatrixSimulator()

        pure_result = simulator.simulate(pure_circuit)
        pure_rho = pure_result.final_density_matrix
        pure_purity = np.trace(pure_rho @ pure_rho).real

        mixed_result = simulator.simulate(mixed_circuit)
        mixed_rho = mixed_result.final_density_matrix
        mixed_purity = np.trace(mixed_rho @ mixed_rho).real

        # Pure state should have purity ≈ 1
        np.testing.assert_allclose(pure_purity, 1.0, atol=1e-6)

        # Mixed state should have purity < 1
        assert mixed_purity < pure_purity


class TestNoisyVsIdealSimulation:
    """Test comparison between ideal and noisy simulations."""

    def test_ideal_bell_state(self):
        """Ideal Bell state shows perfect correlations."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1, key='result')
        )

        simulator = cirq.Simulator(seed=42)
        results = simulator.run(circuit, repetitions=1000)
        counts = results.histogram(key='result')

        # Should only see |00⟩ and |11⟩
        assert 0 in counts  # |00⟩
        assert 3 in counts  # |11⟩

        # Should not see |01⟩ or |10⟩ (or very rarely due to sampling)
        assert counts.get(1, 0) < 50  # |01⟩ should be rare
        assert counts.get(2, 0) < 50  # |10⟩ should be rare

    def test_noisy_bell_state(self):
        """Noisy Bell state shows degraded correlations."""
        q0, q1 = cirq.LineQubit.range(2)

        # Add noise after entanglement
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.depolarize(0.1)(q0),
            cirq.depolarize(0.1)(q1),
            cirq.measure(q0, q1, key='result')
        )

        simulator = cirq.DensityMatrixSimulator()
        results = simulator.run(circuit, repetitions=1000)
        counts = results.histogram(key='result')

        # Should see all outcomes due to noise
        # At minimum, should have some |00⟩ and |11⟩
        assert len(counts) >= 2

    def test_amplitude_damping_increases_ground_state(self):
        """Amplitude damping should increase |0⟩ population."""
        q = cirq.LineQubit(0)

        # Prepare |1⟩ state
        ideal_circuit = cirq.Circuit(cirq.X(q))
        noisy_circuit = cirq.Circuit(
            cirq.X(q),
            cirq.amplitude_damp(0.5)(q)
        )

        simulator = cirq.DensityMatrixSimulator()

        ideal_result = simulator.simulate(ideal_circuit)
        noisy_result = simulator.simulate(noisy_circuit)

        # Extract |0⟩ population from density matrix diagonal
        ideal_pop_0 = ideal_result.final_density_matrix[0, 0].real
        noisy_pop_0 = noisy_result.final_density_matrix[0, 0].real

        # Amplitude damping should increase ground state population
        assert noisy_pop_0 > ideal_pop_0

    def test_phase_damping_destroys_coherence(self):
        """Phase damping should reduce off-diagonal elements."""
        q = cirq.LineQubit(0)

        # Prepare superposition
        ideal_circuit = cirq.Circuit(cirq.H(q))
        noisy_circuit = cirq.Circuit(
            cirq.H(q),
            cirq.phase_damp(0.5)(q)
        )

        simulator = cirq.DensityMatrixSimulator()

        ideal_result = simulator.simulate(ideal_circuit)
        noisy_result = simulator.simulate(noisy_circuit)

        # Check off-diagonal coherence terms
        ideal_coherence = abs(ideal_result.final_density_matrix[0, 1])
        noisy_coherence = abs(noisy_result.final_density_matrix[0, 1])

        # Phase damping should reduce coherence
        assert noisy_coherence < ideal_coherence

    def test_depolarizing_approaches_maximally_mixed(self):
        """Strong depolarizing noise approaches maximally mixed state."""
        q = cirq.LineQubit(0)

        # Start with pure state
        circuit = cirq.Circuit(
            cirq.X(q),
            cirq.depolarize(0.99)(q)  # Very strong noise
        )

        simulator = cirq.DensityMatrixSimulator()
        result = simulator.simulate(circuit)
        rho = result.final_density_matrix

        # Maximally mixed state is I/2
        maximally_mixed = np.eye(2) / 2

        # Should be close to maximally mixed (not exact due to p < 1)
        distance = np.linalg.norm(rho - maximally_mixed)
        assert distance < 0.5  # Reasonable threshold


class TestKrausOperators:
    """Test Kraus operator representations of noise."""

    def test_kraus_representation_completeness(self):
        """Kraus operators must satisfy completeness relation."""
        channels = [
            cirq.bit_flip(0.2),
            cirq.depolarize(0.1),
            cirq.amplitude_damp(0.3),
            cirq.phase_damp(0.25)
        ]

        for channel in channels:
            kraus_ops = cirq.kraus(channel)

            # Verify completeness: sum(A_k† A_k) = I
            sum_kraus = sum(K.conj().T @ K for K in kraus_ops)
            np.testing.assert_allclose(sum_kraus, np.eye(2), atol=1e-8)

    def test_channel_preserves_trace(self):
        """Quantum channels must preserve trace of density matrix."""
        q = cirq.LineQubit(0)

        # Start with arbitrary density matrix
        initial_circuit = cirq.Circuit(cirq.ry(np.pi/3)(q))

        channels = [
            cirq.bit_flip(0.2),
            cirq.depolarize(0.1),
            cirq.amplitude_damp(0.3),
            cirq.phase_damp(0.25)
        ]

        simulator = cirq.DensityMatrixSimulator()
        initial_result = simulator.simulate(initial_circuit)
        initial_trace = np.trace(initial_result.final_density_matrix).real

        for channel in channels:
            circuit = initial_circuit + cirq.Circuit(channel(q))
            result = simulator.simulate(circuit)
            final_trace = np.trace(result.final_density_matrix).real

            # Trace should be preserved
            np.testing.assert_allclose(final_trace, initial_trace, atol=1e-8)


class TestMultiQubitNoiseSimulation:
    """Test noisy simulation with multiple qubits."""

    def test_independent_noise_on_qubits(self):
        """Each qubit can have independent noise."""
        q0, q1 = cirq.LineQubit.range(2)

        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.H(q1),
            cirq.depolarize(0.1)(q0),
            cirq.bit_flip(0.2)(q1)
        )

        simulator = cirq.DensityMatrixSimulator()
        result = simulator.simulate(circuit)

        # Should produce valid 4x4 density matrix
        assert result.final_density_matrix.shape == (4, 4)

        # Verify density matrix properties
        rho = result.final_density_matrix
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-8)  # Hermitian
        np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-6)  # Unit trace

    def test_noise_after_entanglement(self):
        """Noise applied after entanglement degrades correlations."""
        q0, q1 = cirq.LineQubit.range(2)

        # Create Bell state then add noise
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.depolarize(0.2)(q0),
            cirq.depolarize(0.2)(q1)
        )

        simulator = cirq.DensityMatrixSimulator()
        result = simulator.simulate(circuit)
        rho = result.final_density_matrix

        # Calculate purity - should be less than 1 due to noise
        purity = np.trace(rho @ rho).real
        assert purity < 1.0
        assert purity > 0.0  # Should still have some coherence
