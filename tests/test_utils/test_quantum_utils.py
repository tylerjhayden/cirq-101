# ABOUTME: Comprehensive tests for quantum utility functions covering Bell state preparation,
# circuit sampling, expectation values, and plot saving with quantum physics correctness validation.

import pytest
import numpy as np
import cirq
import matplotlib.pyplot as plt
import tempfile
import os
from pathlib import Path
from collections import Counter

from utils.quantum_utils import (
    save_and_show_plot,
    prepare_bell_state,
    simulate_and_sample,
    compute_pauli_expectation
)


class TestSaveAndShowPlot:
    """Test plot saving and display functionality."""

    def test_saves_plot_to_file(self, tmp_path):
        """Plot is saved to the specified file path."""
        # Create a simple plot
        plt.figure()
        plt.plot([1, 2, 3], [1, 4, 9])

        # Save to temporary file
        save_path = tmp_path / "test_plot.png"
        save_and_show_plot(str(save_path), show=False, print_message=False)

        # Verify file exists
        assert save_path.exists()
        assert save_path.stat().st_size > 0

        plt.close()

    def test_respects_dpi_setting(self, tmp_path):
        """Different DPI settings produce different file sizes."""
        plt.figure(figsize=(6, 4))
        plt.plot([1, 2, 3], [1, 4, 9])

        # Save with low DPI
        low_dpi_path = tmp_path / "low_dpi.png"
        save_and_show_plot(str(low_dpi_path), dpi=50, show=False, print_message=False)

        plt.figure(figsize=(6, 4))
        plt.plot([1, 2, 3], [1, 4, 9])

        # Save with high DPI
        high_dpi_path = tmp_path / "high_dpi.png"
        save_and_show_plot(str(high_dpi_path), dpi=300, show=False, print_message=False)

        # High DPI should produce larger file
        assert high_dpi_path.stat().st_size > low_dpi_path.stat().st_size

        plt.close('all')

    def test_print_message_parameter(self, tmp_path, capsys):
        """Print message parameter controls output."""
        plt.figure()
        plt.plot([1, 2, 3])

        save_path = tmp_path / "test.png"

        # With print_message=True
        save_and_show_plot(str(save_path), show=False, print_message=True)
        captured = capsys.readouterr()
        assert "Saved to:" in captured.out

        # With print_message=False
        save_and_show_plot(str(save_path), show=False, print_message=False)
        captured = capsys.readouterr()
        assert "Saved to:" not in captured.out

        plt.close()

    def test_bbox_inches_parameter(self, tmp_path):
        """bbox_inches parameter is accepted and used."""
        plt.figure()
        plt.plot([1, 2, 3])
        plt.ylabel("Test Label That Extends")

        save_path = tmp_path / "test_bbox.png"

        # Should not raise an error
        save_and_show_plot(
            str(save_path),
            bbox_inches='tight',
            show=False,
            print_message=False
        )

        assert save_path.exists()
        plt.close()


class TestPrepareBellState:
    """Test Bell state preparation functionality."""

    def test_phi_plus_state_vector(self):
        """Phi plus produces (|00⟩ + |11⟩)/√2."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = prepare_bell_state(q0, q1, variant='phi_plus')

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)

        # Expected: [1/√2, 0, 0, 1/√2]
        expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        np.testing.assert_allclose(result.final_state_vector, expected, atol=1e-8)

    def test_phi_minus_state_vector(self):
        """Phi minus produces (|00⟩ - |11⟩)/√2."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = prepare_bell_state(q0, q1, variant='phi_minus')

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)

        # Expected: [1/√2, 0, 0, -1/√2]
        expected = np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)])
        np.testing.assert_allclose(result.final_state_vector, expected, atol=1e-8)

    def test_psi_plus_state_vector(self):
        """Psi plus produces (|01⟩ + |10⟩)/√2."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = prepare_bell_state(q0, q1, variant='psi_plus')

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)

        # Expected: [0, 1/√2, 1/√2, 0]
        expected = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0])
        np.testing.assert_allclose(result.final_state_vector, expected, atol=1e-8)

    def test_psi_minus_state_vector(self):
        """Psi minus produces (|01⟩ - |10⟩)/√2."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = prepare_bell_state(q0, q1, variant='psi_minus')

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)

        # Expected: [0, 1/√2, -1/√2, 0]
        expected = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0])
        np.testing.assert_allclose(result.final_state_vector, expected, atol=1e-8)

    def test_all_bell_states_normalized(self):
        """All Bell states are properly normalized."""
        q0, q1 = cirq.LineQubit.range(2)
        simulator = cirq.Simulator()

        for variant in ['phi_plus', 'phi_minus', 'psi_plus', 'psi_minus']:
            circuit = prepare_bell_state(q0, q1, variant=variant)
            result = simulator.simulate(circuit)

            # Check normalization: Σ|ψᵢ|² = 1
            norm = np.sum(np.abs(result.final_state_vector)**2)
            np.testing.assert_allclose(norm, 1.0, atol=1e-8,
                                       err_msg=f"State {variant} not normalized")

    def test_bell_states_maximally_entangled(self):
        """All Bell states show perfect measurement correlation."""
        q0, q1 = cirq.LineQubit.range(2)
        simulator = cirq.Simulator()

        for variant in ['phi_plus', 'phi_minus', 'psi_plus', 'psi_minus']:
            circuit = prepare_bell_state(q0, q1, variant=variant)
            circuit.append(cirq.measure(q0, q1, key='result'))

            result = simulator.run(circuit, repetitions=1000)
            measurements = result.measurements['result']

            # For phi states: q0 and q1 should always match
            # For psi states: q0 and q1 should always differ
            if 'phi' in variant:
                # Check perfect correlation
                correlations = np.sum(measurements[:, 0] == measurements[:, 1])
                assert correlations == 1000, f"{variant} should have perfect correlation"
            else:  # psi states
                # Check perfect anti-correlation
                anti_correlations = np.sum(measurements[:, 0] != measurements[:, 1])
                assert anti_correlations == 1000, f"{variant} should have perfect anti-correlation"

    def test_invalid_variant_raises_error(self):
        """Invalid Bell state variant raises ValueError."""
        q0, q1 = cirq.LineQubit.range(2)

        with pytest.raises(ValueError, match="Invalid variant"):
            prepare_bell_state(q0, q1, variant='invalid_state')

    def test_returns_circuit(self):
        """Function returns a valid Cirq circuit."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = prepare_bell_state(q0, q1, variant='phi_plus')

        assert isinstance(circuit, cirq.Circuit)
        assert len(circuit) > 0

    def test_works_with_different_qubit_types(self):
        """Bell state preparation works with different qubit types."""
        # LineQubits
        q0, q1 = cirq.LineQubit.range(2)
        circuit1 = prepare_bell_state(q0, q1)
        assert isinstance(circuit1, cirq.Circuit)

        # GridQubits
        q2 = cirq.GridQubit(0, 0)
        q3 = cirq.GridQubit(0, 1)
        circuit2 = prepare_bell_state(q2, q3)
        assert isinstance(circuit2, cirq.Circuit)

        # NamedQubits
        q4 = cirq.NamedQubit('a')
        q5 = cirq.NamedQubit('b')
        circuit3 = prepare_bell_state(q4, q5)
        assert isinstance(circuit3, cirq.Circuit)


class TestSimulateAndSample:
    """Test circuit sampling and measurement statistics."""

    def test_basic_sampling(self):
        """Basic circuit sampling returns expected results."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.X(q),  # Prepare |1⟩
            cirq.measure(q, key='result')
        )

        results = simulate_and_sample(circuit, repetitions=100)

        assert 'histogram' in results
        assert 'repetitions' in results
        assert results['repetitions'] == 100

    def test_histogram_accuracy(self):
        """Histogram correctly counts measurement outcomes."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.X(q),  # Always measure |1⟩
            cirq.measure(q, key='result')
        )

        results = simulate_and_sample(circuit, repetitions=1000)
        histogram = results['histogram']

        # Should see only outcome 1
        assert 1 in histogram
        assert histogram[1] == 1000
        assert 0 not in histogram

    def test_superposition_statistics(self):
        """Superposition produces expected statistical distribution."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.H(q),  # Equal superposition
            cirq.measure(q, key='result')
        )

        results = simulate_and_sample(circuit, repetitions=10000)
        histogram = results['histogram']

        # Should see roughly 50/50 split (allow 5% deviation)
        assert 0 in histogram
        assert 1 in histogram
        assert 4500 < histogram[0] < 5500
        assert 4500 < histogram[1] < 5500

    def test_most_common_outcome(self):
        """most_common field returns the most frequent outcome."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.X(q),
            cirq.measure(q, key='result')
        )

        results = simulate_and_sample(circuit, repetitions=100)

        assert 'most_common' in results
        outcome, count = results['most_common']
        assert outcome == 1
        assert count == 100

    def test_custom_measurement_key(self):
        """Function works with custom measurement keys."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.H(q),
            cirq.measure(q, key='my_measurement')
        )

        results = simulate_and_sample(
            circuit,
            repetitions=100,
            measurement_key='my_measurement'
        )

        assert 'histogram' in results
        assert len(results['histogram']) > 0

    def test_invalid_measurement_key_raises_error(self):
        """Invalid measurement key raises ValueError."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.H(q),
            cirq.measure(q, key='result')
        )

        with pytest.raises(ValueError, match="Measurement key"):
            simulate_and_sample(circuit, measurement_key='wrong_key')

    def test_return_measurements_parameter(self):
        """return_measurements parameter includes raw data."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.X(q),
            cirq.measure(q, key='result')
        )

        results = simulate_and_sample(
            circuit,
            repetitions=50,
            return_measurements=True
        )

        assert 'measurements' in results
        assert results['measurements'].shape == (50, 1)
        assert np.all(results['measurements'] == 1)

    def test_return_histogram_false(self):
        """return_histogram=False excludes histogram from results."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.X(q),
            cirq.measure(q, key='result')
        )

        results = simulate_and_sample(
            circuit,
            repetitions=100,
            return_histogram=False
        )

        assert 'histogram' not in results
        assert 'most_common' not in results
        assert 'repetitions' in results

    def test_custom_simulator(self):
        """Function accepts custom simulator instance."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.H(q),
            cirq.measure(q, key='result')
        )

        custom_simulator = cirq.Simulator(seed=42)
        results = simulate_and_sample(
            circuit,
            repetitions=100,
            simulator=custom_simulator
        )

        assert 'histogram' in results

    def test_multi_qubit_sampling(self):
        """Function works correctly with multi-qubit circuits."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1, key='result')
        )

        results = simulate_and_sample(circuit, repetitions=1000)
        histogram = results['histogram']

        # Bell state: should only see |00⟩ (0) and |11⟩ (3)
        assert 0 in histogram
        assert 3 in histogram
        assert 1 not in histogram
        assert 2 not in histogram


class TestComputePauliExpectation:
    """Test Pauli expectation value calculations."""

    def test_z_eigenstate_plus_one(self):
        """Z expectation on |0⟩ equals +1."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.I(q))  # |0⟩ state

        observable = cirq.Z(q)
        expectation = compute_pauli_expectation(circuit, observable)

        np.testing.assert_allclose(expectation, 1.0, atol=1e-8)

    def test_z_eigenstate_minus_one(self):
        """Z expectation on |1⟩ equals -1."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.X(q))  # |1⟩ state

        observable = cirq.Z(q)
        expectation = compute_pauli_expectation(circuit, observable)

        np.testing.assert_allclose(expectation, -1.0, atol=1e-8)

    def test_x_eigenstate_plus(self):
        """X expectation on |+⟩ equals +1."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.H(q))  # |+⟩ state

        observable = cirq.X(q)
        expectation = compute_pauli_expectation(circuit, observable)

        np.testing.assert_allclose(expectation, 1.0, atol=1e-8)

    def test_x_eigenstate_minus(self):
        """X expectation on |-⟩ equals -1."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.X(q),
            cirq.H(q)
        )  # |-⟩ state

        observable = cirq.X(q)
        expectation = compute_pauli_expectation(circuit, observable)

        np.testing.assert_allclose(expectation, -1.0, atol=1e-8)

    def test_superposition_expectation_zero(self):
        """Z expectation on |+⟩ equals 0."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.H(q))  # |+⟩ state

        observable = cirq.Z(q)
        expectation = compute_pauli_expectation(circuit, observable)

        np.testing.assert_allclose(expectation, 0.0, atol=1e-7)

    def test_two_qubit_observable(self):
        """ZZ expectation on Bell state equals +1."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1)
        )

        observable = cirq.Z(q0) * cirq.Z(q1)
        expectation = compute_pauli_expectation(circuit, observable)

        np.testing.assert_allclose(expectation, 1.0, atol=1e-8)

    def test_pauli_string_observable(self):
        """Function works with PauliString observables."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1)
        )

        # Create PauliString directly
        observable = cirq.PauliString({q0: cirq.Z, q1: cirq.Z})
        expectation = compute_pauli_expectation(circuit, observable)

        np.testing.assert_allclose(expectation, 1.0, atol=1e-8)

    def test_custom_simulator(self):
        """Function accepts custom simulator instance."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.H(q))

        observable = cirq.X(q)
        custom_simulator = cirq.Simulator()

        expectation = compute_pauli_expectation(
            circuit,
            observable,
            simulator=custom_simulator
        )

        np.testing.assert_allclose(expectation, 1.0, atol=1e-8)

    def test_custom_qubit_ordering(self):
        """Function respects custom qubit ordering."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1)
        )

        observable = cirq.Z(q0) * cirq.Z(q1)

        # Explicit qubit ordering
        expectation = compute_pauli_expectation(
            circuit,
            observable,
            qubits=[q0, q1]
        )

        np.testing.assert_allclose(expectation, 1.0, atol=1e-8)

    def test_expectation_value_range(self):
        """Expectation values for Pauli operators lie in [-1, +1]."""
        q = cirq.LineQubit(0)
        simulator = cirq.Simulator()

        # Test various random states
        for theta in np.linspace(0, 2*np.pi, 10):
            circuit = cirq.Circuit(cirq.ry(theta)(q))

            for obs_gate in [cirq.X, cirq.Y, cirq.Z]:
                observable = obs_gate(q)
                expectation = compute_pauli_expectation(circuit, observable)

                # Expectation should be in [-1, 1]
                assert -1.0 <= expectation <= 1.0, \
                    f"Expectation {expectation} outside [-1, 1] for theta={theta}"

    def test_returns_real_value(self):
        """Function returns a real float, not complex."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.H(q))

        observable = cirq.Z(q)
        expectation = compute_pauli_expectation(circuit, observable)

        assert isinstance(expectation, float)
        assert not isinstance(expectation, complex)

    def test_y_operator_expectation(self):
        """Y expectation values are correctly calculated."""
        q = cirq.LineQubit(0)

        # |0⟩ + i|1⟩ is +1 eigenstate of Y
        circuit = cirq.Circuit(
            cirq.H(q),
            cirq.S(q)
        )

        observable = cirq.Y(q)
        expectation = compute_pauli_expectation(circuit, observable)

        np.testing.assert_allclose(expectation, 1.0, atol=1e-8)

    def test_three_qubit_observable(self):
        """Function works with three-qubit observables."""
        q0, q1, q2 = cirq.LineQubit.range(3)
        # GHZ state
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.CNOT(q0, q2)
        )

        observable = cirq.Z(q0) * cirq.Z(q1) * cirq.Z(q2)
        expectation = compute_pauli_expectation(circuit, observable)

        # GHZ state = (|000⟩ + |111⟩)/√2
        # ZZZ|000⟩ = +|000⟩ and ZZZ|111⟩ = -|111⟩
        # Therefore ⟨ZZZ⟩ = (1/2)(+1) + (1/2)(-1) = 0
        np.testing.assert_allclose(expectation, 0.0, atol=1e-8)


class TestQuantumPhysicsCorrectness:
    """Test that utilities respect fundamental quantum mechanics."""

    def test_bell_states_orthogonal(self):
        """Different Bell states are orthogonal (⟨ψ|φ⟩ = 0)."""
        q0, q1 = cirq.LineQubit.range(2)
        simulator = cirq.Simulator()

        # Get all Bell state vectors
        states = {}
        for variant in ['phi_plus', 'phi_minus', 'psi_plus', 'psi_minus']:
            circuit = prepare_bell_state(q0, q1, variant=variant)
            result = simulator.simulate(circuit)
            states[variant] = result.final_state_vector

        # Check orthogonality between different states
        variants = ['phi_plus', 'phi_minus', 'psi_plus', 'psi_minus']
        for i, v1 in enumerate(variants):
            for v2 in variants[i+1:]:
                inner_product = np.abs(np.vdot(states[v1], states[v2]))
                np.testing.assert_allclose(inner_product, 0.0, atol=1e-8,
                    err_msg=f"{v1} and {v2} should be orthogonal")

    def test_expectation_hermiticity(self):
        """Expectation values of Hermitian operators are real."""
        q = cirq.LineQubit(0)

        # Test with various states
        for gate in [cirq.I, cirq.X, cirq.H, cirq.S]:
            circuit = cirq.Circuit(gate(q))

            # Pauli operators are Hermitian
            for obs_gate in [cirq.X, cirq.Y, cirq.Z]:
                observable = obs_gate(q)
                expectation = compute_pauli_expectation(circuit, observable)

                # Should be real
                assert isinstance(expectation, (int, float))
                # If it were complex, imaginary part would not be zero
                # But we return float, so just verify type

    def test_measurement_probabilities_sum_to_one(self):
        """Measurement outcome probabilities sum to 1."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.ry(np.pi/3)(q1),
            cirq.measure(q0, q1, key='result')
        )

        results = simulate_and_sample(circuit, repetitions=10000)
        histogram = results['histogram']

        # Total counts should equal repetitions
        total_counts = sum(histogram.values())
        assert total_counts == 10000

        # Probabilities should sum to 1
        probabilities = np.array(list(histogram.values())) / total_counts
        np.testing.assert_allclose(np.sum(probabilities), 1.0, atol=1e-10)

    def test_unitarity_preservation(self):
        """Bell state preparation preserves state vector normalization."""
        q0, q1 = cirq.LineQubit.range(2)
        simulator = cirq.Simulator()

        for variant in ['phi_plus', 'phi_minus', 'psi_plus', 'psi_minus']:
            circuit = prepare_bell_state(q0, q1, variant=variant)

            # Apply some additional unitary operations
            circuit.append([
                cirq.ry(0.5)(q0),
                cirq.rz(0.3)(q1),
                cirq.CNOT(q0, q1)
            ])

            result = simulator.simulate(circuit)
            norm = np.sum(np.abs(result.final_state_vector)**2)

            # Norm should still be 1 (unitarity preserved)
            # Use slightly looser tolerance to account for floating point precision
            np.testing.assert_allclose(norm, 1.0, atol=1e-6,
                err_msg=f"Unitarity not preserved for {variant}")
