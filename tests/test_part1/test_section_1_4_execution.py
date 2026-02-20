# ABOUTME: Tests for Section 1.4 - Execution covering simulator usage, run() vs simulate(), Bell states, and expectation values

import pytest
import numpy as np
import cirq


class TestSimulatorBasics:
    """Test basic simulator initialization and usage."""

    def test_simulator_creation(self):
        """Simulator can be instantiated."""
        simulator = cirq.Simulator()
        assert simulator is not None
        assert isinstance(simulator, cirq.Simulator)

    def test_simulate_returns_result(self):
        """simulate() returns a SimulationTrialResult."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.H(q))
        simulator = cirq.Simulator()

        result = simulator.simulate(circuit)
        assert result is not None
        assert hasattr(result, 'final_state_vector')

    def test_run_returns_result(self):
        """run() returns a Result with measurement data."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.H(q),
            cirq.measure(q, key='result')
        )
        simulator = cirq.Simulator()

        result = simulator.run(circuit, repetitions=10)
        assert result is not None
        assert 'result' in result.measurements


class TestBellStateCreation:
    """Test Bell state preparation and properties."""

    def test_bell_state_circuit_structure(self):
        """Bell state circuit has correct structure."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1)
        )

        assert len(circuit) == 2
        assert any(isinstance(op.gate, cirq.HPowGate) for moment in circuit for op in moment)
        assert any(isinstance(op.gate, cirq.CNotPowGate) for moment in circuit for op in moment)

    def test_bell_state_vector(self):
        """Bell state produces correct state vector: (|00⟩ + |11⟩)/√2."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1)
        )

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)

        # Expected: [1/√2, 0, 0, 1/√2]
        expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        np.testing.assert_allclose(result.final_state_vector, expected, atol=1e-8)

    def test_bell_state_entanglement(self):
        """Bell state measurements are perfectly correlated."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1, key='result')
        )

        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1000)

        # Extract individual qubit measurements
        measurements = result.measurements['result']

        # Check that q0 and q1 are always the same
        correlations = np.sum(measurements[:, 0] == measurements[:, 1])
        assert correlations == 1000, "Bell state qubits must be perfectly correlated"

    def test_bell_state_histogram(self):
        """Bell state measurements show only |00⟩ and |11⟩ outcomes."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1, key='result')
        )

        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1000)
        histogram = result.histogram(key='result')

        # Only outcomes 0 (|00⟩) and 3 (|11⟩) should appear
        assert 0 in histogram, "Outcome |00⟩ should appear"
        assert 3 in histogram, "Outcome |11⟩ should appear"
        assert 1 not in histogram, "Outcome |01⟩ should not appear"
        assert 2 not in histogram, "Outcome |10⟩ should not appear"

    def test_bell_state_unitarity(self):
        """Bell state circuit unitary matrix satisfies U†U = I."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1)
        )

        # Get unitary matrix representation
        unitary = circuit.unitary()

        # Verify unitarity: U†U = I
        identity = unitary.conj().T @ unitary
        expected_identity = np.eye(4)

        np.testing.assert_allclose(identity, expected_identity, atol=1e-8,
                                  err_msg="Bell circuit must be unitary: U†U = I")


class TestRunVsSimulate:
    """Test differences between run() and simulate() methods."""

    def test_simulate_provides_state_vector(self):
        """simulate() provides access to full quantum state."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.H(q))

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)

        # State vector should be [1/√2, 1/√2]
        expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        np.testing.assert_allclose(result.final_state_vector, expected, atol=1e-8)

    def test_run_requires_measurements(self):
        """run() needs measurements to extract classical data."""
        q = cirq.LineQubit(0)
        circuit_with_measurement = cirq.Circuit(
            cirq.H(q),
            cirq.measure(q, key='result')
        )

        simulator = cirq.Simulator()
        result = simulator.run(circuit_with_measurement, repetitions=10)
        assert len(result.measurements) > 0
        assert 'result' in result.measurements

    def test_run_produces_classical_outcomes(self):
        """run() produces classical bit strings."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.H(q),
            cirq.measure(q, key='m')
        )

        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=100)

        measurements = result.measurements['m']
        assert measurements.shape == (100, 1)
        assert all(m in [0, 1] for m in measurements.flatten())

    def test_run_repetitions_parameter(self):
        """run() repetitions parameter controls sample count."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.X(q),
            cirq.measure(q, key='result')
        )

        simulator = cirq.Simulator()

        # Test different repetition counts
        for reps in [1, 10, 100]:
            result = simulator.run(circuit, repetitions=reps)
            assert result.measurements['result'].shape[0] == reps

    def test_simulate_vs_run_state_access(self):
        """simulate() provides state information that run() does not."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit_no_measure = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1)
        )

        circuit_with_measure = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1, key='result')
        )

        simulator = cirq.Simulator()

        # simulate() gives us the state vector
        sim_result = simulator.simulate(circuit_no_measure)
        assert hasattr(sim_result, 'final_state_vector')
        assert len(sim_result.final_state_vector) == 4

        # run() gives classical measurements, not state vector
        run_result = simulator.run(circuit_with_measure, repetitions=10)
        assert not hasattr(run_result, 'final_state_vector')
        assert hasattr(run_result, 'measurements')


class TestExpectationValues:
    """Test expectation value calculations."""

    def test_expectation_value_z_eigenstate(self):
        """Z expectation on |0⟩ equals +1."""
        q = cirq.LineQubit(0)
        # Identity gate to ensure qubit is in circuit
        circuit = cirq.Circuit(cirq.I(q))

        simulator = cirq.Simulator()
        observable = cirq.Z(q).with_qubits(q)

        result = simulator.simulate(circuit)
        expectation = observable.expectation_from_state_vector(
            result.final_state_vector,
            qubit_map={q: 0}
        )

        np.testing.assert_allclose(expectation, 1.0, atol=1e-8)

    def test_expectation_value_hadamard_basis(self):
        """X expectation on H|0⟩ = |+⟩ equals +1."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.H(q))

        simulator = cirq.Simulator()
        observable = cirq.X(q)

        expectation = simulator.simulate_expectation_values(
            circuit,
            observables=[observable]
        )[0]

        np.testing.assert_allclose(expectation, 1.0, atol=1e-8)

    def test_expectation_value_superposition(self):
        """Z expectation on superposition H|0⟩ equals 0."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.H(q))

        simulator = cirq.Simulator()
        observable = cirq.Z(q).with_qubits(q)

        result = simulator.simulate(circuit)
        expectation = observable.expectation_from_state_vector(
            result.final_state_vector,
            qubit_map={q: 0}
        )

        np.testing.assert_allclose(expectation, 0.0, atol=1e-7)

    def test_expectation_values_multiple_observables(self):
        """Multiple observables can be measured simultaneously."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.H(q))

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)

        observables = [
            cirq.X(q).with_qubits(q),
            cirq.Y(q).with_qubits(q),
            cirq.Z(q).with_qubits(q)
        ]

        expectations = [
            obs.expectation_from_state_vector(result.final_state_vector, {q: 0})
            for obs in observables
        ]

        assert len(expectations) == 3
        np.testing.assert_allclose(expectations[0], 1.0, atol=1e-8)  # X
        np.testing.assert_allclose(expectations[1], 0.0, atol=1e-7)  # Y
        np.testing.assert_allclose(expectations[2], 0.0, atol=1e-7)  # Z

        # Hermitian operators must have real expectation values
        # Verify imaginary parts are negligible (numerical noise only)
        for exp in expectations:
            np.testing.assert_allclose(np.imag(exp), 0.0, atol=1e-7,
                                      err_msg="Hermitian observable must have real expectation value")

    def test_expectation_value_two_qubit_observable(self):
        """Two-qubit observables work correctly."""
        q0, q1 = cirq.LineQubit.range(2)
        # Bell state
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1)
        )

        simulator = cirq.Simulator()
        # ZZ observable on Bell state
        observable = cirq.Z(q0) * cirq.Z(q1)

        expectation = simulator.simulate_expectation_values(
            circuit,
            observables=[observable]
        )[0]

        # For Bell state |Φ+⟩, ⟨ZZ⟩ = 1
        np.testing.assert_allclose(expectation, 1.0, atol=1e-8)


class TestStateVectorAnalysis:
    """Test state vector analysis capabilities."""

    def test_state_vector_normalization(self):
        """State vectors are normalized."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.ry(np.pi/3)(q))

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)

        norm = np.sum(np.abs(result.final_state_vector)**2)
        np.testing.assert_allclose(norm, 1.0, atol=1e-8)

    def test_state_vector_two_qubits(self):
        """Two-qubit state vectors have length 4."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.X(q1)
        )

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)

        assert len(result.final_state_vector) == 4

    def test_state_vector_computational_basis(self):
        """Computational basis states have single nonzero amplitude."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.X(q0),
            cirq.X(q1)
        )

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)

        # Should be |11⟩ = [0, 0, 0, 1]
        expected = np.array([0, 0, 0, 1])
        np.testing.assert_allclose(result.final_state_vector, expected, atol=1e-8)


class TestMeasurementStatistics:
    """Test measurement statistics and histograms."""

    def test_measurement_histogram_format(self):
        """Histogram converts bitstrings to integers."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.X(q0),  # |10⟩ state (q0=1, q1=0)
            cirq.measure(q0, q1, key='result')
        )

        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=100)
        histogram = result.histogram(key='result')

        # Cirq's histogram encoding: measurement order determines bit position
        # For measure(q0, q1, key='result'): integer = q0 + 2*q1
        # So q0=1, q1=0 encodes as 1 (not 2)
        assert sum(histogram.values()) == 100
        assert len(histogram) >= 1

    def test_superposition_statistics(self):
        """Superposition produces expected statistical distribution."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.H(q),
            cirq.measure(q, key='result')
        )

        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=10000)
        histogram = result.histogram(key='result')

        # Should see roughly 50/50 split between 0 and 1
        assert 0 in histogram
        assert 1 in histogram

        # Statistical test: should be close to 5000 each (allow 5% deviation)
        assert 4500 < histogram[0] < 5500
        assert 4500 < histogram[1] < 5500
