# ABOUTME: Tests for Section 1.3 - Circuits covering moments, circuit construction, insert strategies, and circuit manipulation

import pytest
import numpy as np
import cirq
from part1_cirq_sdk.section_1_3_circuits import (
    demonstrate_moments,
    demonstrate_circuit_construction,
    demonstrate_insert_strategies,
    demonstrate_circuit_manipulation,
    create_bell_state_circuit,
    compare_insert_strategies,
    get_circuit_depth,
    get_circuit_moments,
)


class TestMoments:
    """Test moment creation and properties."""

    def test_moment_creation(self):
        """Moments group operations that execute simultaneously."""
        q0, q1 = cirq.LineQubit.range(2)
        moment = cirq.Moment([cirq.H(q0), cirq.X(q1)])

        assert len(moment) == 2
        assert isinstance(moment, cirq.Moment)

    def test_moment_disjoint_qubits(self):
        """Operations in a moment must act on disjoint qubits."""
        q0, q1, q2 = cirq.LineQubit.range(3)

        # Valid moment - disjoint qubits
        moment = cirq.Moment([cirq.H(q0), cirq.X(q1), cirq.Y(q2)])
        assert len(moment) == 3

    def test_moment_overlapping_qubits_raises(self):
        """Moments cannot contain operations on the same qubit."""
        q = cirq.LineQubit(0)

        with pytest.raises(ValueError):
            cirq.Moment([cirq.H(q), cirq.X(q)])

    def test_moment_iteration(self):
        """Moments are iterable over their operations."""
        q0, q1 = cirq.LineQubit.range(2)
        ops = [cirq.H(q0), cirq.X(q1)]
        moment = cirq.Moment(ops)

        moment_ops = list(moment)
        assert len(moment_ops) == 2


class TestCircuitConstruction:
    """Test circuit creation and basic properties."""

    def test_empty_circuit(self):
        """Circuits can be created empty."""
        circuit = cirq.Circuit()
        assert len(circuit) == 0

    def test_circuit_from_operations(self):
        """Circuits can be created from a list of operations."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))

        assert len(circuit) > 0
        assert isinstance(circuit, cirq.Circuit)

    def test_circuit_from_moments(self):
        """Circuits can be created from moments."""
        q0, q1 = cirq.LineQubit.range(2)
        moment1 = cirq.Moment([cirq.H(q0)])
        moment2 = cirq.Moment([cirq.CNOT(q0, q1)])

        circuit = cirq.Circuit(moment1, moment2)
        assert len(circuit) == 2

    def test_circuit_append(self):
        """Operations can be appended to circuits."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit()
        circuit.append(cirq.H(q))
        circuit.append(cirq.X(q))

        assert len(circuit) >= 2

    def test_create_bell_state_circuit(self):
        """Test Bell state circuit creation and verify quantum state."""
        circuit = create_bell_state_circuit()

        assert isinstance(circuit, cirq.Circuit)
        # Bell state requires H and CNOT
        assert len(circuit) >= 1

        # Verify the circuit produces the correct Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        # using ideal statevector simulation (per CLAUDE.md:102)
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        statevector = result.final_state_vector

        # Expected amplitudes for Bell state |Φ+⟩
        expected = np.zeros(4, dtype=complex)
        expected[0] = 1/np.sqrt(2)  # |00⟩
        expected[3] = 1/np.sqrt(2)  # |11⟩

        # Verify statevector matches expected Bell state
        np.testing.assert_allclose(statevector, expected, atol=1e-7)


class TestInsertStrategies:
    """Test different circuit insertion strategies."""

    def test_new_then_inline_strategy(self):
        """NEW_THEN_INLINE creates new moment, then tries to inline."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit()

        # First operation creates new moment
        circuit.append(cirq.H(q0), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        # Second operation creates another new moment (NEW takes precedence)
        circuit.append(cirq.H(q1), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

        # NEW_THEN_INLINE creates new moments
        assert len(circuit) == 2
        assert len(list(circuit.all_operations())) == 2

    def test_new_then_inline_conflict(self):
        """NEW_THEN_INLINE creates new moment for conflicting qubits."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit()

        circuit.append(cirq.H(q), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        circuit.append(cirq.X(q), strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

        # Two operations on same qubit must be in different moments
        assert len(circuit) == 2

    def test_earliest_strategy(self):
        """EARLIEST finds the first moment where operation fits."""
        q0, q1, q2 = cirq.LineQubit.range(3)
        circuit = cirq.Circuit()

        # Add operations
        circuit.append(cirq.H(q0), strategy=cirq.InsertStrategy.EARLIEST)
        circuit.append(cirq.H(q1), strategy=cirq.InsertStrategy.EARLIEST)
        circuit.append(cirq.X(q0), strategy=cirq.InsertStrategy.EARLIEST)
        circuit.append(cirq.H(q2), strategy=cirq.InsertStrategy.EARLIEST)

        # q2 operation should slide back to first moment
        assert len(circuit) <= 2

    def test_compare_insert_strategies(self):
        """Test function comparing different insert strategies."""
        circuits = compare_insert_strategies()

        assert 'NEW_THEN_INLINE' in circuits
        assert 'EARLIEST' in circuits
        assert isinstance(circuits['NEW_THEN_INLINE'], cirq.Circuit)
        assert isinstance(circuits['EARLIEST'], cirq.Circuit)

        # EARLIEST should be more compact (fewer or equal moments)
        assert len(circuits['EARLIEST']) <= len(circuits['NEW_THEN_INLINE'])

    def test_inline_strategy(self):
        """INLINE strategy adds to existing moment."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit()

        # Using INLINE strategy
        circuit.append(cirq.H(q0), strategy=cirq.InsertStrategy.INLINE)
        circuit.append(cirq.H(q1), strategy=cirq.InsertStrategy.INLINE)

        # Both operations should be in same moment with INLINE
        assert len(circuit) == 1


class TestCircuitManipulation:
    """Test circuit inspection and manipulation."""

    def test_circuit_depth(self):
        """Circuit depth is the number of moments."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.H(q), cirq.X(q), cirq.Y(q))

        depth = get_circuit_depth(circuit)
        assert depth == len(circuit)
        assert isinstance(depth, int)

    def test_circuit_moments(self):
        """Circuits can be iterated as moments."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))

        moments = get_circuit_moments(circuit)
        assert isinstance(moments, list)
        assert len(moments) > 0
        assert all(isinstance(m, cirq.Moment) for m in moments)

    def test_circuit_all_operations(self):
        """Circuits provide access to all operations."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))

        ops = list(circuit.all_operations())
        assert len(ops) >= 2
        assert all(isinstance(op, cirq.Operation) for op in ops)

    def test_circuit_all_qubits(self):
        """Circuits track which qubits are used."""
        q0, q1, q2 = cirq.LineQubit.range(3)
        circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))

        qubits = circuit.all_qubits()
        assert q0 in qubits
        assert q1 in qubits
        assert q2 not in qubits

    def test_circuit_concatenation(self):
        """Circuits can be concatenated."""
        q = cirq.LineQubit(0)
        circuit1 = cirq.Circuit(cirq.H(q))
        circuit2 = cirq.Circuit(cirq.X(q))

        combined = circuit1 + circuit2
        assert len(combined) >= len(circuit1) + len(circuit2) - 1

    def test_circuit_slicing(self):
        """Circuits support slicing operations."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.H(q), cirq.X(q), cirq.Y(q), cirq.Z(q))

        # Get first two moments
        sliced = circuit[0:2]
        assert isinstance(sliced, cirq.Circuit)
        assert len(sliced) == 2

    def test_circuit_insertion(self):
        """Operations can be inserted at specific positions."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(cirq.H(q), cirq.Z(q))

        # Insert X gate between H and Z
        circuit.insert(1, cirq.X(q))

        ops = list(circuit.all_operations())
        assert len(ops) == 3

    def test_circuit_clear_operations(self):
        """Circuit operations can be cleared using clear_operations_touching."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(cirq.H(q0), cirq.X(q0), cirq.H(q1))

        # Clear operations touching q0
        circuit.clear_operations_touching([q0], [0, 1])

        # Only q1 operation should remain
        remaining_qubits = circuit.all_qubits()
        assert q1 in remaining_qubits
        assert q0 not in remaining_qubits


class TestCircuitStringRepresentation:
    """Test circuit string representations."""

    def test_circuit_str(self):
        """Circuits have readable string representation."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))

        circuit_str = str(circuit)
        assert isinstance(circuit_str, str)
        assert len(circuit_str) > 0

    def test_circuit_diagram(self):
        """Circuits can be displayed as text diagrams."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))

        diagram = circuit.to_text_diagram()
        assert isinstance(diagram, str)
        # Should contain qubit labels
        assert 'q(0)' in diagram or '0' in diagram


class TestComplexCircuits:
    """Test more complex circuit patterns."""

    def test_parallel_operations(self):
        """Multiple gates can execute in parallel."""
        qubits = cirq.LineQubit.range(4)
        circuit = cirq.Circuit()

        # All Hadamards in parallel
        circuit.append([cirq.H(q) for q in qubits])

        assert len(circuit) == 1  # Single moment
        assert len(list(circuit.all_operations())) == 4

    def test_sequential_operations(self):
        """Sequential operations on same qubit."""
        q = cirq.LineQubit(0)
        circuit = cirq.Circuit()

        gates = [cirq.H, cirq.X, cirq.Y, cirq.Z]
        for gate in gates:
            circuit.append(gate(q))

        assert len(circuit) == len(gates)

    def test_entangling_circuit(self):
        """Test circuit with entangling gates and verify GHZ state."""
        qubits = cirq.LineQubit.range(3)
        circuit = cirq.Circuit()

        # GHZ state preparation
        circuit.append(cirq.H(qubits[0]))
        circuit.append(cirq.CNOT(qubits[0], qubits[1]))
        circuit.append(cirq.CNOT(qubits[1], qubits[2]))

        assert len(circuit) >= 2
        assert len(list(circuit.all_operations())) == 3

        # Verify the circuit produces the correct GHZ state |GHZ⟩ = (|000⟩ + |111⟩)/√2
        # using ideal statevector simulation (per CLAUDE.md:102)
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        statevector = result.final_state_vector

        # Expected amplitudes for GHZ state
        expected = np.zeros(8, dtype=complex)
        expected[0] = 1/np.sqrt(2)  # |000⟩
        expected[7] = 1/np.sqrt(2)  # |111⟩

        # Verify statevector matches expected GHZ state
        np.testing.assert_allclose(statevector, expected, atol=1e-7)

    def test_measurement_at_end(self):
        """Measurements typically appear at circuit end."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit()

        circuit.append(cirq.H(q0))
        circuit.append(cirq.CNOT(q0, q1))
        circuit.append(cirq.measure(q0, q1, key='result'))

        # Last moment should contain measurement
        last_moment = list(circuit)[-1]
        ops = list(last_moment)
        assert any(cirq.is_measurement(op) for op in ops)


class TestMomentStructure:
    """Test understanding of moment structure in circuits."""

    def test_moment_qubit_coverage(self):
        """Check which qubits are active in each moment."""
        q0, q1, q2 = cirq.LineQubit.range(3)
        circuit = cirq.Circuit()

        circuit.append([cirq.H(q0), cirq.H(q1)])
        circuit.append(cirq.CNOT(q0, q1))
        circuit.append(cirq.X(q2))

        # First moment: q0 and q1
        moment0 = list(circuit)[0]
        qubits0 = set()
        for op in moment0:
            qubits0.update(op.qubits)
        assert q0 in qubits0
        assert q1 in qubits0

    def test_empty_moment_not_allowed(self):
        """Circuits should not contain empty moments."""
        circuit = cirq.Circuit()

        # Adding empty list should not create a moment
        circuit.append([])
        assert len(circuit) == 0

    def test_moment_with_two_qubit_gate(self):
        """Two-qubit gates occupy both qubits in a moment."""
        q0, q1, q2 = cirq.LineQubit.range(3)

        # CNOT on q0,q1 means q2 can still be used
        moment = cirq.Moment([cirq.CNOT(q0, q1), cirq.H(q2)])
        assert len(moment) == 2


class TestCircuitProperties:
    """Test various circuit property queries."""

    def test_circuit_operation_count(self):
        """Count total operations in circuit."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.H(q1),
            cirq.CNOT(q0, q1),
            cirq.X(q0)
        )

        op_count = len(list(circuit.all_operations()))
        assert op_count == 4

    def test_circuit_two_qubit_gate_count(self):
        """Count two-qubit gates separately."""
        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.H(q1),
            cirq.CNOT(q0, q1),
            cirq.CNOT(q0, q1),
        )

        two_qubit_gates = sum(1 for op in circuit.all_operations()
                              if len(op.qubits) == 2)
        assert two_qubit_gates == 2

    def test_circuit_qubit_count(self):
        """Count unique qubits used in circuit."""
        qubits = cirq.LineQubit.range(5)
        circuit = cirq.Circuit(
            cirq.H(qubits[0]),
            cirq.CNOT(qubits[0], qubits[2]),
            cirq.X(qubits[4])
        )

        unique_qubits = circuit.all_qubits()
        assert len(unique_qubits) == 3
