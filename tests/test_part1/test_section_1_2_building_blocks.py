# ABOUTME: Tests for Section 1.2 - Building Blocks covering qubits, gates, and operations

import pytest
import numpy as np
import cirq


class TestQubitTypes:
    """Test different qubit representations in Cirq."""

    def test_line_qubit_creation(self):
        """LineQubits represent qubits arranged in a line."""
        q0 = cirq.LineQubit(0)
        q1 = cirq.LineQubit(1)
        assert q0 != q1
        assert str(q0) == "q(0)"

    def test_line_qubit_range(self):
        """LineQubit.range creates sequential qubits."""
        qubits = cirq.LineQubit.range(3)
        assert len(qubits) == 3
        assert all(isinstance(q, cirq.LineQubit) for q in qubits)

    def test_grid_qubit_creation(self):
        """GridQubits represent qubits in a 2D grid."""
        q00 = cirq.GridQubit(0, 0)
        q01 = cirq.GridQubit(0, 1)
        assert q00 != q01
        assert str(q00) == "q(0, 0)"

    def test_grid_qubit_square(self):
        """GridQubit.square creates a square grid of qubits."""
        qubits = cirq.GridQubit.square(2)
        assert len(qubits) == 4

    def test_named_qubit_creation(self):
        """NamedQubits use arbitrary string identifiers."""
        alice = cirq.NamedQubit("alice")
        bob = cirq.NamedQubit("bob")
        assert alice != bob
        assert str(alice) == "alice"


class TestGatesAndOperations:
    """Test gate application and operation semantics."""

    def test_gate_vs_operation(self):
        """Gates are templates; operations specify target qubits."""
        q = cirq.LineQubit(0)
        h_gate = cirq.H
        h_operation = cirq.H(q)

        assert isinstance(h_gate, cirq.Gate)
        assert isinstance(h_operation, cirq.Operation)
        assert h_operation.gate == h_gate

    def test_single_qubit_gates(self):
        """Common single-qubit gates can be applied to qubits."""
        q = cirq.LineQubit(0)
        gates = [cirq.H, cirq.X, cirq.Y, cirq.Z, cirq.S, cirq.T]

        for gate in gates:
            op = gate(q)
            assert isinstance(op, cirq.Operation)
            assert op.qubits == (q,)

    def test_two_qubit_gates(self):
        """Two-qubit gates require two qubit arguments."""
        q0, q1 = cirq.LineQubit.range(2)
        gates_2q = [cirq.CNOT, cirq.CZ, cirq.SWAP]

        for gate in gates_2q:
            op = gate(q0, q1)
            assert isinstance(op, cirq.Operation)
            assert op.qubits == (q0, q1)

    def test_rotation_gates(self):
        """Rotation gates are parameterized by angle."""
        q = cirq.LineQubit(0)
        angle = np.pi / 4

        rx = cirq.rx(angle)(q)
        ry = cirq.ry(angle)(q)
        rz = cirq.rz(angle)(q)

        assert all(isinstance(op, cirq.Operation) for op in [rx, ry, rz])


class TestGateProtocols:
    """Test gate protocol implementations."""

    def test_unitary_protocol(self):
        """Gates implement the unitary protocol."""
        h_unitary = cirq.unitary(cirq.H)
        expected = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        np.testing.assert_allclose(h_unitary, expected, atol=1e-8)

    def test_gate_is_unitary(self):
        """Verify gate unitarity: U†U = I."""
        gate = cirq.H
        U = cirq.unitary(gate)
        identity = U.conj().T @ U
        np.testing.assert_allclose(identity, np.eye(2), atol=1e-8)

    def test_inverse_protocol(self):
        """Gates can be inverted."""
        q = cirq.LineQubit(0)
        op = cirq.S(q)
        inv_op = cirq.inverse(op)

        # S gate is sqrt(Z), so S†S = I (up to global phase)
        # Verify S * S^-1 = I
        s_matrix = cirq.unitary(cirq.S)
        s_inv_matrix = cirq.unitary(cirq.S ** -1)
        result = s_matrix @ s_inv_matrix
        np.testing.assert_allclose(result, np.eye(2), atol=1e-8)

    def test_pow_protocol(self):
        """Gates support exponentiation."""
        q = cirq.LineQubit(0)

        # X^2 = I
        x_squared = cirq.X ** 2
        U = cirq.unitary(x_squared)
        np.testing.assert_allclose(U, np.eye(2), atol=1e-8)

    def test_decompose_protocol(self):
        """Gates can be decomposed into simpler gates."""
        q0, q1 = cirq.LineQubit.range(2)
        swap_op = cirq.SWAP(q0, q1)

        # SWAP decomposes into CNOTs
        decomposition = cirq.decompose(swap_op)
        assert len(list(decomposition)) > 1

    def test_rotation_gates_are_unitary(self):
        """Rotation gates are unitary for arbitrary angles."""
        # Test various angles including edge cases
        test_angles = [0, np.pi / 4, np.pi / 2, np.pi, 2 * np.pi, 3.7]

        for angle in test_angles:
            # Test Rx gate: rotation about X-axis
            rx_matrix = cirq.unitary(cirq.rx(angle))
            identity_x = rx_matrix.conj().T @ rx_matrix
            np.testing.assert_allclose(identity_x, np.eye(2), atol=1e-8,
                                      err_msg=f"Rx({angle}) is not unitary")

            # Test Ry gate: rotation about Y-axis
            ry_matrix = cirq.unitary(cirq.ry(angle))
            identity_y = ry_matrix.conj().T @ ry_matrix
            np.testing.assert_allclose(identity_y, np.eye(2), atol=1e-8,
                                      err_msg=f"Ry({angle}) is not unitary")

            # Test Rz gate: rotation about Z-axis
            rz_matrix = cirq.unitary(cirq.rz(angle))
            identity_z = rz_matrix.conj().T @ rz_matrix
            np.testing.assert_allclose(identity_z, np.eye(2), atol=1e-8,
                                      err_msg=f"Rz({angle}) is not unitary")

    def test_two_qubit_gates_are_unitary(self):
        """Two-qubit gates (CNOT, CZ, SWAP) are unitary."""
        # Test CNOT gate
        cnot_matrix = cirq.unitary(cirq.CNOT)
        identity_cnot = cnot_matrix.conj().T @ cnot_matrix
        np.testing.assert_allclose(identity_cnot, np.eye(4), atol=1e-8,
                                  err_msg="CNOT is not unitary")

        # Test CZ gate
        cz_matrix = cirq.unitary(cirq.CZ)
        identity_cz = cz_matrix.conj().T @ cz_matrix
        np.testing.assert_allclose(identity_cz, np.eye(4), atol=1e-8,
                                  err_msg="CZ is not unitary")

        # Test SWAP gate
        swap_matrix = cirq.unitary(cirq.SWAP)
        identity_swap = swap_matrix.conj().T @ swap_matrix
        np.testing.assert_allclose(identity_swap, np.eye(4), atol=1e-8,
                                  err_msg="SWAP is not unitary")

    def test_swap_decomposition_preserves_unitarity(self):
        """SWAP decomposition produces equivalent unitary matrix."""
        q0, q1 = cirq.LineQubit.range(2)
        swap_op = cirq.SWAP(q0, q1)

        # Get original SWAP unitary
        original_unitary = cirq.unitary(cirq.SWAP)

        # Get decomposed circuit
        decomposed_ops = list(cirq.decompose(swap_op))

        # Build circuit from decomposed operations
        circuit = cirq.Circuit(decomposed_ops)

        # Get unitary of decomposed circuit
        decomposed_unitary = cirq.unitary(circuit)

        # Verify decomposition produces equivalent unitary (up to global phase)
        # Check if unitaries are equal up to a phase factor
        # For this, we can check if |<U1|U2>| / dim = 1
        inner_product = np.trace(original_unitary.conj().T @ decomposed_unitary)
        normalized = np.abs(inner_product) / 4  # 4 = dimension of 2-qubit system
        np.testing.assert_allclose(normalized, 1.0, atol=1e-8,
                                  err_msg="SWAP decomposition does not preserve unitarity")


class TestMeasurementOperations:
    """Test measurement operations."""

    def test_measurement_gate(self):
        """Measurements extract classical information."""
        q = cirq.LineQubit(0)
        measurement = cirq.measure(q, key='result')

        assert isinstance(measurement, cirq.Operation)
        assert measurement.gate is not None

    def test_multi_qubit_measurement(self):
        """Multiple qubits can be measured jointly."""
        qubits = cirq.LineQubit.range(3)
        measurement = cirq.measure(*qubits, key='results')

        assert len(measurement.qubits) == 3
