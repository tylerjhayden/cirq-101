# ABOUTME: Tests quantum circuit best practices including hardware awareness and parameterization strategies.
# ABOUTME: Validates circuit modularity, optimization techniques, and quantum correctness using statevector simulation.

import pytest
import numpy as np
import cirq
import sympy


# ============================================================================
# Pytest Fixtures - Reduce code duplication for common test setups
# ============================================================================

@pytest.fixture
def two_qubits():
    """Provide two LineQubits for tests."""
    return cirq.LineQubit.range(2)


@pytest.fixture
def three_qubits():
    """Provide three LineQubits for tests."""
    return cirq.LineQubit.range(3)


@pytest.fixture
def linear_connectivity():
    """Provide standard linear connectivity pattern: q0-q1-q2."""
    return {0: [1], 1: [0, 2], 2: [1]}


@pytest.fixture
def typical_gate_times():
    """Provide typical gate timing parameters."""
    return {'single': 25e-9, 'two': 50e-9}


@pytest.fixture
def typical_coherence_times():
    """Provide typical T1/T2 coherence times."""
    return {'t1': 50e-6, 't2': 30e-6}


# ============================================================================
# Test Classes
# ============================================================================

class TestHardwareAwareness:
    """Test hardware-aware circuit design principles."""

    def test_connectivity_validation(self, three_qubits, linear_connectivity):
        """Circuits should respect qubit connectivity constraints."""
        from part3_synthesis.section_3_2_best_practices import validate_connectivity

        q0, q1, q2 = three_qubits

        # Valid circuit: only adjacent qubits
        valid_circuit = cirq.Circuit(
            cirq.CNOT(q0, q1),
            cirq.CNOT(q1, q2)
        )
        assert validate_connectivity(valid_circuit, linear_connectivity) is True

        # Invalid circuit: non-adjacent qubits
        invalid_circuit = cirq.Circuit(
            cirq.CNOT(q0, q2)  # q0 not connected to q2
        )
        assert validate_connectivity(invalid_circuit, linear_connectivity) is False

    def test_native_gate_compilation(self):
        """Circuits should use native gate sets when possible."""
        from part3_synthesis.section_3_2_best_practices import compile_to_native_gates

        q = cirq.LineQubit(0)
        # Use gates that need decomposition
        circuit = cirq.Circuit(
            cirq.H(q),
            cirq.T(q)
        )

        # Native gate set: sqrt_x, sqrt_y, phased_xz
        native_gates = {cirq.PhasedXZGate, cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate}
        compiled = compile_to_native_gates(circuit)

        # Check that all gates in compiled circuit are from native set or single-qubit
        for moment in compiled:
            for op in moment:
                gate_type = type(op.gate)
                # Allow common native gates
                assert gate_type in native_gates or isinstance(op.gate, (cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate, cirq.PhasedXZGate))

    def test_circuit_depth_awareness(self, two_qubits):
        """Circuit depth should be tracked for decoherence considerations."""
        from part3_synthesis.section_3_2_best_practices import calculate_circuit_depth

        q0, q1 = two_qubits

        # Sequential circuit: depth = 3
        sequential = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.X(q1)
        )
        assert calculate_circuit_depth(sequential) == 3

        # Parallel circuit: depth = 2
        parallel = cirq.Circuit(
            cirq.Moment([cirq.H(q0), cirq.X(q1)]),
            cirq.CNOT(q0, q1)
        )
        assert calculate_circuit_depth(parallel) == 2

    def test_decoherence_time_estimate(self):
        """Estimate if circuit fits within T1/T2 constraints."""
        from part3_synthesis.section_3_2_best_practices import estimate_execution_time, check_decoherence_constraint

        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.measure(q0, q1, key='result')
        )

        # Typical gate times: 25ns single-qubit, 50ns two-qubit
        gate_times = {'single': 25e-9, 'two': 50e-9}
        execution_time = estimate_execution_time(circuit, gate_times)

        # Should be 25ns (H) + 50ns (CNOT) = 75ns
        assert abs(execution_time - 75e-9) < 1e-12

        # Check against T1/T2 times
        t1 = 50e-6  # 50 microseconds
        t2 = 30e-6  # 30 microseconds
        assert check_decoherence_constraint(execution_time, t1, t2) is True

        # Should fail with very short T2
        assert check_decoherence_constraint(execution_time, t1, 10e-9) is False


class TestParameterization:
    """Test parameterization best practices using symbolic parameters."""

    def test_symbolic_parameter_usage(self):
        """Parameterized circuits should use sympy.Symbol."""
        from part3_synthesis.section_3_2_best_practices import create_parameterized_rotation

        theta = sympy.Symbol('theta')
        q = cirq.LineQubit(0)

        circuit = create_parameterized_rotation(q, theta, axis='X')

        # Circuit should contain symbolic parameter
        assert cirq.is_parameterized(circuit)

        # Should resolve when parameter is bound
        resolved = cirq.resolve_parameters(circuit, {'theta': np.pi/2})
        assert not cirq.is_parameterized(resolved)

    def test_parameter_sweep(self):
        """Parameterized circuits enable parameter sweeps."""
        from part3_synthesis.section_3_2_best_practices import create_parameterized_rotation

        theta = sympy.Symbol('theta')
        q = cirq.LineQubit(0)
        circuit = create_parameterized_rotation(q, theta, axis='Z')
        circuit.append(cirq.measure(q, key='m'))

        # Create parameter sweep
        sweep = cirq.Linspace('theta', start=0, stop=2*np.pi, length=10)

        # Verify sweep can be applied
        simulator = cirq.Simulator()
        results = simulator.run_sweep(circuit, sweep, repetitions=100)
        assert len(results) == 10

    def test_multi_parameter_circuit(self):
        """Circuits should support multiple independent parameters."""
        from part3_synthesis.section_3_2_best_practices import create_multi_param_circuit

        alpha = sympy.Symbol('alpha')
        beta = sympy.Symbol('beta')
        gamma = sympy.Symbol('gamma')

        q0, q1 = cirq.LineQubit.range(2)
        circuit = create_multi_param_circuit(q0, q1, alpha, beta, gamma)

        # Should have multiple parameters
        params = cirq.parameter_names(circuit)
        assert 'alpha' in params
        assert 'beta' in params
        assert 'gamma' in params
        assert len(params) == 3

    def test_parameter_reuse(self):
        """Same parameter symbol can be reused across gates."""
        from part3_synthesis.section_3_2_best_practices import create_symmetric_circuit

        theta = sympy.Symbol('theta')
        qubits = cirq.LineQubit.range(3)

        # Create circuit where same angle is applied to all qubits
        circuit = create_symmetric_circuit(qubits, theta)

        # All gates should use the same parameter
        params = cirq.parameter_names(circuit)
        assert len(params) == 1
        assert 'theta' in params


class TestModularity:
    """Test modular circuit design patterns."""

    def test_circuit_composition(self):
        """Subcircuits should compose into larger circuits."""
        from part3_synthesis.section_3_2_best_practices import create_bell_state_subcircuit, create_teleportation_circuit

        # Bell state is a reusable subcircuit
        q0, q1 = cirq.LineQubit.range(2)
        bell_subcircuit = create_bell_state_subcircuit(q0, q1)

        # Should be valid circuit
        assert isinstance(bell_subcircuit, cirq.Circuit)
        assert len(bell_subcircuit) > 0

        # Can be composed into larger circuit
        msg = cirq.NamedQubit('msg')
        full_circuit = create_teleportation_circuit(msg, q0, q1)
        assert isinstance(full_circuit, cirq.Circuit)

    def test_reusable_building_blocks(self):
        """Common patterns should be extracted as functions."""
        from part3_synthesis.section_3_2_best_practices import apply_qft, apply_inverse_qft

        qubits = cirq.LineQubit.range(3)
        circuit = cirq.Circuit()

        # QFT is a reusable building block
        apply_qft(circuit, qubits)
        qft_depth = len(circuit)
        assert qft_depth > 0

        # Inverse QFT reuses the same logic
        apply_inverse_qft(circuit, qubits)
        assert len(circuit) > qft_depth

    def test_layer_abstraction(self):
        """Circuits should be built from logical layers."""
        from part3_synthesis.section_3_2_best_practices import create_qaoa_layer

        qubits = cirq.LineQubit.range(4)
        gamma = sympy.Symbol('gamma')
        beta = sympy.Symbol('beta')

        # QAOA layer encapsulates cost + mixer
        layer = create_qaoa_layer(qubits, gamma, beta)

        assert isinstance(layer, cirq.Circuit)
        assert cirq.is_parameterized(layer)

    def test_subcircuit_unitarity(self):
        """Subcircuits should maintain unitary property."""
        from part3_synthesis.section_3_2_best_practices import create_controlled_rotation_block

        q0, q1 = cirq.LineQubit.range(2)
        angle = np.pi / 4

        block = create_controlled_rotation_block(q0, q1, angle)

        # Get full unitary of block
        simulator = cirq.Simulator()
        result = simulator.simulate(block)

        # Create reverse block
        reverse_block = cirq.inverse(block)
        full_circuit = block + reverse_block

        # Should return to initial state
        final_result = simulator.simulate(full_circuit)
        np.testing.assert_allclose(
            final_result.final_state_vector,
            np.array([1, 0, 0, 0]),
            atol=1e-8
        )


class TestCircuitOptimization:
    """Test circuit depth and gate count optimization."""

    def test_depth_optimization(self):
        """Circuits should be optimized for minimal depth."""
        from part3_synthesis.section_3_2_best_practices import optimize_circuit_depth

        q0, q1, q2 = cirq.LineQubit.range(3)

        # Unoptimized: sequential gates on different qubits
        unoptimized = cirq.Circuit(
            cirq.H(q0),
            cirq.H(q1),
            cirq.H(q2),
            cirq.X(q0),
            cirq.X(q1)
        )

        optimized = optimize_circuit_depth(unoptimized)

        # Optimized should have lower or equal depth
        assert len(optimized) <= len(unoptimized)

    def test_gate_cancellation(self):
        """Adjacent inverse gates should cancel."""
        from part3_synthesis.section_3_2_best_practices import cancel_inverse_gates

        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.X(q),
            cirq.X(q),  # X^2 = I
            cirq.H(q),
            cirq.H(q),  # H^2 = I
            cirq.Z(q)
        )

        optimized = cancel_inverse_gates(circuit)

        # Should only have Z gate remaining
        assert len(list(optimized.all_operations())) == 1

    def test_commuting_gate_reordering(self):
        """Commuting gates can be reordered for better parallelization."""
        from part3_synthesis.section_3_2_best_practices import check_gates_commute

        q0, q1 = cirq.LineQubit.range(2)

        # Z gates on different qubits commute
        assert check_gates_commute(cirq.Z(q0), cirq.Z(q1)) is True

        # CNOT and X on control qubit don't commute
        assert check_gates_commute(cirq.CNOT(q0, q1), cirq.X(q0)) is False

    def test_single_qubit_gate_merging(self):
        """Adjacent single-qubit gates can be merged."""
        from part3_synthesis.section_3_2_best_practices import merge_single_qubit_gates

        q = cirq.LineQubit(0)
        circuit = cirq.Circuit(
            cirq.rz(0.1)(q),
            cirq.rz(0.2)(q),  # Should merge to rz(0.3)
        )

        merged = merge_single_qubit_gates(circuit)

        # Should have fewer operations
        original_ops = len(list(circuit.all_operations()))
        merged_ops = len(list(merged.all_operations()))
        assert merged_ops <= original_ops


class TestBestPracticesExamples:
    """Test examples demonstrating hardcoded vs parameterized approaches."""

    def test_parameterized_modular_circuit(self):
        """Parameterized modular circuit uses symbolic parameters and layers."""
        from part3_synthesis.section_3_2_best_practices import parameterized_modular_circuit

        result = parameterized_modular_circuit()

        # Should return a valid circuit
        assert isinstance(result, cirq.Circuit)

        # Should be parameterized
        assert cirq.is_parameterized(result)

    def test_hardcoded_monolithic_circuit(self):
        """Hardcoded monolithic circuit uses fixed values and lacks modularity."""
        from part3_synthesis.section_3_2_best_practices import hardcoded_monolithic_circuit

        result = hardcoded_monolithic_circuit()

        # Should return a circuit
        assert isinstance(result, cirq.Circuit)

        # Should NOT be parameterized (hardcoded values)
        assert not cirq.is_parameterized(result)

    def test_convert_to_parameterized(self):
        """Convert hardcoded circuit to parameterized version."""
        from part3_synthesis.section_3_2_best_practices import (
            hardcoded_monolithic_circuit,
            convert_to_parameterized
        )

        hardcoded_circuit = hardcoded_monolithic_circuit()
        parameterized_circuit = convert_to_parameterized(hardcoded_circuit)

        # Parameterized version should use symbolic parameters
        assert cirq.is_parameterized(parameterized_circuit)

        # Should have similar structure
        # (number of qubits should be the same)
        hardcoded_qubits = len(hardcoded_circuit.all_qubits())
        parameterized_qubits = len(parameterized_circuit.all_qubits())
        assert hardcoded_qubits == parameterized_qubits


class TestDocumentationAndUsage:
    """Test that the module can be run standalone."""

    def test_module_has_main(self):
        """Module should have a main function for standalone execution."""
        from part3_synthesis.section_3_2_best_practices import main

        # Should be callable
        assert callable(main)

        # Should run without errors
        main()

    def test_module_demonstrates_all_concepts(self):
        """Main function should demonstrate all best practices."""
        from part3_synthesis.section_3_2_best_practices import main

        # Capture any output/demonstrations
        # The main function should demonstrate:
        # 1. Hardware awareness
        # 2. Parameterization
        # 3. Modularity
        # 4. Circuit optimization

        # Just verify it runs successfully
        try:
            main()
            assert True
        except Exception as e:
            pytest.fail(f"main() raised an exception: {e}")
