# ABOUTME: Tests for Section 2.2 - QAOA Max-Cut covering graph construction, cost functions, unitaries, circuit building, and optimization

import pytest
import numpy as np
import cirq
import networkx as nx
import sympy
from part2_applications.section_2_2_qaoa_maxcut import calculate_cut_value


class TestGraphConstruction:
    """Test graph construction and Max-Cut problem setup."""

    def test_graph_creation(self):
        """Graph can be created with weighted edges."""
        G = nx.Graph()
        G.add_weighted_edges_from([
            (0, 1, 5.0), (0, 3, 2.0), (1, 2, 3.0)
        ])

        assert G.number_of_nodes() == 4
        assert G.number_of_edges() == 3
        assert G[0][1]['weight'] == 5.0

    def test_graph_has_nodes(self):
        """Graph nodes are accessible."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 2.0)])

        nodes = list(G.nodes())
        assert 0 in nodes
        assert 1 in nodes
        assert 2 in nodes

    def test_graph_has_edges_with_weights(self):
        """Graph edges have weight attribute."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 3.5)])

        edge_data = G.get_edge_data(0, 1)
        assert 'weight' in edge_data
        assert edge_data['weight'] == 3.5


class TestMaxCutCostFunction:
    """Test Max-Cut cost function calculation."""

    def test_maxcut_hamiltonian_is_hermitian(self):
        """Max-Cut Hamiltonian H_C is Hermitian: H = H†."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1.5), (1, 2, 2.0)])

        # Construct Max-Cut Hamiltonian: H_C = sum_edges w_ij * (I - Z_i Z_j) / 2
        n_qubits = 3
        dim = 2 ** n_qubits
        hamiltonian = np.zeros((dim, dim), dtype=complex)

        # Pauli Z matrix
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.eye(2, dtype=complex)

        for u, v, data in G.edges(data=True):
            weight = data['weight']

            # Construct Z_i Z_j operator
            # For qubit i, it's I⊗...⊗I⊗Z⊗I⊗...⊗I (Z at position i)
            Z_u = [I] * n_qubits
            Z_u[u] = Z
            Z_v = [I] * n_qubits
            Z_v[v] = Z

            # Compute Z_i operator
            Z_u_op = Z_u[0]
            for mat in Z_u[1:]:
                Z_u_op = np.kron(Z_u_op, mat)

            # Compute Z_j operator
            Z_v_op = Z_v[0]
            for mat in Z_v[1:]:
                Z_v_op = np.kron(Z_v_op, mat)

            # Z_i Z_j = Z_u_op @ Z_v_op (element-wise when diagonal)
            # Actually, need to be more careful - construct full operator
            ZZ_op = np.eye(dim, dtype=complex)
            for i in range(dim):
                # Get bit values for qubits u and v
                bit_u = (i >> (n_qubits - 1 - u)) & 1
                bit_v = (i >> (n_qubits - 1 - v)) & 1
                # Z eigenvalues: |0⟩ → +1, |1⟩ → -1
                z_u_val = 1 - 2 * bit_u
                z_v_val = 1 - 2 * bit_v
                ZZ_op[i, i] = z_u_val * z_v_val

            # Add to Hamiltonian: w_ij * (I - Z_i Z_j) / 2
            identity_op = np.eye(dim, dtype=complex)
            hamiltonian += weight * (identity_op - ZZ_op) / 2

        # Verify Hermiticity: H = H†
        hermitian_conjugate = hamiltonian.conj().T
        assert np.allclose(hamiltonian, hermitian_conjugate, atol=1e-10)

        # Additionally verify diagonal (Max-Cut Hamiltonian should be diagonal)
        off_diagonal = hamiltonian - np.diag(np.diag(hamiltonian))
        assert np.allclose(off_diagonal, 0, atol=1e-10)

    def test_cost_function_simple_cut(self):
        """Cost function correctly calculates cut value."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1.0)])

        # Partition: node 0 in set A (0), node 1 in set B (1)
        partition = np.array([0, 1])

        # Calculate cut: edge (0,1) crosses partition
        cost = calculate_cut_value(partition, G)

        assert cost == 1.0

    def test_cost_function_no_cut(self):
        """Cost function returns zero when all nodes in same partition."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 5.0), (1, 2, 3.0)])

        # All nodes in same partition
        partition = np.array([0, 0, 0])

        cost = calculate_cut_value(partition, G)

        assert cost == 0.0

    def test_cost_function_maximum_cut(self):
        """Cost function identifies maximum cut."""
        G = nx.Graph()
        G.add_weighted_edges_from([
            (0, 1, 2.0), (0, 2, 3.0), (1, 2, 4.0)
        ])

        # Optimal partition for triangle: one node vs two nodes
        partition = np.array([0, 1, 1])  # Node 0 separate from nodes 1,2

        cost = calculate_cut_value(partition, G)

        # Edges (0,1) and (0,2) are cut
        assert cost == 5.0


class TestCostUnitary:
    """Test cost unitary (problem Hamiltonian) implementation."""

    def test_cost_unitary_is_unitary(self):
        """Cost unitary satisfies U†U = I (unitarity condition)."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1.5)])

        qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]
        gamma_val = 0.7  # Specific value for testing

        # Build cost unitary circuit (without Hadamards or measurements)
        circuit = cirq.Circuit()
        for u, v, data in G.edges(data=True):
            weight = data['weight']
            circuit.append([
                cirq.CNOT(qubits[u], qubits[v]),
                cirq.rz(2 * gamma_val * weight).on(qubits[v]),
                cirq.CNOT(qubits[u], qubits[v])
            ])

        # Get unitary matrix
        unitary = cirq.unitary(circuit)

        # Verify U†U = I
        identity = unitary @ unitary.conj().T
        expected_identity = np.eye(len(identity))

        assert np.allclose(identity, expected_identity, atol=1e-10)

    def test_cost_unitary_structure(self):
        """Cost unitary has correct gate structure."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1.0)])

        qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]
        gamma = sympy.Symbol('gamma')

        circuit = cirq.Circuit()
        for u, v, data in G.edges(data=True):
            weight = data['weight']
            # Cost unitary: exp(-i * gamma * weight * (I - ZZ)/2)
            # Implements as CNOT-RZ-CNOT sequence
            circuit.append([
                cirq.CNOT(qubits[u], qubits[v]),
                cirq.rz(2 * gamma * weight).on(qubits[v]),
                cirq.CNOT(qubits[u], qubits[v])
            ])

        # Check circuit has expected gates
        operations = list(circuit.all_operations())
        assert len(operations) == 3

    def test_cost_unitary_multiple_edges(self):
        """Cost unitary handles multiple edges."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 2.0)])

        qubits = [cirq.LineQubit(i) for i in range(3)]
        gamma = sympy.Symbol('gamma')

        circuit = cirq.Circuit()
        edge_count = 0
        for u, v, data in G.edges(data=True):
            weight = data['weight']
            circuit.append([
                cirq.CNOT(qubits[u], qubits[v]),
                cirq.rz(2 * gamma * weight).on(qubits[v]),
                cirq.CNOT(qubits[u], qubits[v])
            ])
            edge_count += 1

        # Should have 3 operations per edge
        assert len(list(circuit.all_operations())) == 3 * edge_count


class TestMixerUnitary:
    """Test mixer unitary (driver Hamiltonian) implementation."""

    def test_mixer_unitary_is_unitary(self):
        """Mixer unitary satisfies U†U = I (unitarity condition)."""
        qubits = [cirq.LineQubit(i) for i in range(3)]
        beta_val = 0.4  # Specific value for testing

        # Build mixer unitary circuit
        circuit = cirq.Circuit()
        circuit.append(cirq.rx(-2 * beta_val).on_each(*qubits))

        # Get unitary matrix
        unitary = cirq.unitary(circuit)

        # Verify U†U = I
        identity = unitary @ unitary.conj().T
        expected_identity = np.eye(len(identity))

        assert np.allclose(identity, expected_identity, atol=1e-10)

    def test_mixer_unitary_applies_rx_to_all_qubits(self):
        """Mixer unitary applies RX rotation to every qubit."""
        qubits = [cirq.LineQubit(i) for i in range(3)]
        beta = sympy.Symbol('beta')

        circuit = cirq.Circuit()
        circuit.append(cirq.rx(-2 * beta).on_each(*qubits))

        operations = list(circuit.all_operations())
        assert len(operations) == 3

    def test_mixer_unitary_rotation_angle(self):
        """Mixer unitary uses correct rotation angle."""
        q = cirq.LineQubit(0)
        beta = sympy.Symbol('beta')

        gate = cirq.rx(-2 * beta)
        operation = gate.on(q)

        # Gate should be parameterized
        assert isinstance(operation.gate, cirq.Rx)


class TestQAOACircuitConstruction:
    """Test QAOA circuit assembly."""

    def test_complete_qaoa_circuit_is_unitary(self):
        """Complete QAOA circuit (before measurement) is unitary."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 2.0)])

        qubits = [cirq.LineQubit(i) for i in range(3)]
        gamma_val = 0.5
        beta_val = 0.3

        # Build complete QAOA circuit WITHOUT measurements
        circuit = cirq.Circuit()

        # Initial state: uniform superposition
        circuit.append(cirq.H.on_each(*qubits))

        # Cost layer
        for u, v, data in G.edges(data=True):
            weight = data['weight']
            circuit.append([
                cirq.CNOT(qubits[u], qubits[v]),
                cirq.rz(2 * gamma_val * weight).on(qubits[v]),
                cirq.CNOT(qubits[u], qubits[v])
            ])

        # Mixer layer
        circuit.append(cirq.rx(-2 * beta_val).on_each(*qubits))

        # Get unitary matrix
        unitary = cirq.unitary(circuit)

        # Verify U†U = I
        identity = unitary @ unitary.conj().T
        expected_identity = np.eye(len(identity))

        assert np.allclose(identity, expected_identity, atol=1e-10)

    def test_qaoa_circuit_initialization(self):
        """QAOA circuit starts with uniform superposition."""
        qubits = [cirq.LineQubit(i) for i in range(3)]

        circuit = cirq.Circuit()
        circuit.append(cirq.H.on_each(*qubits))

        # Verify Hadamard on all qubits
        operations = list(circuit.all_operations())
        assert len(operations) == 3
        assert all(isinstance(op.gate, cirq.HPowGate) for op in operations)

    def test_qaoa_circuit_complete_structure(self):
        """Complete QAOA circuit has all components."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1.0)])

        qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]
        gamma = sympy.Symbol('gamma')
        beta = sympy.Symbol('beta')

        # Build complete QAOA circuit
        circuit = cirq.Circuit()

        # Initial state: uniform superposition
        circuit.append(cirq.H.on_each(*qubits))

        # Cost layer
        for u, v, data in G.edges(data=True):
            weight = data['weight']
            circuit.append([
                cirq.CNOT(qubits[u], qubits[v]),
                cirq.rz(2 * gamma * weight).on(qubits[v]),
                cirq.CNOT(qubits[u], qubits[v])
            ])

        # Mixer layer
        circuit.append(cirq.rx(-2 * beta).on_each(*qubits))

        # Should have: 2 Hadamards + 3 cost gates + 2 mixers = 7 operations
        operations = list(circuit.all_operations())
        assert len(operations) == 7

    def test_qaoa_circuit_parameterization(self):
        """QAOA circuit parameters can be resolved."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1.0)])

        qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]
        gamma = sympy.Symbol('gamma')
        beta = sympy.Symbol('beta')

        circuit = cirq.Circuit()
        circuit.append(cirq.H.on_each(*qubits))

        for u, v, data in G.edges(data=True):
            weight = data['weight']
            circuit.append([
                cirq.CNOT(qubits[u], qubits[v]),
                cirq.rz(2 * gamma * weight).on(qubits[v]),
                cirq.CNOT(qubits[u], qubits[v])
            ])

        circuit.append(cirq.rx(-2 * beta).on_each(*qubits))

        # Resolve parameters
        resolver = cirq.ParamResolver({gamma: 0.5, beta: 0.3})
        resolved_circuit = cirq.resolve_parameters(circuit, resolver)

        # Resolved circuit should have no free symbols
        assert not cirq.is_parameterized(resolved_circuit)

    def test_qaoa_circuit_with_measurements(self):
        """QAOA circuit can include measurement operations."""
        qubits = [cirq.LineQubit(i) for i in range(2)]

        circuit = cirq.Circuit()
        circuit.append(cirq.H.on_each(*qubits))
        circuit.append(cirq.measure(*qubits, key='result'))

        # Check measurement is present
        has_measurement = any(
            isinstance(op.gate, cirq.MeasurementGate)
            for op in circuit.all_operations()
        )
        assert has_measurement


class TestQAOASimulation:
    """Test QAOA circuit simulation and measurement."""

    def test_qaoa_statevector_simulation(self):
        """QAOA circuit produces correct quantum state via statevector simulation."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1.0)])

        qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]
        gamma_val = np.pi / 4
        beta_val = np.pi / 8

        # Build QAOA circuit WITHOUT measurements for statevector simulation
        circuit = cirq.Circuit()
        circuit.append(cirq.H.on_each(*qubits))
        circuit.append([
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.rz(2 * gamma_val).on(qubits[1]),
            cirq.CNOT(qubits[0], qubits[1])
        ])
        circuit.append(cirq.rx(-2 * beta_val).on_each(*qubits))

        # Simulate to get statevector
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        state_vector = result.final_state_vector

        # Verify state vector is normalized
        norm = np.sum(np.abs(state_vector) ** 2)
        assert np.isclose(norm, 1.0, atol=1e-10)

        # Verify state vector has correct dimension (2^n qubits)
        assert len(state_vector) == 4  # 2^2 for 2 qubits

        # Verify amplitudes are complex numbers
        assert all(isinstance(amp, (complex, np.complex128, np.complex64)) for amp in state_vector)

    def test_qaoa_initial_state_is_uniform_superposition(self):
        """After Hadamards, state is uniform superposition |+⟩⊗n."""
        qubits = [cirq.LineQubit(i) for i in range(2)]

        # Just Hadamards
        circuit = cirq.Circuit()
        circuit.append(cirq.H.on_each(*qubits))

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        state_vector = result.final_state_vector

        # Uniform superposition: all amplitudes equal to 1/√4 = 0.5
        expected_amplitude = 0.5
        for amp in state_vector:
            assert np.isclose(np.abs(amp), expected_amplitude, atol=1e-10)

    def test_qaoa_simulation_runs(self):
        """QAOA circuit can be simulated."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1.0)])

        qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]

        circuit = cirq.Circuit()
        circuit.append(cirq.H.on_each(*qubits))
        circuit.append([
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.rz(1.0).on(qubits[1]),
            cirq.CNOT(qubits[0], qubits[1])
        ])
        circuit.append(cirq.rx(-0.6).on_each(*qubits))
        circuit.append(cirq.measure(*qubits, key='result'))

        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=100)

        assert 'result' in result.measurements
        assert result.measurements['result'].shape == (100, 2)

    def test_qaoa_produces_bitstrings(self):
        """QAOA measurements produce valid bitstrings."""
        qubits = [cirq.LineQubit(i) for i in range(2)]

        circuit = cirq.Circuit()
        circuit.append(cirq.H.on_each(*qubits))
        circuit.append(cirq.measure(*qubits, key='result'))

        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=50)

        measurements = result.measurements['result']
        # All measurements should be 0 or 1
        assert all(bit in [0, 1] for bit in measurements.flatten())


class TestCostCalculationFromMeasurements:
    """Test cost calculation from measurement results."""

    def test_cost_from_single_bitstring(self):
        """Calculate cost for a single bitstring."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 2.0), (1, 2, 3.0)])

        # Bitstring: [0, 1, 1] (node 0 in partition A, nodes 1,2 in partition B)
        bitstring = np.array([0, 1, 1])

        cost = calculate_cut_value(bitstring, G)

        # Only edge (0,1) is cut
        assert cost == 2.0

    def test_average_cost_from_samples(self):
        """Calculate average cost from multiple samples."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1.0)])

        samples = np.array([
            [0, 1],  # Cost = 1.0
            [1, 0],  # Cost = 1.0
            [0, 0],  # Cost = 0.0
        ])

        total_cost = 0.0
        for sample in samples:
            total_cost += calculate_cut_value(sample, G)

        average_cost = total_cost / len(samples)
        expected_average = (1.0 + 1.0 + 0.0) / 3
        assert np.isclose(average_cost, expected_average)


class TestParameterOptimization:
    """Test optimization aspects of QAOA."""

    def test_cost_varies_with_parameters(self):
        """Different parameters produce different costs."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1.0)])

        qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]
        simulator = cirq.Simulator()

        def build_and_evaluate(gamma_val, beta_val):
            circuit = cirq.Circuit()
            circuit.append(cirq.H.on_each(*qubits))
            circuit.append([
                cirq.CNOT(qubits[0], qubits[1]),
                cirq.rz(2 * gamma_val).on(qubits[1]),
                cirq.CNOT(qubits[0], qubits[1])
            ])
            circuit.append(cirq.rx(-2 * beta_val).on_each(*qubits))
            circuit.append(cirq.measure(*qubits, key='result'))

            result = simulator.run(circuit, repetitions=1000)
            measurements = result.measurements['result']

            cost = 0.0
            for sample in measurements:
                cost += calculate_cut_value(sample, G)

            return cost / len(measurements)

        # Different parameters should give different results
        cost1 = build_and_evaluate(0.1, 0.1)
        cost2 = build_and_evaluate(1.0, 1.0)

        # Costs should be different (with high probability)
        # Use a small tolerance due to statistical sampling
        assert not np.isclose(cost1, cost2, atol=0.05)

    def test_initial_state_gives_expected_cost(self):
        """Initial uniform superposition gives expected baseline cost."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1.0)])

        qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]

        # No cost/mixer layers - just uniform superposition
        circuit = cirq.Circuit()
        circuit.append(cirq.H.on_each(*qubits))
        circuit.append(cirq.measure(*qubits, key='result'))

        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=5000)
        measurements = result.measurements['result']

        cost = 0.0
        for sample in measurements:
            cost += calculate_cut_value(sample, G)

        average_cost = cost / len(measurements)

        # For uniform superposition over 2 qubits: 50% of samples cut the edge
        # Expected cost = 0.5 * 1.0 = 0.5
        assert 0.4 < average_cost < 0.6


class TestSolutionQuality:
    """Test solution quality and verification."""

    def test_most_common_bitstring_extraction(self):
        """Can extract most frequently measured bitstring."""
        qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]

        # Create biased circuit favoring |11⟩
        circuit = cirq.Circuit()
        circuit.append([cirq.X(qubits[0]), cirq.X(qubits[1])])
        circuit.append(cirq.measure(*qubits, key='result'))

        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=100)

        histogram = result.histogram(key='result')
        most_common = histogram.most_common(1)[0]

        # Most common should be 3 (binary 11)
        assert most_common[0] == 3
        assert most_common[1] == 100

    def test_solution_validation(self):
        """Solution can be validated against graph."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 2.0), (1, 2, 3.0), (0, 2, 1.0)])

        # Proposed solution: [0, 1, 0]
        solution = np.array([0, 1, 0])

        # Calculate actual cut value
        cut_value = calculate_cut_value(solution, G)

        # Edges (0,1) and (1,2) are cut: 2.0 + 3.0 = 5.0
        assert cut_value == 5.0


class TestMultiLayerQAOA:
    """Test QAOA with multiple layers (p > 1)."""

    def test_two_layer_qaoa_structure(self):
        """QAOA with p=2 has correct structure."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1.0)])

        qubits = [cirq.LineQubit(0), cirq.LineQubit(1)]
        params = [sympy.Symbol(f'{name}_{i}') for i in range(2) for name in ['gamma', 'beta']]

        circuit = cirq.Circuit()
        circuit.append(cirq.H.on_each(*qubits))

        # Two layers
        for layer in range(2):
            gamma = params[layer * 2]
            beta = params[layer * 2 + 1]

            # Cost layer
            for u, v, data in G.edges(data=True):
                weight = data['weight']
                circuit.append([
                    cirq.CNOT(qubits[u], qubits[v]),
                    cirq.rz(2 * gamma * weight).on(qubits[v]),
                    cirq.CNOT(qubits[u], qubits[v])
                ])

            # Mixer layer
            circuit.append(cirq.rx(-2 * beta).on_each(*qubits))

        # Should have more operations than single layer
        operations = list(circuit.all_operations())
        # 2 Hadamards + 2 * (3 cost ops + 2 mixer ops) = 12 operations
        assert len(operations) == 12


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_node_graph(self):
        """Single node graph has no edges."""
        G = nx.Graph()
        G.add_node(0)

        assert G.number_of_nodes() == 1
        assert G.number_of_edges() == 0

    def test_disconnected_graph(self):
        """Disconnected graph components are handled."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1.0), (2, 3, 2.0)])

        # Graph has 2 components
        components = list(nx.connected_components(G))
        assert len(components) == 2

    def test_empty_measurements_handling(self):
        """Handle case with no measurements."""
        qubits = [cirq.LineQubit(0)]
        circuit = cirq.Circuit(cirq.H(qubits[0]))

        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)

        # Should get state vector with empty measurements dict
        assert hasattr(result, 'final_state_vector')
        assert hasattr(result, 'measurements')
        assert len(result.measurements) == 0
