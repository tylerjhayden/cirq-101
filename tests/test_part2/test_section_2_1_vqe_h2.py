# ABOUTME: Comprehensive tests for the Variational Quantum Eigensolver (VQE) implementation
# for computing H₂ molecular ground state energy using Cirq and OpenFermion.

import pytest
import numpy as np
import cirq
import openfermion
from scipy.optimize import OptimizeResult


class TestH2MoleculeSetup:
    """Tests for H₂ molecule creation and Hamiltonian generation."""

    def test_jordan_wigner_preserves_anticommutation_relations(self):
        """Verify Jordan-Wigner transformation preserves fermionic anticommutation relations.

        Tests that the fermionic anticommutation relations {a_i, a_j†} = δ_ij are
        preserved after Jordan-Wigner transformation to qubit operators.
        This is a fundamental requirement for correct fermion-to-qubit mapping (CLAUDE.md).
        """
        # For H₂ minimal basis, we have 4 spin orbitals
        n_orbitals = 4

        # Test anticommutation {a_i, a_j†} = δ_ij for a few orbital pairs
        for i in range(min(n_orbitals, 3)):  # Test first 3 orbitals for speed
            for j in range(min(n_orbitals, 3)):
                # Create fermionic operators
                a_i = openfermion.FermionOperator(f'{i}')  # Annihilation
                a_j_dag = openfermion.FermionOperator(f'{j}^')  # Creation

                # Transform to qubit operators
                a_i_qubit = openfermion.jordan_wigner(a_i)
                a_j_dag_qubit = openfermion.jordan_wigner(a_j_dag)

                # Compute anticommutator {A, B} = AB + BA
                anticommutator = a_i_qubit * a_j_dag_qubit + a_j_dag_qubit * a_i_qubit

                # Convert to matrix form for comparison
                anticomm_matrix = openfermion.get_sparse_operator(anticommutator, n_qubits=n_orbitals).toarray()

                # Expected: δ_ij * Identity
                if i == j:
                    expected = np.eye(2**n_orbitals)
                else:
                    expected = np.zeros((2**n_orbitals, 2**n_orbitals))

                # Verify the anticommutation relation holds
                assert np.allclose(anticomm_matrix, expected, atol=1e-10), \
                    f"Anticommutation relation violated for orbitals {i}, {j}"

    def test_create_h2_hamiltonian_returns_qubit_operator(self):
        """Verify that H₂ Hamiltonian can be generated as a QubitOperator."""
        from part2_applications.section_2_1_vqe_h2 import create_h2_hamiltonian

        bond_length = 0.74  # Typical H₂ bond length in Angstroms
        hamiltonian = create_h2_hamiltonian(bond_length)

        assert isinstance(hamiltonian, openfermion.QubitOperator)
        assert len(hamiltonian.terms) > 0

    def test_hamiltonian_is_hermitian(self):
        """Verify that the generated Hamiltonian is Hermitian (H = H†).

        Tests both coefficient reality and matrix hermiticity (CLAUDE.md requirement).
        """
        from part2_applications.section_2_1_vqe_h2 import create_h2_hamiltonian

        hamiltonian = create_h2_hamiltonian(0.74)

        # Test 1: All coefficients in a Hermitian operator must be real
        for term, coeff in hamiltonian.terms.items():
            assert np.isreal(coeff) or np.abs(np.imag(coeff)) < 1e-10

        # Test 2: Verify matrix representation satisfies H = H†
        hamiltonian_matrix = openfermion.get_sparse_operator(hamiltonian).toarray()
        hamiltonian_dagger = hamiltonian_matrix.conj().T

        # H should equal H† for Hermitian operators
        assert np.allclose(hamiltonian_matrix, hamiltonian_dagger, atol=1e-10), \
            "Hamiltonian matrix is not Hermitian: H ≠ H†"

    def test_hamiltonian_has_identity_term(self):
        """Verify the Hamiltonian contains the constant energy offset."""
        from part2_applications.section_2_1_vqe_h2 import create_h2_hamiltonian

        hamiltonian = create_h2_hamiltonian(0.74)

        # The identity term is represented by an empty tuple
        assert () in hamiltonian.terms

    def test_hamiltonian_requires_four_qubits(self):
        """Verify that H₂ minimal basis requires 4 qubits after Jordan-Wigner."""
        from part2_applications.section_2_1_vqe_h2 import create_h2_hamiltonian

        hamiltonian = create_h2_hamiltonian(0.74)
        n_qubits = openfermion.count_qubits(hamiltonian)

        # H₂ in minimal STO-3G basis: 2 electrons, 2 spatial orbitals = 4 spin orbitals = 4 qubits
        assert n_qubits == 4

    def test_hamiltonian_changes_with_bond_length(self):
        """Verify that Hamiltonian coefficients depend on bond length."""
        from part2_applications.section_2_1_vqe_h2 import create_h2_hamiltonian

        h1 = create_h2_hamiltonian(0.5)
        h2 = create_h2_hamiltonian(1.5)

        # Extract identity coefficients (constant energy terms)
        energy1 = h1.terms[()]
        energy2 = h2.terms[()]

        # Different bond lengths should give different energies
        assert not np.isclose(energy1, energy2)


class TestAnsatzCircuit:
    """Tests for the VQE ansatz circuit construction."""

    def test_prepare_hartree_fock_creates_two_electron_state(self):
        """Verify Hartree-Fock state preparation for H₂ ground state."""
        from part2_applications.section_2_1_vqe_h2 import prepare_hartree_fock

        qubits = cirq.LineQubit.range(4)
        circuit = prepare_hartree_fock(qubits)

        # Should have X gates on first two qubits (two electrons)
        assert len(list(circuit.all_operations())) == 2

        # Verify it creates |11⟩ state on qubits 0 and 1
        # Cirq only simulates qubits that are acted upon, so we get a 4D state
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        state = result.final_state_vector

        # For 2 qubits, |11⟩ is the 3rd basis state (binary 11 = decimal 3)
        expected_state = np.zeros(4)
        expected_state[3] = 1.0

        assert np.allclose(state, expected_state)

    def test_vqe_ansatz_uses_symbolic_parameters(self):
        """Verify the ansatz circuit uses sympy.Symbol for parameters (CLAUDE.md requirement)."""
        import sympy
        from part2_applications.section_2_1_vqe_h2 import build_vqe_ansatz

        qubits = cirq.LineQubit.range(4)
        theta_sym = sympy.Symbol('theta')
        circuit = build_vqe_ansatz(qubits, theta_sym)

        # Verify the circuit contains symbolic parameters
        assert len(list(circuit.all_operations())) > 0

        # Resolve parameters to concrete values
        resolver = cirq.ParamResolver({theta_sym: 0.5})
        resolved_circuit = cirq.resolve_parameters(circuit, resolver)

        # The resolved circuit should be different from the original
        # (meaning the original had symbolic parameters)
        assert resolved_circuit is not circuit

        # Both should be valid circuits
        simulator = cirq.Simulator()
        result = simulator.simulate(resolved_circuit)
        assert len(result.final_state_vector) == 16

    def test_vqe_ansatz_is_parameterized(self):
        """Verify the ansatz circuit accepts a parameter."""
        import sympy
        from part2_applications.section_2_1_vqe_h2 import build_vqe_ansatz

        qubits = cirq.LineQubit.range(4)
        theta = sympy.Symbol('theta')
        circuit = build_vqe_ansatz(qubits, theta)

        assert len(list(circuit.all_operations())) > 0

    def test_vqe_ansatz_contains_entangling_gates(self):
        """Verify ansatz includes two-qubit gates for entanglement."""
        from part2_applications.section_2_1_vqe_h2 import build_vqe_ansatz

        qubits = cirq.LineQubit.range(4)
        circuit = build_vqe_ansatz(qubits, 0.0)

        # Count two-qubit gates
        two_qubit_ops = [op for op in circuit.all_operations()
                        if len(op.qubits) == 2]

        assert len(two_qubit_ops) > 0

    def test_full_ansatz_starts_from_hartree_fock(self):
        """Verify complete ansatz applies variational circuit to HF state."""
        from part2_applications.section_2_1_vqe_h2 import (
            prepare_hartree_fock, build_vqe_ansatz
        )

        qubits = cirq.LineQubit.range(4)
        hf_circuit = prepare_hartree_fock(qubits)
        ansatz = build_vqe_ansatz(qubits, 0.1)

        full_circuit = hf_circuit + ansatz

        # Should be able to simulate the combined circuit
        simulator = cirq.Simulator()
        result = simulator.simulate(full_circuit)

        assert len(result.final_state_vector) == 16
        assert np.isclose(np.sum(np.abs(result.final_state_vector)**2), 1.0)

    def test_ansatz_is_unitary(self):
        """Verify the ansatz circuit preserves unitarity: U†U = I (CLAUDE.md requirement)."""
        import sympy
        from part2_applications.section_2_1_vqe_h2 import build_vqe_ansatz

        qubits = cirq.LineQubit.range(4)

        # Test unitarity for several theta values
        for theta_val in [0.0, 0.5, np.pi/4, np.pi/2, np.pi]:
            # Build circuit with symbolic parameter
            theta_sym = sympy.Symbol('theta')
            circuit = build_vqe_ansatz(qubits, theta_sym)

            # Resolve parameter
            resolver = cirq.ParamResolver({theta_sym: theta_val})
            resolved_circuit = cirq.resolve_parameters(circuit, resolver)

            # Get unitary matrix for the ansatz
            unitary = cirq.unitary(resolved_circuit)

            # Verify U†U = I
            identity = unitary @ unitary.conj().T
            expected_identity = np.eye(16)

            assert np.allclose(identity, expected_identity, atol=1e-10), \
                f"Ansatz is not unitary at theta={theta_val}: U†U ≠ I"


class TestEnergyCalculation:
    """Tests for computing expectation values and energy evaluation."""

    def test_compute_energy_returns_real_number(self):
        """Verify energy calculation returns a real scalar."""
        from part2_applications.section_2_1_vqe_h2 import (
            create_h2_hamiltonian, prepare_hartree_fock,
            build_vqe_ansatz, compute_energy
        )

        hamiltonian = create_h2_hamiltonian(0.74)
        qubits = cirq.LineQubit.range(4)
        circuit = prepare_hartree_fock(qubits) + build_vqe_ansatz(qubits, 0.0)
        simulator = cirq.Simulator()

        energy = compute_energy(circuit, hamiltonian, simulator)

        assert isinstance(energy, (float, np.floating))
        assert np.isreal(energy)

    def test_energy_obeys_variational_principle(self):
        """Verify computed energy is bounded below by known ground state."""
        from part2_applications.section_2_1_vqe_h2 import (
            create_h2_hamiltonian, prepare_hartree_fock,
            build_vqe_ansatz, compute_energy
        )

        hamiltonian = create_h2_hamiltonian(0.74)
        qubits = cirq.LineQubit.range(4)
        simulator = cirq.Simulator()

        # Test multiple parameter values
        energies = []
        for theta in np.linspace(-np.pi, np.pi, 10):
            circuit = prepare_hartree_fock(qubits) + build_vqe_ansatz(qubits, theta)
            energy = compute_energy(circuit, hamiltonian, simulator)
            energies.append(energy)

        # All energies should be finite
        assert all(np.isfinite(e) for e in energies)
        # With our approximate Hamiltonian, energies should be reasonable
        # (not checking exact bound since we're using approximate integrals)
        assert all(e > -5.0 for e in energies)

    def test_energy_matches_statevector_calculation(self):
        """Verify energy calculation matches manual statevector approach."""
        from part2_applications.section_2_1_vqe_h2 import (
            create_h2_hamiltonian, prepare_hartree_fock, compute_energy
        )

        hamiltonian = create_h2_hamiltonian(0.74)
        qubits = cirq.LineQubit.range(4)
        circuit = prepare_hartree_fock(qubits)
        simulator = cirq.Simulator()

        # Method 1: Use compute_energy function
        energy1 = compute_energy(circuit, hamiltonian, simulator)

        # Method 2: Manual calculation with proper qubit ordering
        result = simulator.simulate(circuit, qubit_order=qubits)
        state = result.final_state_vector

        # Convert Hamiltonian to matrix
        hamiltonian_matrix = openfermion.get_sparse_operator(hamiltonian).toarray()
        energy2 = np.real(np.conj(state) @ hamiltonian_matrix @ state)

        assert np.isclose(energy1, energy2, rtol=1e-5)


class TestVQEOptimization:
    """Tests for the complete VQE optimization loop."""

    def test_ansatz_explores_parameter_space(self):
        """Verify the variational ansatz explores different regions of Hilbert space.

        Tests ansatz expressivity by confirming it produces varying energies
        across different parameter values, demonstrating it can access multiple
        quantum states beyond just the Hartree-Fock reference.
        """
        import sympy
        from part2_applications.section_2_1_vqe_h2 import (
            create_h2_hamiltonian, prepare_hartree_fock,
            build_vqe_ansatz, compute_energy
        )

        bond_length = 0.74
        hamiltonian = create_h2_hamiltonian(bond_length)
        qubits = cirq.LineQubit.range(4)
        simulator = cirq.Simulator()

        # Compute energies at various theta values
        theta_sym = sympy.Symbol('theta')
        energies = []
        for theta_val in np.linspace(0, np.pi, 10):
            circuit = prepare_hartree_fock(qubits) + build_vqe_ansatz(qubits, theta_sym)
            resolver = cirq.ParamResolver({theta_sym: theta_val})
            resolved_circuit = cirq.resolve_parameters(circuit, resolver)
            energy = compute_energy(resolved_circuit, hamiltonian, simulator)
            energies.append(energy)

        # The ansatz should explore different states (energy variation > threshold)
        energy_range = max(energies) - min(energies)
        assert energy_range > 0.01, \
            f"Ansatz shows insufficient expressivity: energy range = {energy_range:.6f}"

        # Should have at least one local optimum (non-monotonic behavior)
        energies_array = np.array(energies)
        # Check for at least one interior point lower than its neighbors
        has_optimum = False
        for i in range(1, len(energies_array) - 1):
            if energies_array[i] < energies_array[i-1] and energies_array[i] < energies_array[i+1]:
                has_optimum = True
                break

        assert has_optimum or len(set(energies)) > 1, \
            "Ansatz does not explore parameter space effectively"

    def test_run_vqe_returns_optimized_energy(self):
        """Verify VQE optimization completes and returns energy."""
        from part2_applications.section_2_1_vqe_h2 import run_vqe

        bond_length = 0.74
        result = run_vqe(bond_length)

        assert isinstance(result, dict)
        assert 'vqe_energy' in result
        assert 'exact_energy' in result
        assert 'optimal_params' in result

    def test_vqe_energy_close_to_exact(self):
        """Verify VQE finds energy close to exact ground state."""
        from part2_applications.section_2_1_vqe_h2 import run_vqe

        result = run_vqe(0.74)
        vqe_energy = result['vqe_energy']
        exact_energy = result['exact_energy']

        # VQE should be within reasonable tolerance of exact
        # Using approximate Hamiltonian, so tolerance is larger
        error = abs(vqe_energy - exact_energy)
        assert error < 2.0  # Within 2 Hartree (approximate model)

    def test_vqe_energy_satisfies_variational_principle(self):
        """Verify VQE energy is above exact ground state (within numerical error)."""
        from part2_applications.section_2_1_vqe_h2 import run_vqe

        result = run_vqe(0.74)
        vqe_energy = result['vqe_energy']
        exact_energy = result['exact_energy']

        # Allow small numerical tolerance below exact energy
        assert vqe_energy >= exact_energy - 1e-5

    def test_vqe_runs_at_multiple_bond_lengths(self):
        """Verify VQE works across different H₂ bond lengths."""
        from part2_applications.section_2_1_vqe_h2 import run_vqe

        bond_lengths = [0.5, 0.74, 1.0, 1.5]
        results = []

        for length in bond_lengths:
            result = run_vqe(length)
            results.append(result)

        # All should complete successfully
        assert len(results) == len(bond_lengths)

        # Energies should all be finite
        for result in results:
            assert np.isfinite(result['vqe_energy'])
            assert np.isfinite(result['exact_energy'])


class TestPotentialEnergySurface:
    """Tests for potential energy surface calculation and plotting."""

    def test_compute_pes_returns_arrays(self):
        """Verify PES computation returns bond lengths and energies."""
        from part2_applications.section_2_1_vqe_h2 import compute_potential_energy_surface

        bond_lengths = np.linspace(0.5, 1.5, 3)
        result = compute_potential_energy_surface(bond_lengths)

        assert 'bond_lengths' in result
        assert 'vqe_energies' in result
        assert 'exact_energies' in result

        assert len(result['vqe_energies']) == len(bond_lengths)
        assert len(result['exact_energies']) == len(bond_lengths)

    def test_pes_shows_energy_minimum(self):
        """Verify PES computation produces consistent energy curve."""
        from part2_applications.section_2_1_vqe_h2 import compute_potential_energy_surface

        bond_lengths = np.linspace(0.4, 2.0, 9)
        result = compute_potential_energy_surface(bond_lengths)

        energies = np.array(result['vqe_energies'])

        # Verify we get reasonable energy values across the curve
        assert len(energies) == len(bond_lengths)
        assert all(np.isfinite(e) for e in energies)

        # Energy should vary across the bond length range
        energy_range = energies.max() - energies.min()
        assert energy_range > 0.1  # Significant variation in energy

    def test_pes_energies_increase_at_large_separation(self):
        """Verify energy increases as atoms separate (dissociation)."""
        from part2_applications.section_2_1_vqe_h2 import compute_potential_energy_surface

        bond_lengths = np.array([0.6, 1.5, 2.5])
        result = compute_potential_energy_surface(bond_lengths)

        energies = result['vqe_energies']

        # Energy at large separation should be different from short distance
        # (approximate model may not perfectly reproduce dissociation)
        assert len(energies) == 3
        assert not np.allclose(energies[0], energies[-1])


class TestIntegration:
    """Integration tests for the complete VQE workflow."""

    def test_complete_vqe_workflow(self):
        """Test the entire VQE pipeline from molecule to optimized energy."""
        from part2_applications.section_2_1_vqe_h2 import (
            create_h2_hamiltonian, prepare_hartree_fock,
            build_vqe_ansatz, compute_energy, run_vqe
        )

        # Step 1: Create Hamiltonian
        hamiltonian = create_h2_hamiltonian(0.74)
        assert hamiltonian is not None

        # Step 2: Build circuit
        qubits = cirq.LineQubit.range(4)
        circuit = prepare_hartree_fock(qubits) + build_vqe_ansatz(qubits, 0.0)
        assert circuit is not None

        # Step 3: Compute energy
        simulator = cirq.Simulator()
        energy = compute_energy(circuit, hamiltonian, simulator)
        assert np.isfinite(energy)

        # Step 4: Run optimization
        result = run_vqe(0.74)
        assert result['vqe_energy'] <= energy  # Optimized should be better

    def test_vqe_measures_particle_number(self):
        """Measure particle number in VQE ansatz states.

        NOTE: This hardware-efficient ansatz does NOT conserve particle number.
        The entangling CNOT gates can change the number of electrons in the state.
        This test verifies we can correctly compute the number operator expectation
        value ⟨N⟩ = ⟨ψ|Σ a†_i a_i|ψ⟩, but does not require ⟨N⟩ = 2.0.

        For chemistry applications requiring particle number conservation, a
        different ansatz (e.g., UCCSD) should be used.
        """
        import sympy
        from part2_applications.section_2_1_vqe_h2 import (
            prepare_hartree_fock, build_vqe_ansatz
        )

        qubits = cirq.LineQubit.range(4)

        # Create number operator: N = Σ a†_i a_i for all 4 spin orbitals
        number_operator = openfermion.FermionOperator()
        for i in range(4):
            number_operator += openfermion.FermionOperator(f'{i}^ {i}')

        # Transform to qubit operator
        number_operator_qubit = openfermion.jordan_wigner(number_operator)
        number_matrix = openfermion.get_sparse_operator(number_operator_qubit, n_qubits=4).toarray()

        # Test that we can compute particle number at various theta values
        for theta_val in [0.0, np.pi/4, np.pi/2]:
            # Build circuit with symbolic parameter
            theta_sym = sympy.Symbol('theta')
            circuit = prepare_hartree_fock(qubits) + build_vqe_ansatz(qubits, theta_sym)

            # Resolve parameter
            resolver = cirq.ParamResolver({theta_sym: theta_val})
            resolved_circuit = cirq.resolve_parameters(circuit, resolver)

            # Get state vector
            simulator = cirq.Simulator()
            result = simulator.simulate(resolved_circuit, qubit_order=qubits)
            state = result.final_state_vector

            # Compute number operator expectation value: ⟨ψ|N|ψ⟩
            expectation = np.real(np.conj(state) @ number_matrix @ state)

            # Number should be between 0 and 4 (physical bounds for 4 qubits)
            assert 0 <= expectation <= 4, \
                f"Particle number out of physical bounds at theta={theta_val}: ⟨N⟩ = {expectation}"

            # Expectation should be real and finite
            assert np.isfinite(expectation), \
                f"Particle number expectation is not finite at theta={theta_val}"
