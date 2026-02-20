# ABOUTME: Variational Quantum Eigensolver (VQE) implementation for computing the ground state
# energy of the hydrogen molecule (H₂) across multiple bond lengths using Cirq and OpenFermion.

import cirq
import openfermion
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import sympy


def create_h2_hamiltonian(bond_length):
    """
    Create the H₂ molecular Hamiltonian at a given bond length.

    EDUCATIONAL APPROXIMATION: This function uses approximate analytical formulas
    fitted to reproduce qualitatively correct H₂ behavior without external dependencies.
    Real quantum chemistry calculations should use pyscf or other ab initio packages
    for accurate molecular integrals. This approximation is sufficient for demonstrating
    the VQE algorithm workflow.

    Uses analytical formulas for H₂ in the minimal STO-3G basis to generate
    one- and two-body integrals, then constructs the fermionic Hamiltonian
    and transforms it to a qubit operator using Jordan-Wigner transformation.

    Args:
        bond_length: H-H separation distance in Angstroms

    Returns:
        openfermion.QubitOperator: The qubit Hamiltonian for H₂ at the specified bond length
    """
    # For H₂ in minimal basis, we use analytical formulas to approximate integrals
    # These are fitted to reproduce qualitatively correct behavior
    # Real calculations would use pyscf, but this works without that dependency

    # Nuclear repulsion energy (always positive)
    nuclear_repulsion = 1.0 / bond_length

    # One-body integrals (kinetic + nuclear attraction)
    # These should be negative (electrons attracted to nuclei)
    # Approximated based on known H₂ behavior
    r = bond_length / 0.74  # Normalized to equilibrium distance
    h_core = -1.5 - 0.5 / r

    # Construct one-body integral matrix in spin orbital basis
    # 2 spatial orbitals × 2 spins = 4 spin orbitals
    # Ordering: [0α, 0β, 1α, 1β] where 0,1 are spatial orbitals and α,β are spins
    one_body_integrals = np.zeros((4, 4))
    one_body_integrals[0, 0] = h_core  # 0α
    one_body_integrals[1, 1] = h_core  # 0β
    one_body_integrals[2, 2] = h_core  # 1α
    one_body_integrals[3, 3] = h_core  # 1β

    # Two-body integrals (electron-electron repulsion)
    # In physicist notation: h[p,q,r,s] = (pq|rs)
    # Approximated Coulomb and exchange integrals
    j_integral = 0.6 / (1.0 + 0.5 * r**2)  # Coulomb
    k_integral = 0.2 * np.exp(-0.5 * (r - 1.0)**2)  # Exchange

    two_body_integrals = np.zeros((4, 4, 4, 4))

    # Coulomb integrals: electrons in same or different spatial orbitals
    # Same spatial orbital, opposite spins: (00|00) type
    two_body_integrals[0, 1, 1, 0] = j_integral  # (0α 0β|0β 0α)
    two_body_integrals[1, 0, 0, 1] = j_integral  # (0β 0α|0α 0β)
    two_body_integrals[2, 3, 3, 2] = j_integral  # (1α 1β|1β 1α)
    two_body_integrals[3, 2, 2, 3] = j_integral  # (1β 1α|1α 1β)

    # Different spatial orbitals: (01|01) type
    for i in range(2):  # spin for orbital 0
        for j in range(2):  # spin for orbital 1
            idx0, idx1 = i, 2 + j
            two_body_integrals[idx0, idx1, idx1, idx0] = j_integral * 0.8
            two_body_integrals[idx1, idx0, idx0, idx1] = j_integral * 0.8

    # Exchange integrals: same-spin electrons between different orbitals
    two_body_integrals[0, 2, 2, 0] = k_integral  # (0α 1α|1α 0α)
    two_body_integrals[2, 0, 0, 2] = k_integral  # (1α 0α|0α 1α)
    two_body_integrals[1, 3, 3, 1] = k_integral  # (0β 1β|1β 0β)
    two_body_integrals[3, 1, 1, 3] = k_integral  # (1β 0β|0β 1β)

    # Create InteractionOperator (fermionic Hamiltonian)
    fermionic_hamiltonian = openfermion.InteractionOperator(
        constant=nuclear_repulsion,
        one_body_tensor=one_body_integrals,
        two_body_tensor=two_body_integrals
    )

    # Transform to qubit operator using Jordan-Wigner transformation
    qubit_hamiltonian = openfermion.jordan_wigner(fermionic_hamiltonian)

    return qubit_hamiltonian


def prepare_hartree_fock(qubits):
    """
    Prepare the Hartree-Fock reference state for H₂.

    For H₂ with 2 electrons in the minimal basis, the HF state is |1100⟩,
    meaning the two lowest energy spin orbitals are occupied.

    Args:
        qubits: List of 4 qubits for the H₂ calculation

    Returns:
        cirq.Circuit: Circuit that prepares the Hartree-Fock state
    """
    circuit = cirq.Circuit()

    # Apply X gates to occupy the first two spin orbitals
    # This creates |1100⟩ representing two electrons in bonding orbital
    circuit.append([
        cirq.X(qubits[0]),
        cirq.X(qubits[1])
    ])

    return circuit


def build_vqe_ansatz(qubits, theta: sympy.Symbol):
    """
    Build a hardware-efficient variational ansatz for H₂ VQE.

    This ansatz uses a simple excitation-inspired circuit that can model
    electron correlation effects. It applies parameterized rotations and
    entangling gates to explore the Hilbert space near the HF reference.

    Args:
        qubits: List of 4 qubits
        theta: Symbolic variational parameter (sympy.Symbol) controlling the circuit

    Returns:
        cirq.Circuit: Parameterized ansatz circuit with symbolic parameters
    """
    q0, q1, q2, q3 = qubits

    circuit = cirq.Circuit()

    # Apply single-qubit rotations to create superposition
    circuit.append([
        cirq.ry(theta).on(q0),
        cirq.ry(theta).on(q1)
    ])

    # Entangling gates to couple qubits
    circuit.append([
        cirq.CNOT(q0, q1),
        cirq.CNOT(q1, q2),
        cirq.CNOT(q2, q3)
    ])

    # Additional parameterized rotations
    circuit.append([
        cirq.ry(-theta).on(q0),
        cirq.ry(-theta).on(q1)
    ])

    return circuit


def compute_energy(circuit, hamiltonian, simulator):
    """
    Compute the expectation value of the Hamiltonian for a given circuit.

    Evaluates ⟨ψ(θ)|H|ψ(θ)⟩ by computing expectation values of each Pauli term
    in the Hamiltonian decomposition and summing the weighted contributions.

    NOTE: This function assumes the observable (Hamiltonian) is Hermitian,
    which ensures the expectation value is real. For non-Hermitian observables,
    the result would be complex and require different handling.

    Args:
        circuit: cirq.Circuit preparing the quantum state |ψ(θ)⟩
        hamiltonian: openfermion.QubitOperator representing H (must be Hermitian)
        simulator: cirq.Simulator for computing expectation values

    Returns:
        float: The real energy expectation value
    """
    # Get the final state vector
    # Use qubit_order to ensure we always get a state vector for all 4 qubits
    n_qubits = openfermion.count_qubits(hamiltonian)
    qubits = cirq.LineQubit.range(n_qubits)

    result = simulator.simulate(circuit, qubit_order=qubits)
    state_vector = result.final_state_vector

    # Convert Hamiltonian to sparse matrix and compute expectation value
    hamiltonian_matrix = openfermion.get_sparse_operator(hamiltonian)

    # Compute ⟨ψ|H|ψ⟩
    energy = np.real(
        np.conj(state_vector) @ hamiltonian_matrix @ state_vector
    )

    return float(energy)


def run_vqe(bond_length, initial_params=None):
    """
    Execute the complete VQE algorithm for H₂ at a given bond length.

    Optimizes the variational parameters to find the ground state energy
    using classical optimization (COBYLA method) combined with quantum
    expectation value evaluation.

    Args:
        bond_length: H-H separation in Angstroms
        initial_params: Starting parameters for optimization (optional)

    Returns:
        dict: Results containing:
            - vqe_energy: Optimized ground state energy
            - exact_energy: Exact ground state from full diagonalization
            - optimal_params: Optimized variational parameters
            - optimization_result: Full scipy optimization result
    """
    # Create the molecular Hamiltonian
    hamiltonian = create_h2_hamiltonian(bond_length)

    # Set up qubits
    n_qubits = openfermion.count_qubits(hamiltonian)
    qubits = cirq.LineQubit.range(n_qubits)

    # Initialize simulator
    simulator = cirq.Simulator()

    # Define the objective function for the optimizer
    def objective(params):
        theta_val = params[0]

        # Build circuit with symbolic parameter
        theta_sym = sympy.Symbol('theta')
        circuit = prepare_hartree_fock(qubits) + build_vqe_ansatz(qubits, theta_sym)

        # Resolve parameters
        resolver = cirq.ParamResolver({theta_sym: theta_val})
        resolved_circuit = cirq.resolve_parameters(circuit, resolver)

        energy = compute_energy(resolved_circuit, hamiltonian, simulator)
        return energy

    # Set initial parameters
    if initial_params is None:
        initial_params = [0.0]

    # Run classical optimization
    result = scipy.optimize.minimize(
        objective,
        initial_params,
        method='COBYLA',
        options={'maxiter': 100}
    )

    # Compute exact energy for comparison using sparse matrix diagonalization
    hamiltonian_sparse = openfermion.get_sparse_operator(hamiltonian)
    eigenvalues, _ = np.linalg.eigh(hamiltonian_sparse.toarray())
    exact_energy = eigenvalues[0]

    return {
        'vqe_energy': result.fun,
        'exact_energy': exact_energy,
        'optimal_params': result.x,
        'optimization_result': result
    }


def compute_potential_energy_surface(bond_lengths):
    """
    Compute the H₂ potential energy surface across multiple bond lengths.

    Runs VQE optimization at each specified bond length to map out the
    molecular potential energy curve.

    Args:
        bond_lengths: Array of H-H distances in Angstroms

    Returns:
        dict: Arrays of bond_lengths, vqe_energies, and exact_energies
    """
    vqe_energies = []
    exact_energies = []

    print("Computing potential energy surface...")
    for i, length in enumerate(bond_lengths):
        print(f"  [{i+1}/{len(bond_lengths)}] Bond length: {length:.3f} Å", end='')

        result = run_vqe(length)
        vqe_energies.append(result['vqe_energy'])
        exact_energies.append(result['exact_energy'])

        error = abs(result['vqe_energy'] - result['exact_energy'])
        print(f"  →  VQE: {result['vqe_energy']:.6f} Ha  (error: {error:.6f} Ha)")

    return {
        'bond_lengths': np.array(bond_lengths),
        'vqe_energies': np.array(vqe_energies),
        'exact_energies': np.array(exact_energies)
    }


def plot_potential_energy_surface(pes_data, save_path=None):
    """
    Visualize the H₂ potential energy surface.

    Creates a publication-quality plot comparing VQE energies to exact
    results across the bond length range.

    Args:
        pes_data: Dictionary from compute_potential_energy_surface()
        save_path: Optional path to save the figure
    """
    bond_lengths = pes_data['bond_lengths']
    vqe_energies = pes_data['vqe_energies']
    exact_energies = pes_data['exact_energies']

    plt.figure(figsize=(10, 6))

    # Plot both curves
    plt.plot(bond_lengths, vqe_energies, 'o-', label='VQE', linewidth=2, markersize=6)
    plt.plot(bond_lengths, exact_energies, 'x--', label='Exact (FCI)',
             linewidth=2, markersize=8, alpha=0.7)

    # Formatting
    plt.xlabel('Bond Length (Å)', fontsize=12)
    plt.ylabel('Energy (Hartree)', fontsize=12)
    plt.title('H₂ Potential Energy Surface via VQE', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    plt.show()


def main():
    """
    Main execution function demonstrating the complete VQE workflow.

    This function:
    1. Runs VQE at a single bond length to verify functionality
    2. Computes the full potential energy surface
    3. Visualizes and analyzes the results
    """
    print("=" * 70)
    print("VQE for H₂ Molecule - Ground State Energy Calculation")
    print("=" * 70)

    # Test at equilibrium bond length
    print("\n1. Single-point calculation at equilibrium geometry:")
    print("-" * 70)
    equilibrium_length = 0.74
    result = run_vqe(equilibrium_length)

    print(f"\nBond length: {equilibrium_length} Å")
    print(f"VQE Energy:   {result['vqe_energy']:.8f} Hartree")
    print(f"Exact Energy: {result['exact_energy']:.8f} Hartree")
    print(f"Error:        {abs(result['vqe_energy'] - result['exact_energy']):.8f} Hartree")
    print(f"Optimal θ:    {result['optimal_params'][0]:.6f} radians")

    # Compute full potential energy surface
    print("\n2. Computing potential energy surface:")
    print("-" * 70)
    bond_lengths = np.linspace(0.4, 2.5, 15)
    pes_data = compute_potential_energy_surface(bond_lengths)

    # Find equilibrium from VQE
    min_idx = np.argmin(pes_data['vqe_energies'])
    min_energy = pes_data['vqe_energies'][min_idx]
    min_bond_length = pes_data['bond_lengths'][min_idx]

    print("\n3. Analysis:")
    print("-" * 70)
    print(f"Equilibrium bond length: {min_bond_length:.3f} Å")
    print(f"Ground state energy:     {min_energy:.8f} Hartree")

    # Calculate mean absolute error
    errors = np.abs(pes_data['vqe_energies'] - pes_data['exact_energies'])
    print(f"\nMean absolute error:     {np.mean(errors):.8f} Hartree")
    print(f"Max absolute error:      {np.max(errors):.8f} Hartree")

    # Visualize results
    print("\n4. Generating visualization...")
    print("-" * 70)
    plot_potential_energy_surface(pes_data)

    print("\n" + "=" * 70)
    print("VQE calculation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
