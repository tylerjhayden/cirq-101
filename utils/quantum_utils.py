# ABOUTME: Core quantum computing utility functions for circuit preparation, simulation,
# measurement analysis, and visualization - eliminates code duplication across the project.

import cirq
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Tuple, List
from collections import Counter


def save_and_show_plot(
    save_path: str,
    dpi: int = 150,
    bbox_inches: str = 'tight',
    show: bool = True,
    print_message: bool = True
) -> None:
    """
    Save and optionally display the current matplotlib figure.

    Standardizes figure output across the project with consistent quality settings.
    This function handles both saving to disk and displaying to screen, with optional
    user feedback via print statements.

    Args:
        save_path: File path where the figure should be saved (e.g., 'notebooks/plot.png')
        dpi: Resolution in dots per inch for the saved figure (default: 150)
        bbox_inches: Bounding box setting for tight layout (default: 'tight')
        show: Whether to display the figure after saving (default: True)
        print_message: Whether to print confirmation message (default: True)

    Example:
        >>> plt.plot([1, 2, 3], [1, 4, 9])
        >>> plt.xlabel('x')
        >>> plt.ylabel('x²')
        >>> save_and_show_plot('notebooks/parabola.png', dpi=300)
    """
    plt.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)

    if print_message:
        print(f"  Saved to: {save_path}")

    if show:
        plt.show()


def prepare_bell_state(
    q0: cirq.Qid,
    q1: cirq.Qid,
    variant: str = 'phi_plus'
) -> cirq.Circuit:
    """
    Prepare one of the four maximally entangled Bell states.

    Bell states are the fundamental building blocks of quantum entanglement and
    quantum information protocols. This function creates any of the four Bell basis
    states using Hadamard and CNOT gates, with optional Pauli corrections.

    The four Bell states are:
        - |Φ+⟩ = (|00⟩ + |11⟩)/√2  [phi_plus]
        - |Φ-⟩ = (|00⟩ - |11⟩)/√2  [phi_minus]
        - |Ψ+⟩ = (|01⟩ + |10⟩)/√2  [psi_plus]
        - |Ψ-⟩ = (|01⟩ - |10⟩)/√2  [psi_minus]

    Args:
        q0: First qubit (control qubit for the CNOT gate)
        q1: Second qubit (target qubit for the CNOT gate)
        variant: Which Bell state to prepare. Options:
            - 'phi_plus' (default): |Φ+⟩ = (|00⟩ + |11⟩)/√2
            - 'phi_minus': |Φ-⟩ = (|00⟩ - |11⟩)/√2
            - 'psi_plus': |Ψ+⟩ = (|01⟩ + |10⟩)/√2
            - 'psi_minus': |Ψ-⟩ = (|01⟩ - |10⟩)/√2

    Returns:
        cirq.Circuit: Circuit that prepares the requested Bell state from |00⟩

    Raises:
        ValueError: If variant is not one of the four valid Bell states

    Example:
        >>> q0, q1 = cirq.LineQubit.range(2)
        >>> circuit = prepare_bell_state(q0, q1, variant='phi_plus')
        >>> print(circuit)
        0: ───H───@───
                  │
        1: ───────X───
    """
    valid_variants = ['phi_plus', 'phi_minus', 'psi_plus', 'psi_minus']
    if variant not in valid_variants:
        raise ValueError(
            f"Invalid variant '{variant}'. Must be one of {valid_variants}"
        )

    circuit = cirq.Circuit()

    # All Bell states start with H on q0 and CNOT
    circuit.append([
        cirq.H(q0),
        cirq.CNOT(q0, q1)
    ])

    # Apply corrections for specific variants
    if variant == 'phi_minus':
        # Apply Z to introduce phase flip: (|00⟩ + |11⟩)/√2 → (|00⟩ - |11⟩)/√2
        circuit.append(cirq.Z(q1))
    elif variant == 'psi_plus':
        # Apply X to flip second qubit: (|00⟩ + |11⟩)/√2 → (|01⟩ + |10⟩)/√2
        circuit.append(cirq.X(q1))
    elif variant == 'psi_minus':
        # Create (|01⟩ - |10⟩)/√2
        # After CNOT we have (|00⟩ + |11⟩)/√2
        # Apply X to q1: (|01⟩ + |10⟩)/√2 (this is |Ψ+⟩)
        # Apply Z to q0 to flip sign of |10⟩: (|01⟩ - |10⟩)/√2 (this is |Ψ-⟩)
        circuit.append([cirq.X(q1), cirq.Z(q0)])

    return circuit


def simulate_and_sample(
    circuit: cirq.Circuit,
    repetitions: int = 1000,
    measurement_key: str = 'result',
    simulator: Optional[cirq.Simulator] = None,
    return_histogram: bool = True,
    return_measurements: bool = False
) -> Dict[str, Any]:
    """
    Execute a quantum circuit and collect measurement statistics.

    This function runs a quantum circuit multiple times (shots) and analyzes the
    measurement outcomes, providing both histogram statistics and optional raw
    measurement data. It automatically handles simulator creation and provides
    a unified interface for circuit sampling.

    Args:
        circuit: Quantum circuit to execute (must include measurements)
        repetitions: Number of times to execute the circuit (shots) (default: 1000)
        measurement_key: Key used in the circuit's measurement operations (default: 'result')
        simulator: Optional Cirq simulator instance. If None, creates cirq.Simulator()
        return_histogram: Whether to include histogram in results (default: True)
        return_measurements: Whether to include raw measurements in results (default: False)

    Returns:
        Dictionary containing:
            - 'histogram': Counter mapping bitstrings to counts (if return_histogram=True)
            - 'measurements': Raw measurement array (if return_measurements=True)
            - 'repetitions': Number of shots executed
            - 'most_common': Most frequently measured outcome (bitstring, count)

    Raises:
        ValueError: If circuit does not contain measurements with the specified key

    Example:
        >>> q0, q1 = cirq.LineQubit.range(2)
        >>> circuit = cirq.Circuit(
        ...     cirq.H(q0),
        ...     cirq.CNOT(q0, q1),
        ...     cirq.measure(q0, q1, key='result')
        ... )
        >>> results = simulate_and_sample(circuit, repetitions=1000)
        >>> print(f"Most common: {results['most_common']}")
        Most common: (0, 502)  # |00⟩ appeared 502 times
    """
    # Create simulator if not provided
    if simulator is None:
        simulator = cirq.Simulator()

    # Run the circuit
    result = simulator.run(circuit, repetitions=repetitions)

    # Verify measurement key exists
    if measurement_key not in result.measurements:
        available_keys = list(result.measurements.keys())
        raise ValueError(
            f"Measurement key '{measurement_key}' not found in circuit. "
            f"Available keys: {available_keys}"
        )

    # Build results dictionary
    results = {
        'repetitions': repetitions
    }

    # Add histogram if requested
    if return_histogram:
        histogram = result.histogram(key=measurement_key)
        results['histogram'] = histogram
        # Most common outcome
        if histogram:
            most_common = histogram.most_common(1)[0]
            results['most_common'] = most_common
        else:
            results['most_common'] = None

    # Add raw measurements if requested
    if return_measurements:
        results['measurements'] = result.measurements[measurement_key]

    return results


def compute_pauli_expectation(
    circuit: cirq.Circuit,
    observable: cirq.PauliString,
    qubits: Optional[List[cirq.Qid]] = None,
    simulator: Optional[cirq.Simulator] = None
) -> float:
    """
    Compute the expectation value of a Pauli observable for a quantum state.

    Calculates ⟨ψ|O|ψ⟩ where |ψ⟩ is the state prepared by the circuit and O is
    a Pauli observable (product of X, Y, Z operators). This is fundamental for
    variational quantum algorithms like VQE and QAOA.

    The expectation value represents the average measurement outcome of the
    observable, which must be Hermitian for physical observables. Pauli operators
    have eigenvalues ±1, so expectation values lie in the range [-1, +1].

    Args:
        circuit: Quantum circuit preparing the state |ψ⟩ (should not include measurements)
        observable: Pauli observable as a cirq.PauliString or product of Pauli operators
        qubits: Optional list of qubits defining the qubit ordering for the state vector.
                If None, extracted from the circuit.
        simulator: Optional Cirq simulator instance. If None, creates cirq.Simulator()

    Returns:
        Expectation value ⟨ψ|O|ψ⟩ as a float (always real for Hermitian observables)

    Example:
        >>> # Compute ⟨Z⟩ for |+⟩ state (should be 0)
        >>> q = cirq.LineQubit(0)
        >>> circuit = cirq.Circuit(cirq.H(q))
        >>> observable = cirq.Z(q)
        >>> exp_val = compute_pauli_expectation(circuit, observable)
        >>> print(f"⟨+|Z|+⟩ = {exp_val:.6f}")
        ⟨+|Z|+⟩ = 0.000000

        >>> # Compute ⟨ZZ⟩ for Bell state (should be +1)
        >>> q0, q1 = cirq.LineQubit.range(2)
        >>> circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
        >>> observable = cirq.Z(q0) * cirq.Z(q1)
        >>> exp_val = compute_pauli_expectation(circuit, observable)
        >>> print(f"⟨Φ+|ZZ|Φ+⟩ = {exp_val:.6f}")
        ⟨Φ+|ZZ|Φ+⟩ = 1.000000
    """
    # Create simulator if not provided
    if simulator is None:
        simulator = cirq.Simulator()

    # Get qubits from circuit if not specified
    if qubits is None:
        qubits = sorted(circuit.all_qubits())

    # Simulate to get state vector
    result = simulator.simulate(circuit, qubit_order=qubits)
    state_vector = result.final_state_vector

    # Create qubit map for expectation calculation
    qubit_map = {q: i for i, q in enumerate(qubits)}

    # Compute expectation value using Cirq's built-in simulator method
    # This handles all types of observables correctly (single, multi-qubit, PauliString, etc.)
    expectation_values = simulator.simulate_expectation_values(
        circuit,
        observables=[observable],
        qubit_order=qubits
    )

    # Return as float (take real part since Hermitian observables have real expectation)
    return float(np.real(expectation_values[0]))
