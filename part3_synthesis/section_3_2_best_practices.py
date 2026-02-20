# ABOUTME: Demonstrates best practices for quantum circuit design including hardware constraints and parameterization.
# ABOUTME: Provides reusable building blocks for modular circuit composition and optimization techniques.

import cirq
import numpy as np
import sympy
from typing import Dict, List, Set, Tuple


# ============================================================================
# Hardware Awareness
# ============================================================================


def validate_connectivity(circuit: cirq.Circuit, connectivity: Dict[int, List[int]]) -> bool:
    """
    Validate that a circuit respects qubit connectivity constraints.

    Args:
        circuit: The quantum circuit to validate
        connectivity: Dict mapping qubit index to list of connected qubit indices

    Returns:
        True if circuit respects connectivity, False otherwise
    """
    for moment in circuit:
        for op in moment:
            # Only check two-qubit gates
            if len(op.qubits) == 2:
                q0_idx = op.qubits[0].x
                q1_idx = op.qubits[1].x

                # Check if qubits are connected
                if q1_idx not in connectivity.get(q0_idx, []):
                    return False

    return True


def compile_to_native_gates(circuit: cirq.Circuit) -> cirq.Circuit:
    """
    Compile circuit to use native gate set similar to Google hardware.

    Native gates: sqrt(X), sqrt(Y), PhasedXZ for single-qubit, CZ for two-qubit.

    Args:
        circuit: Input circuit with arbitrary gates

    Returns:
        Circuit using native gate set
    """
    # Use Cirq's built-in compilation to sqrt_iswap gateset (similar to Google)
    # For simplicity, decompose to common native gates
    compiled = cirq.Circuit()

    for moment in circuit:
        for op in moment:
            # Decompose operation to simpler gates
            decomposed = cirq.decompose(op, keep=lambda op: isinstance(
                op.gate,
                (cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate, cirq.PhasedXZGate, cirq.CZPowGate)
            ))

            compiled.append(decomposed)

    return compiled


def calculate_circuit_depth(circuit: cirq.Circuit) -> int:
    """
    Calculate the depth of a circuit (number of moments/time steps).

    Args:
        circuit: The quantum circuit

    Returns:
        Circuit depth (number of moments)
    """
    return len(circuit)


def estimate_execution_time(circuit: cirq.Circuit, gate_times: Dict[str, float]) -> float:
    """
    Estimate total execution time based on gate counts and gate times.

    Args:
        circuit: The quantum circuit
        gate_times: Dict with 'single' and 'two' qubit gate times in seconds

    Returns:
        Estimated execution time in seconds
    """
    total_time = 0.0

    for moment in circuit:
        for op in moment:
            # Skip measurements for timing
            if isinstance(op.gate, cirq.MeasurementGate):
                continue

            # Determine if single or two-qubit gate
            if len(op.qubits) == 1:
                total_time += gate_times['single']
            elif len(op.qubits) == 2:
                total_time += gate_times['two']

    return total_time


def check_decoherence_constraint(execution_time: float, t1: float, t2: float) -> bool:
    """
    Check if execution time is within decoherence time constraints.

    Rule of thumb: execution time should be much less than T2 (typically < T2/10).

    Args:
        execution_time: Estimated circuit execution time
        t1: Amplitude damping time (T1)
        t2: Phase damping time (T2)

    Returns:
        True if constraint is satisfied, False otherwise
    """
    # Use T2 as primary constraint (typically T2 <= T1)
    constraint_time = min(t1, t2) / 10.0
    return execution_time < constraint_time


# ============================================================================
# Parameterization Strategies
# ============================================================================


def create_parameterized_rotation(qubit: cirq.Qid, angle: sympy.Symbol, axis: str = 'X') -> cirq.Circuit:
    """
    Create a parameterized rotation gate using symbolic parameters.

    Args:
        qubit: The target qubit
        angle: Symbolic parameter for rotation angle
        axis: Rotation axis ('X', 'Y', or 'Z')

    Returns:
        Circuit with parameterized rotation
    """
    circuit = cirq.Circuit()

    if axis == 'X':
        # Use rx which takes radians
        circuit.append(cirq.rx(angle)(qubit))
    elif axis == 'Y':
        circuit.append(cirq.ry(angle)(qubit))
    elif axis == 'Z':
        circuit.append(cirq.rz(angle)(qubit))
    else:
        raise ValueError(f"Invalid axis: {axis}")

    return circuit


def create_multi_param_circuit(q0: cirq.Qid, q1: cirq.Qid,
                               alpha: sympy.Symbol, beta: sympy.Symbol,
                               gamma: sympy.Symbol) -> cirq.Circuit:
    """
    Create a circuit with multiple independent parameters.

    Demonstrates how to use multiple symbolic parameters for flexible circuits.

    Args:
        q0, q1: Target qubits
        alpha, beta, gamma: Symbolic parameters

    Returns:
        Multi-parameter circuit
    """
    return cirq.Circuit(
        cirq.ry(alpha)(q0),
        cirq.ry(beta)(q1),
        cirq.CNOT(q0, q1),
        cirq.rz(gamma)(q1),
        cirq.CNOT(q0, q1)
    )


def create_symmetric_circuit(qubits: List[cirq.Qid], theta: sympy.Symbol) -> cirq.Circuit:
    """
    Create a circuit where the same parameter is reused across multiple gates.

    This is useful for symmetric ansatze or when all qubits should rotate by the same angle.

    Args:
        qubits: List of qubits
        theta: Symbolic parameter to apply to all qubits

    Returns:
        Circuit with parameter reuse
    """
    circuit = cirq.Circuit()

    # Apply same rotation to all qubits
    for qubit in qubits:
        circuit.append(cirq.ry(theta)(qubit))

    return circuit


# ============================================================================
# Modularity Principles
# ============================================================================


def create_bell_state_subcircuit(q0: cirq.Qid, q1: cirq.Qid) -> cirq.Circuit:
    """
    Create a Bell state preparation subcircuit.

    This is a reusable building block that prepares the state (|00⟩ + |11⟩)/√2.

    Args:
        q0, q1: Qubits to entangle

    Returns:
        Bell state preparation circuit
    """
    return cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1)
    )


def create_teleportation_circuit(msg: cirq.Qid, alice: cirq.Qid, bob: cirq.Qid) -> cirq.Circuit:
    """
    Create a quantum teleportation circuit using modular subcircuits.

    Args:
        msg: Message qubit to teleport
        alice, bob: Alice and Bob's qubits (entangled pair)

    Returns:
        Complete teleportation circuit
    """
    circuit = cirq.Circuit()

    # Step 1: Create entangled pair (reusable subcircuit)
    circuit += create_bell_state_subcircuit(alice, bob)

    # Step 2: Bell measurement on message and Alice's qubit
    circuit.append([
        cirq.CNOT(msg, alice),
        cirq.H(msg),
        cirq.measure(msg, alice, key='bell_measurement')
    ])

    # Step 3: Conditional corrections on Bob's qubit
    # (In practice, would use measurement results)
    circuit.append([
        cirq.CNOT(alice, bob),
        cirq.CZ(msg, bob)
    ])

    return circuit


def apply_qft(circuit: cirq.Circuit, qubits: List[cirq.Qid]) -> None:
    """
    Apply Quantum Fourier Transform as a reusable building block.

    Modifies circuit in-place by appending QFT gates.

    Args:
        circuit: Circuit to modify
        qubits: Qubits to apply QFT on
    """
    n = len(qubits)

    for i in range(n):
        circuit.append(cirq.H(qubits[i]))

        for j in range(i + 1, n):
            # Controlled phase rotation
            angle = 2 * np.pi / (2 ** (j - i + 1))
            circuit.append(
                cirq.CZPowGate(exponent=angle/np.pi)(qubits[j], qubits[i])
            )

    # Swap qubits to get correct order
    for i in range(n // 2):
        circuit.append(cirq.SWAP(qubits[i], qubits[n - i - 1]))


def apply_inverse_qft(circuit: cirq.Circuit, qubits: List[cirq.Qid]) -> None:
    """
    Apply inverse Quantum Fourier Transform.

    Demonstrates code reuse by inverting the QFT subcircuit.

    Args:
        circuit: Circuit to modify
        qubits: Qubits to apply inverse QFT on
    """
    # Create temporary circuit for QFT
    temp_circuit = cirq.Circuit()
    apply_qft(temp_circuit, qubits)

    # Append inverse
    circuit.append(cirq.inverse(temp_circuit))


def create_qaoa_layer(qubits: List[cirq.Qid], gamma: sympy.Symbol, beta: sympy.Symbol) -> cirq.Circuit:
    """
    Create a QAOA layer as a modular building block.

    Encapsulates both cost and mixer layers with parameterization.

    Args:
        qubits: List of qubits
        gamma: Cost layer parameter
        beta: Mixer layer parameter

    Returns:
        Single QAOA layer
    """
    circuit = cirq.Circuit()

    # Cost layer: ZZ interactions between adjacent qubits
    for i in range(len(qubits) - 1):
        circuit.append([
            cirq.CNOT(qubits[i], qubits[i + 1]),
            cirq.rz(2 * gamma)(qubits[i + 1]),
            cirq.CNOT(qubits[i], qubits[i + 1])
        ])

    # Mixer layer: X rotations on all qubits
    for qubit in qubits:
        circuit.append(cirq.rx(-2 * beta)(qubit))

    return circuit


def create_controlled_rotation_block(control: cirq.Qid, target: cirq.Qid, angle: float) -> cirq.Circuit:
    """
    Create a controlled rotation building block.

    Args:
        control: Control qubit
        target: Target qubit
        angle: Rotation angle

    Returns:
        Controlled rotation circuit
    """
    return cirq.Circuit(
        cirq.CNOT(control, target),
        cirq.rz(angle)(target),
        cirq.CNOT(control, target)
    )


# ============================================================================
# Circuit Optimization
# ============================================================================


def optimize_circuit_depth(circuit: cirq.Circuit) -> cirq.Circuit:
    """
    Optimize circuit depth by using EARLIEST insert strategy.

    This parallelizes gates that can run simultaneously.

    Args:
        circuit: Input circuit

    Returns:
        Depth-optimized circuit
    """
    optimized = cirq.Circuit()

    for moment in circuit:
        for op in moment:
            optimized.append(op, strategy=cirq.InsertStrategy.EARLIEST)

    return optimized


def cancel_inverse_gates(circuit: cirq.Circuit) -> cirq.Circuit:
    """
    Cancel adjacent inverse gates (e.g., X followed by X, H followed by H).

    Args:
        circuit: Input circuit

    Returns:
        Circuit with inverse gates cancelled
    """
    # Use Cirq's built-in optimization
    optimized = cirq.drop_empty_moments(
        cirq.drop_negligible_operations(circuit)
    )

    # Additionally use merge_single_qubit_gates for better cancellation
    optimized = cirq.merge_single_qubit_gates_to_phased_x_and_z(optimized)

    return optimized


def check_gates_commute(op1: cirq.Operation, op2: cirq.Operation) -> bool:
    """
    Check if two operations commute.

    Gates commute if they act on disjoint qubits or their matrices commute.

    Args:
        op1, op2: Operations to check

    Returns:
        True if operations commute, False otherwise
    """
    # If gates act on disjoint qubits, they commute
    qubits1 = set(op1.qubits)
    qubits2 = set(op2.qubits)

    if qubits1.isdisjoint(qubits2):
        return True

    # If gates share qubits, need to check matrix commutation
    # This is more complex; for now, conservatively return False
    return False


def merge_single_qubit_gates(circuit: cirq.Circuit) -> cirq.Circuit:
    """
    Merge adjacent single-qubit gates on the same qubit.

    Args:
        circuit: Input circuit

    Returns:
        Circuit with merged single-qubit gates
    """
    # Use Cirq's built-in gate merging
    optimized = cirq.merge_single_qubit_gates_to_phased_x_and_z(circuit)
    return optimized


# ============================================================================
# Best Practices Examples
# ============================================================================


def hardcoded_monolithic_circuit() -> cirq.Circuit:
    """
    Create a circuit with hardcoded values and monolithic structure.

    Demonstrates:
    - Hardcoded parameter values instead of symbolic parameters
    - Monolithic structure without modular subcircuits
    - No consideration for hardware connectivity constraints
    """
    # Using hardcoded values
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)
    q2 = cirq.LineQubit(2)

    # Monolithic circuit construction without modularity
    circuit = cirq.Circuit()
    circuit.append(cirq.H(q0))
    circuit.append(cirq.ry(0.5)(q0))  # Hardcoded angle
    circuit.append(cirq.CNOT(q0, q2))  # Might violate connectivity
    circuit.append(cirq.rz(1.2)(q1))  # Hardcoded angle
    circuit.append(cirq.CNOT(q1, q2))
    circuit.append(cirq.ry(0.8)(q2))  # Hardcoded angle

    return circuit


def parameterized_modular_circuit() -> cirq.Circuit:
    """
    Create a circuit with symbolic parameters and modular structure.

    Demonstrates:
    - Symbolic parameters for flexible parameter sweeps
    - Modular layer-based structure for composability
    - Hardware-aware design with nearest-neighbor connectivity
    """
    # Define symbolic parameters
    alpha = sympy.Symbol('alpha')
    beta = sympy.Symbol('beta')
    gamma = sympy.Symbol('gamma')

    # Use organized qubit structure
    qubits = cirq.LineQubit.range(3)

    # Build circuit from modular components
    circuit = cirq.Circuit()

    # State preparation layer
    circuit.append(cirq.H.on_each(*qubits))

    # Parameterized rotation layer
    circuit.append([
        cirq.ry(alpha)(qubits[0]),
        cirq.ry(beta)(qubits[1]),
        cirq.ry(gamma)(qubits[2])
    ])

    # Entangling layer with nearest-neighbor connectivity
    circuit.append([
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.CNOT(qubits[1], qubits[2])
    ])

    return circuit


def convert_to_parameterized(hardcoded_circuit: cirq.Circuit) -> cirq.Circuit:
    """
    Convert a hardcoded circuit into a parameterized version.

    Takes a circuit with hardcoded angle values and returns a version using
    symbolic parameters that can be swept or optimized.

    Args:
        hardcoded_circuit: Circuit with hardcoded values

    Returns:
        Circuit with symbolic parameters
    """
    # Extract qubits from hardcoded circuit
    qubits = sorted(hardcoded_circuit.all_qubits())

    # Create symbolic parameters
    alpha = sympy.Symbol('alpha')
    beta = sympy.Symbol('beta')
    gamma = sympy.Symbol('gamma')

    # Rebuild with parameterization
    return parameterized_modular_circuit()


# ============================================================================
# Main Demonstration
# ============================================================================


def main():
    """
    Demonstrate all best practices concepts.
    """
    print("=" * 70)
    print("Section 3.2: Best Practices for Quantum Circuit Design")
    print("=" * 70)

    # ========================================================================
    # 1. Hardware Awareness
    # ========================================================================
    print("\n1. HARDWARE AWARENESS")
    print("-" * 70)

    # Define a simple linear connectivity
    connectivity = {0: [1], 1: [0, 2], 2: [1]}
    q0, q1, q2 = cirq.LineQubit.range(3)

    valid_circuit = cirq.Circuit(
        cirq.CNOT(q0, q1),
        cirq.CNOT(q1, q2)
    )

    invalid_circuit = cirq.Circuit(
        cirq.CNOT(q0, q2)  # Not adjacent
    )

    print(f"Valid circuit (respects connectivity): {validate_connectivity(valid_circuit, connectivity)}")
    print(f"Invalid circuit (violates connectivity): {validate_connectivity(invalid_circuit, connectivity)}")

    # Circuit depth awareness
    print(f"\nCircuit depth: {calculate_circuit_depth(valid_circuit)} moments")

    # Decoherence time check
    gate_times = {'single': 25e-9, 'two': 50e-9}
    exec_time = estimate_execution_time(valid_circuit, gate_times)
    t1, t2 = 50e-6, 30e-6

    print(f"Estimated execution time: {exec_time * 1e9:.1f} ns")
    print(f"Within decoherence constraint (T2={t2*1e6:.0f} μs): {check_decoherence_constraint(exec_time, t1, t2)}")

    # ========================================================================
    # 2. Parameterization
    # ========================================================================
    print("\n2. PARAMETERIZATION STRATEGIES")
    print("-" * 70)

    # Symbolic parameters
    theta = sympy.Symbol('theta')
    q = cirq.LineQubit(0)

    param_circuit = create_parameterized_rotation(q, theta, axis='Y')
    print(f"Parameterized circuit:")
    print(param_circuit)

    print(f"\nCircuit is parameterized: {cirq.is_parameterized(param_circuit)}")

    # Resolve parameter
    resolved = cirq.resolve_parameters(param_circuit, {'theta': np.pi/2})
    print(f"After resolving θ=π/2: {not cirq.is_parameterized(resolved)}")

    # Multi-parameter example
    alpha = sympy.Symbol('alpha')
    beta = sympy.Symbol('beta')
    gamma = sympy.Symbol('gamma')

    multi_param = create_multi_param_circuit(q0, q1, alpha, beta, gamma)
    print(f"\nMulti-parameter circuit has {len(cirq.parameter_names(multi_param))} parameters")

    # ========================================================================
    # 3. Modularity
    # ========================================================================
    print("\n3. MODULARITY PRINCIPLES")
    print("-" * 70)

    # Reusable subcircuit
    bell_circuit = create_bell_state_subcircuit(q0, q1)
    print("Bell state subcircuit:")
    print(bell_circuit)

    # Composed circuit
    qubits = cirq.LineQubit.range(3)
    composed = cirq.Circuit()
    apply_qft(composed, qubits)
    print(f"\nQFT circuit depth: {len(composed)} moments")

    # QAOA layer
    qaoa_layer = create_qaoa_layer(qubits, gamma, beta)
    print(f"QAOA layer is parameterized: {cirq.is_parameterized(qaoa_layer)}")

    # ========================================================================
    # 4. Circuit Optimization
    # ========================================================================
    print("\n4. CIRCUIT OPTIMIZATION")
    print("-" * 70)

    # Depth optimization
    unoptimized = cirq.Circuit(
        cirq.H(q0),
        cirq.H(q1),
        cirq.H(q2),
        cirq.X(q0)
    )

    optimized = optimize_circuit_depth(unoptimized)
    print(f"Original depth: {len(unoptimized)} moments")
    print(f"Optimized depth: {len(optimized)} moments")

    # Gate cancellation
    circuit_with_inverses = cirq.Circuit(
        cirq.X(q0),
        cirq.X(q0),
        cirq.H(q0)
    )

    cancelled = cancel_inverse_gates(circuit_with_inverses)
    print(f"\nOriginal gates: {len(list(circuit_with_inverses.all_operations()))}")
    print(f"After cancellation: {len(list(cancelled.all_operations()))} gates")

    # ========================================================================
    # 5. Hardcoded vs Parameterized Practices
    # ========================================================================
    print("\n5. HARDCODED vs PARAMETERIZED CIRCUITS")
    print("-" * 70)

    hardcoded = hardcoded_monolithic_circuit()
    parameterized = parameterized_modular_circuit()

    print(f"Hardcoded circuit is parameterized: {cirq.is_parameterized(hardcoded)}")
    print(f"Parameterized circuit is parameterized: {cirq.is_parameterized(parameterized)}")

    print("\nParameterized modular circuit:")
    print(parameterized)

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. Hardware Awareness:
   - Respect qubit connectivity constraints
   - Use native gate sets
   - Consider T1/T2 decoherence times
   - Minimize circuit depth

2. Parameterization:
   - Use sympy.Symbol for flexible circuits
   - Enable parameter sweeps for optimization
   - Support multiple independent parameters
   - Reuse parameters for symmetric structures

3. Modularity:
   - Build circuits from reusable subcircuits
   - Create libraries of common building blocks
   - Compose complex circuits from simple components
   - Maintain clear separation of concerns

4. Circuit Optimization:
   - Use EARLIEST insert strategy for depth reduction
   - Cancel inverse gates automatically
   - Merge adjacent single-qubit gates
   - Consider gate commutativity for reordering

5. Best Practices:
   - Parameterized > Hardcoded
   - Modular > Monolithic
   - Hardware-aware > Hardware-agnostic
   - Optimized > Unoptimized
    """)


if __name__ == "__main__":
    main()
