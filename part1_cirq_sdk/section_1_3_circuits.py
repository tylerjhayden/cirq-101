# ABOUTME: Section 1.3 - Circuits: moments, circuit construction, insert strategies, and circuit manipulation in Cirq

import cirq
import numpy as np
import matplotlib.pyplot as plt


def _print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(title)
    print("="*60)


def demonstrate_moments():
    """Demonstrate moment creation and properties."""
    _print_section_header("MOMENTS - PARALLEL QUANTUM OPERATIONS")

    print("\nMoments represent operations executed during the same time slice.")
    print("All operations in a moment must act on disjoint qubits.")

    # Create qubits
    q0, q1, q2 = cirq.LineQubit.range(3)

    # Valid moment - disjoint qubits
    print("\n1. Valid Moment - operations on disjoint qubits:")
    moment1 = cirq.Moment([cirq.H(q0), cirq.X(q1), cirq.Y(q2)])
    print(f"   Moment: {moment1}")
    print(f"   Number of operations: {len(moment1)}")
    print(f"   Operations execute simultaneously (in parallel)")

    # Moment with two-qubit gate
    print("\n2. Moment with two-qubit gate:")
    moment2 = cirq.Moment([cirq.CNOT(q0, q1), cirq.H(q2)])
    print(f"   Moment: {moment2}")
    print(f"   CNOT uses q0 and q1, H uses q2 - all disjoint")

    # Iterating over moment
    print("\n3. Iterating over operations in a moment:")
    for i, op in enumerate(moment1):
        print(f"   Operation {i}: {op}")

    print("\n4. Attempting to create invalid moment (same qubit twice):")
    print("   This would raise ValueError:")
    print(f"   cirq.Moment([cirq.H(q0), cirq.X(q0)]) -> ValueError")


def demonstrate_circuit_construction():
    """Demonstrate circuit creation methods."""
    _print_section_header("CIRCUIT CONSTRUCTION")

    print("\nCircuits are ordered sequences of moments.")

    # Empty circuit
    print("\n1. Empty circuit:")
    empty_circuit = cirq.Circuit()
    print(f"   Circuit: {empty_circuit}")
    print(f"   Length (moments): {len(empty_circuit)}")

    # Circuit from operations
    print("\n2. Circuit from operations:")
    q0, q1 = cirq.LineQubit.range(2)
    circuit_ops = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
    print(circuit_ops)

    # Circuit from moments
    print("\n3. Circuit from moments:")
    moment1 = cirq.Moment([cirq.H(q0)])
    moment2 = cirq.Moment([cirq.CNOT(q0, q1)])
    circuit_moments = cirq.Circuit(moment1, moment2)
    print(circuit_moments)

    # Appending operations
    print("\n4. Building circuit by appending operations:")
    circuit_append = cirq.Circuit()
    circuit_append.append(cirq.H(q0))
    circuit_append.append(cirq.H(q1))
    circuit_append.append(cirq.CNOT(q0, q1))
    print(circuit_append)


def build_circuit_with_strategy(q0, q1, q2, strategy):
    """Build a circuit using a specific insert strategy.

    Applies the same sequence of operations (H(q0), H(q1), X(q0), H(q2))
    using the given insertion strategy.
    """
    circuit = cirq.Circuit()
    circuit.append(cirq.H(q0), strategy=strategy)
    circuit.append(cirq.H(q1), strategy=strategy)
    circuit.append(cirq.X(q0), strategy=strategy)
    circuit.append(cirq.H(q2), strategy=strategy)
    return circuit


def demonstrate_insert_strategies():
    """Demonstrate different circuit insertion strategies."""
    _print_section_header("INSERT STRATEGIES")

    print("\nInsert strategies determine how operations are placed into moments.")

    q0, q1, q2 = cirq.LineQubit.range(3)

    # NEW_THEN_INLINE strategy (default)
    print("\n1. NEW_THEN_INLINE (Default Strategy):")
    print("   Adds to most recent moment if qubits are available,")
    print("   otherwise creates a new moment.")

    circuit_new = build_circuit_with_strategy(q0, q1, q2, cirq.InsertStrategy.NEW_THEN_INLINE)

    print(f"\n   Circuit with NEW_THEN_INLINE:")
    print(circuit_new)
    print(f"   Depth (moments): {len(circuit_new)}")

    # EARLIEST strategy
    print("\n2. EARLIEST Strategy:")
    print("   Searches backward to find earliest moment where operation fits.")
    print("   Creates more compact circuits - critical for NISQ hardware.")

    circuit_earliest = build_circuit_with_strategy(q0, q1, q2, cirq.InsertStrategy.EARLIEST)

    print(f"\n   Circuit with EARLIEST:")
    print(circuit_earliest)
    print(f"   Depth (moments): {len(circuit_earliest)}")

    print(f"\n   Comparison:")
    print(f"   NEW_THEN_INLINE depth: {len(circuit_new)}")
    print(f"   EARLIEST depth: {len(circuit_earliest)}")
    print(f"   EARLIEST is more compact, reducing circuit execution time")


def demonstrate_circuit_manipulation():
    """Demonstrate circuit inspection and manipulation."""
    _print_section_header("CIRCUIT MANIPULATION AND INSPECTION")

    q0, q1, q2 = cirq.LineQubit.range(3)

    # Create a sample circuit
    circuit = cirq.Circuit()
    circuit.append([cirq.H(q0), cirq.H(q1)])
    circuit.append(cirq.CNOT(q0, q1))
    circuit.append(cirq.X(q2))
    circuit.append(cirq.CNOT(q1, q2))

    print("\nSample circuit:")
    print(circuit)

    # Circuit depth
    print(f"\n1. Circuit depth (number of moments): {len(circuit)}")

    # All operations
    print(f"\n2. All operations:")
    for i, op in enumerate(circuit.all_operations()):
        print(f"   {i}: {op}")

    # All qubits
    print(f"\n3. All qubits used: {sorted(circuit.all_qubits())}")

    # Iterate over moments
    print(f"\n4. Moments in circuit:")
    for i, moment in enumerate(circuit):
        print(f"   Moment {i}: {list(moment)}")

    # Circuit slicing
    print(f"\n5. Circuit slicing (first 2 moments):")
    sliced = circuit[0:2]
    print(sliced)

    # Circuit concatenation
    print(f"\n6. Circuit concatenation:")
    circuit_extra = cirq.Circuit(cirq.measure(q0, q1, q2, key='result'))
    combined = circuit + circuit_extra
    print(combined)

    # Inserting operations
    print(f"\n7. Inserting operation at position 1:")
    circuit_copy = circuit.copy()
    circuit_copy.insert(1, cirq.Z(q0))
    print(circuit_copy)


def create_bell_state_circuit():
    """Create a Bell state preparation circuit."""
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    circuit.append(cirq.H(q0))
    circuit.append(cirq.CNOT(q0, q1))
    return circuit


def compare_insert_strategies():
    """Compare different insert strategies and return both circuits."""
    q0, q1, q2 = cirq.LineQubit.range(3)

    return {
        'NEW_THEN_INLINE': build_circuit_with_strategy(q0, q1, q2, cirq.InsertStrategy.NEW_THEN_INLINE),
        'EARLIEST': build_circuit_with_strategy(q0, q1, q2, cirq.InsertStrategy.EARLIEST)
    }


def get_circuit_depth(circuit):
    """Return the depth (number of moments) of a circuit."""
    return len(circuit)


def get_circuit_moments(circuit):
    """Return a list of all moments in a circuit."""
    return list(circuit)


def visualize_circuits():
    """Visualize different circuit patterns."""
    _print_section_header("CIRCUIT VISUALIZATION")

    # Create various circuits
    q0, q1, q2 = cirq.LineQubit.range(3)

    # 1. Sequential circuit
    circuit_sequential = cirq.Circuit()
    for gate in [cirq.H, cirq.X, cirq.Y, cirq.Z]:
        circuit_sequential.append(gate(q0))

    # 2. Parallel circuit
    circuit_parallel = cirq.Circuit()
    circuit_parallel.append([cirq.H(q) for q in [q0, q1, q2]])

    # 3. Bell state
    circuit_bell = create_bell_state_circuit()

    # 4. GHZ state
    circuit_ghz = cirq.Circuit()
    circuit_ghz.append(cirq.H(q0))
    circuit_ghz.append(cirq.CNOT(q0, q1))
    circuit_ghz.append(cirq.CNOT(q1, q2))

    # Display circuits
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Common Quantum Circuit Patterns', fontsize=16, fontweight='bold')

    circuits = [
        (circuit_sequential, "Sequential Operations", axes[0, 0]),
        (circuit_parallel, "Parallel Operations", axes[0, 1]),
        (circuit_bell, "Bell State Preparation", axes[1, 0]),
        (circuit_ghz, "GHZ State Preparation", axes[1, 1])
    ]

    for circuit, title, ax in circuits:
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

        # Create text representation
        diagram = str(circuit)
        ax.text(0.05, 0.5, diagram, fontfamily='monospace',
                fontsize=10, verticalalignment='center',
                transform=ax.transAxes)

    plt.tight_layout()
    print("\nDisplaying circuit pattern visualization...")
    plt.savefig('notebooks/circuit_patterns.png', dpi=150, bbox_inches='tight')
    print("  Saved to: notebooks/circuit_patterns.png")
    plt.show()


def demonstrate_complex_circuits():
    """Demonstrate more complex circuit patterns."""
    _print_section_header("COMPLEX CIRCUIT PATTERNS")

    qubits = cirq.LineQubit.range(4)

    # 1. Parallel Hadamards
    print("\n1. Parallel operations - all Hadamards execute simultaneously:")
    circuit_parallel = cirq.Circuit()
    circuit_parallel.append([cirq.H(q) for q in qubits])
    print(circuit_parallel)
    print(f"   Depth: {len(circuit_parallel)} (single moment)")
    print(f"   Operations: {len(list(circuit_parallel.all_operations()))}")

    # 2. Sequential operations
    print("\n2. Sequential operations on same qubit:")
    circuit_sequential = cirq.Circuit()
    for gate in [cirq.H, cirq.X, cirq.Y, cirq.Z]:
        circuit_sequential.append(gate(qubits[0]))
    print(circuit_sequential)
    print(f"   Depth: {len(circuit_sequential)} moments")

    # 3. GHZ state preparation
    print("\n3. GHZ state preparation (maximally entangled 4-qubit state):")
    circuit_ghz = cirq.Circuit()
    circuit_ghz.append(cirq.H(qubits[0]))
    for i in range(3):
        circuit_ghz.append(cirq.CNOT(qubits[i], qubits[i+1]))
    print(circuit_ghz)

    # 4. Circuit with measurements
    print("\n4. Complete circuit with measurements:")
    circuit_measure = cirq.Circuit()
    circuit_measure.append(cirq.H(qubits[0]))
    circuit_measure.append(cirq.CNOT(qubits[0], qubits[1]))
    circuit_measure.append(cirq.measure(qubits[0], qubits[1], key='result'))
    print(circuit_measure)


def demonstrate_circuit_properties():
    """Demonstrate various circuit property queries."""
    _print_section_header("CIRCUIT PROPERTIES AND STATISTICS")

    qubits = cirq.LineQubit.range(4)

    # Create sample circuit
    circuit = cirq.Circuit()
    circuit.append(cirq.H(qubits[0]))
    circuit.append([cirq.H(q) for q in qubits[1:]])
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    circuit.append(cirq.CNOT(qubits[2], qubits[3]))
    circuit.append(cirq.X(qubits[0]))

    print("\nSample circuit:")
    print(circuit)

    # Count operations
    all_ops = list(circuit.all_operations())
    print(f"\n1. Total operations: {len(all_ops)}")

    # Count two-qubit gates
    two_qubit_gates = sum(1 for op in all_ops if len(op.qubits) == 2)
    print(f"2. Two-qubit gates: {two_qubit_gates}")

    # Count single-qubit gates
    single_qubit_gates = sum(1 for op in all_ops if len(op.qubits) == 1)
    print(f"3. Single-qubit gates: {single_qubit_gates}")

    # Qubits used
    print(f"4. Qubits used: {sorted(circuit.all_qubits())}")
    print(f"5. Number of unique qubits: {len(circuit.all_qubits())}")

    # Depth
    print(f"6. Circuit depth (moments): {len(circuit)}")

    # Operations per moment
    print(f"\n7. Operations per moment:")
    for i, moment in enumerate(circuit):
        print(f"   Moment {i}: {len(moment)} operations - {list(moment)}")


def main():
    """Run all demonstrations for Section 1.3."""
    print("\n" + "#"*60)
    print("# SECTION 1.3: CIRCUITS - MOMENTS AND CIRCUIT CONSTRUCTION")
    print("# Moments, Insert Strategies, and Circuit Manipulation")
    print("#"*60)

    demonstrate_moments()
    demonstrate_circuit_construction()
    demonstrate_insert_strategies()
    demonstrate_circuit_manipulation()
    demonstrate_complex_circuits()
    demonstrate_circuit_properties()
    visualize_circuits()

    print("\n" + "#"*60)
    print("# Section 1.3 Complete!")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
