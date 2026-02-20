# ABOUTME: Section 1.2 - Building Blocks: qubits, gates, operations, and gate protocols in Cirq

import cirq
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def _print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(title)
    print("="*60)


def demonstrate_qubit_types():
    """Demonstrate different qubit representations in Cirq."""
    _print_section_header("QUBIT TYPES")

    # LineQubits: Qubits arranged in a line
    print("\n1. LineQubits - arranged in a line:")
    line_qubits = cirq.LineQubit.range(3)
    for q in line_qubits:
        print(f"   {q}")

    # GridQubits: Qubits arranged in a 2D grid
    print("\n2. GridQubits - arranged in a 2D grid:")
    grid_qubits = cirq.GridQubit.square(2)
    for q in grid_qubits:
        print(f"   {q}")

    # NamedQubits: Custom string identifiers
    print("\n3. NamedQubits - custom identifiers:")
    alice = cirq.NamedQubit("alice")
    bob = cirq.NamedQubit("bob")
    print(f"   {alice}")
    print(f"   {bob}")


def demonstrate_gates_vs_operations():
    """Demonstrate the distinction between gates and operations."""
    _print_section_header("GATES VS OPERATIONS")

    print("\nGates are abstract templates, operations apply gates to specific qubits:")

    q = cirq.LineQubit(0)

    # Gate: abstract Hadamard gate
    h_gate = cirq.H
    print(f"\nGate: {h_gate}")
    print(f"  Type: {type(h_gate)}")

    # Operation: H gate applied to qubit 0
    h_operation = cirq.H(q)
    print(f"\nOperation: {h_operation}")
    print(f"  Type: {type(h_operation)}")
    print(f"  Targets: {h_operation.qubits}")
    print(f"  Gate: {h_operation.gate}")


def demonstrate_common_gates():
    """Demonstrate common single and two-qubit gates."""
    _print_section_header("COMMON QUANTUM GATES")

    q0, q1 = cirq.LineQubit.range(2)

    print("\nSingle-qubit gates:")
    single_gates = [
        (cirq.H, "Hadamard - creates superposition"),
        (cirq.X, "Pauli-X - bit flip (NOT gate)"),
        (cirq.Y, "Pauli-Y - bit and phase flip"),
        (cirq.Z, "Pauli-Z - phase flip"),
        (cirq.S, "S gate - sqrt(Z)"),
        (cirq.T, "T gate - fourth root of Z"),
    ]

    for gate, description in single_gates:
        print(f"  {str(gate(q0)).ljust(15)} - {description}")

    print("\nParameterized rotation gates:")
    angle = np.pi / 4
    rotations = [
        (cirq.rx(angle), "Rotation about X-axis"),
        (cirq.ry(angle), "Rotation about Y-axis"),
        (cirq.rz(angle), "Rotation about Z-axis"),
    ]

    for gate_op, description in rotations:
        print(f"  {str(gate_op(q0)).ljust(25)} - {description}")

    print("\nTwo-qubit gates:")
    two_qubit_gates = [
        (cirq.CNOT, "CNOT - controlled-NOT (entangling gate)"),
        (cirq.CZ, "CZ - controlled-Z (entangling gate)"),
        (cirq.SWAP, "SWAP - exchanges quantum states"),
    ]

    for gate, description in two_qubit_gates:
        print(f"  {str(gate(q0, q1)).ljust(20)} - {description}")


def demonstrate_gate_protocols():
    """Demonstrate gate protocol capabilities."""
    _print_section_header("GATE PROTOCOLS")

    # Unitary protocol: get matrix representation
    print("\n1. Unitary Protocol - matrix representation:")
    h_matrix = cirq.unitary(cirq.H)
    print(f"\nHadamard matrix:")
    print(h_matrix)

    # Verify unitarity: U†U = I
    print(f"\nVerify H is unitary (H†H = I):")
    identity = h_matrix.conj().T @ h_matrix
    print(f"H†H =")
    print(identity)
    print(f"Is identity? {np.allclose(identity, np.eye(2))}")

    # Inverse protocol
    print("\n2. Inverse Protocol:")
    q = cirq.LineQubit(0)
    s_op = cirq.S(q)
    s_inv_op = cirq.inverse(s_op)
    print(f"S gate: {s_op}")
    print(f"S† gate: {s_inv_op}")

    # Verify S * S† = I
    s_matrix = cirq.unitary(cirq.S)
    s_inv_matrix = cirq.unitary(cirq.S ** -1)
    result = s_matrix @ s_inv_matrix
    print(f"\nS * S† =")
    print(result)
    print(f"Is identity? {np.allclose(result, np.eye(2))}")

    # Power protocol
    print("\n3. Power Protocol - gate exponentiation:")
    print(f"X^(1/2) (sqrt-X):")
    sqrt_x = cirq.X ** 0.5
    print(cirq.unitary(sqrt_x))

    print(f"\nX^2 should equal I:")
    x_squared = cirq.X ** 2
    print(cirq.unitary(x_squared))

    # Decomposition protocol
    print("\n4. Decomposition Protocol:")
    q0, q1 = cirq.LineQubit.range(2)
    swap_op = cirq.SWAP(q0, q1)
    print(f"\nOriginal operation: {swap_op}")
    print(f"Decomposed into:")
    for op in cirq.decompose(swap_op):
        print(f"  {op}")


def demonstrate_measurements():
    """Demonstrate measurement operations."""
    _print_section_header("MEASUREMENT OPERATIONS")

    print("\nMeasurements extract classical information from quantum states:")

    q0 = cirq.LineQubit(0)
    measurement_single = cirq.measure(q0, key='result')
    print(f"\nSingle qubit measurement: {measurement_single}")

    qubits = cirq.LineQubit.range(3)
    measurement_multi = cirq.measure(*qubits, key='results')
    print(f"Multi-qubit measurement: {measurement_multi}")
    print(f"  Measures {len(measurement_multi.qubits)} qubits jointly")


def visualize_gate_matrices():
    """Visualize unitary matrices of common gates."""
    _print_section_header("GATE MATRIX VISUALIZATION")

    gates = [
        ("Pauli-X", cirq.X),
        ("Pauli-Y", cirq.Y),
        ("Pauli-Z", cirq.Z),
        ("Hadamard", cirq.H),
        ("S", cirq.S),
        ("T", cirq.T),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (name, gate) in enumerate(gates):
        matrix = cirq.unitary(gate)

        # Plot magnitude
        im = axes[idx].imshow(np.abs(matrix), cmap='viridis', vmin=0, vmax=1)
        axes[idx].set_title(f"{name} Gate", fontsize=12, fontweight='bold')
        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xlabel('Column')
        axes[idx].set_ylabel('Row')

        # Add colorbar
        plt.colorbar(im, ax=axes[idx], fraction=0.046)

        # Add matrix values as text
        for i in range(2):
            for j in range(2):
                val = matrix[i, j]
                text_str = f"{val.real:.2f}"
                if abs(val.imag) > 1e-10:
                    text_str += f"\n{val.imag:+.2f}i"
                axes[idx].text(j, i, text_str, ha="center", va="center",
                             color="white" if np.abs(matrix[i, j]) > 0.5 else "black",
                             fontsize=9)

    plt.tight_layout()
    print("\nDisplaying gate matrix visualization...")
    _notebooks_dir = Path(__file__).parent.parent / 'notebooks'
    _notebooks_dir.mkdir(exist_ok=True)
    plt.savefig(_notebooks_dir / 'gate_matrices.png', dpi=150, bbox_inches='tight')
    print("  Saved to: notebooks/gate_matrices.png")
    plt.show()


def main():
    """Run all demonstrations for Section 1.2."""
    print("\n" + "#"*60)
    print("# SECTION 1.2: BUILDING BLOCKS OF CIRQ")
    print("# Qubits, Gates, Operations, and Protocols")
    print("#"*60)

    demonstrate_qubit_types()
    demonstrate_gates_vs_operations()
    demonstrate_common_gates()
    demonstrate_gate_protocols()
    demonstrate_measurements()
    visualize_gate_matrices()

    print("\n" + "#"*60)
    print("# Section 1.2 Complete!")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
