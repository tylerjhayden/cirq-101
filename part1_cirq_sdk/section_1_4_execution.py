# ABOUTME: Section 1.4 - Execution: running and analyzing quantum circuits with cirq.Simulator

import cirq
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path


def _print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(title)
    print("="*60)


def demonstrate_simulator_basics():
    """Demonstrate basic simulator initialization and usage."""
    _print_section_header("SIMULATOR BASICS")

    print("\nThe cirq.Simulator performs ideal (noiseless) simulation:")
    simulator = cirq.Simulator()
    print(f"  Simulator type: {type(simulator)}")
    print(f"  Simulator: {simulator}")

    print("\nSimulator executes circuits and returns quantum state information")


def demonstrate_run_vs_simulate():
    """Demonstrate the difference between run() and simulate() methods."""
    _print_section_header("RUN() VS SIMULATE()")

    q = cirq.LineQubit(0)
    simulator = cirq.Simulator()

    # simulate() - provides full state vector
    print("\n1. simulate() - Returns the complete quantum state vector")
    print("   This is impossible on real quantum hardware!")

    circuit = cirq.Circuit(cirq.H(q))
    print(f"\nCircuit:")
    print(circuit)

    result = simulator.simulate(circuit)
    print(f"\nFinal state vector:")
    print(f"  {result.final_state_vector}")
    print(f"\nInterpretation:")
    print(f"  |0⟩ amplitude: {result.final_state_vector[0]:.6f}")
    print(f"  |1⟩ amplitude: {result.final_state_vector[1]:.6f}")
    print(f"  Probabilities: P(|0⟩)={abs(result.final_state_vector[0])**2:.3f}, "
          f"P(|1⟩)={abs(result.final_state_vector[1])**2:.3f}")

    # run() - mimics real quantum computer
    print("\n2. run() - Mimics real quantum computers")
    print("   Returns only classical measurement outcomes")

    circuit_with_measurement = cirq.Circuit(
        cirq.H(q),
        cirq.measure(q, key='result')
    )

    print(f"\nCircuit with measurement:")
    print(circuit_with_measurement)

    result = simulator.run(circuit_with_measurement, repetitions=10)
    print(f"\nMeasurement results (10 shots):")
    print(f"  {result.measurements['result'].flatten()}")
    print(f"\nNote: No state vector is accessible with run()")


def create_bell_state():
    """Create and return a Bell state circuit."""
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1)
    )
    return circuit, q0, q1


def demonstrate_bell_state_creation():
    """Demonstrate Bell state preparation and measurement."""
    _print_section_header("BELL STATE CREATION AND MEASUREMENT")

    print("\nBell states are maximally entangled two-qubit states.")
    print("Creating |Φ+⟩ = (|00⟩ + |11⟩)/√2")

    circuit, q0, q1 = create_bell_state()

    print(f"\nBell state circuit:")
    print(circuit)

    # Analyze with simulate()
    print("\n--- State Vector Analysis ---")
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)

    print(f"\nFinal state vector:")
    for i, amplitude in enumerate(result.final_state_vector):
        basis_state = format(i, '02b')
        print(f"  |{basis_state}⟩: {amplitude:.6f}")

    print(f"\nExpected: [1/√2, 0, 0, 1/√2] = [0.707, 0, 0, 0.707]")
    print(f"This confirms perfect entanglement: only |00⟩ and |11⟩ appear!")

    # Analyze with run()
    print("\n--- Measurement Statistics ---")
    circuit.append(cirq.measure(q0, q1, key='result'))

    result = simulator.run(circuit, repetitions=1000)
    histogram = result.histogram(key='result')

    print(f"\nMeasurement outcomes (1000 shots):")
    for bitstring, count in sorted(histogram.items()):
        binary = format(bitstring, '02b')
        percentage = 100 * count / 1000
        print(f"  |{binary}⟩: {count:4d} times ({percentage:5.1f}%)")

    print(f"\nNotice:")
    print(f"  - Only |00⟩ and |11⟩ outcomes appear (never |01⟩ or |10⟩)")
    print(f"  - Perfect correlation: q0 and q1 always match")
    print(f"  - Roughly 50/50 distribution between |00⟩ and |11⟩")


def demonstrate_expectation_values():
    """Demonstrate expectation value calculations."""
    _print_section_header("EXPECTATION VALUE CALCULATIONS")

    print("\nExpectation values ⟨ψ|O|ψ⟩ are crucial for variational algorithms")
    print("(VQE, QAOA, quantum machine learning)")

    simulator = cirq.Simulator()

    # Example 1: Z expectation on |0⟩
    print("\n1. Z expectation on |0⟩ state:")
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.I(q))  # |0⟩ state

    result = simulator.simulate(circuit)
    observable = cirq.Z(q).with_qubits(q)
    expectation = observable.expectation_from_state_vector(
        result.final_state_vector,
        qubit_map={q: 0}
    )

    print(f"  ⟨0|Z|0⟩ = {expectation:.6f}")
    print(f"  Expected: +1 (eigenvalue of Z for |0⟩)")

    # Example 2: X expectation on |+⟩
    print("\n2. X expectation on |+⟩ state:")
    circuit = cirq.Circuit(cirq.H(q))  # |+⟩ state

    result = simulator.simulate(circuit)
    observable = cirq.X(q).with_qubits(q)
    expectation = observable.expectation_from_state_vector(
        result.final_state_vector,
        qubit_map={q: 0}
    )

    print(f"  ⟨+|X|+⟩ = {expectation:.6f}")
    print(f"  Expected: +1 (|+⟩ is +1 eigenstate of X)")

    # Example 3: Multiple observables on superposition
    print("\n3. Multiple observables on |+⟩ state:")
    circuit = cirq.Circuit(cirq.H(q))

    result = simulator.simulate(circuit)

    observables = {
        'X': cirq.X(q).with_qubits(q),
        'Y': cirq.Y(q).with_qubits(q),
        'Z': cirq.Z(q).with_qubits(q)
    }

    print(f"  State: |+⟩ = (|0⟩ + |1⟩)/√2")
    for name, obs in observables.items():
        exp = obs.expectation_from_state_vector(
            result.final_state_vector,
            qubit_map={q: 0}
        )
        # Hermitian operators (like X, Y, Z) guarantee real expectation values
        # Extract .real to discard numerical noise in imaginary part
        print(f"  ⟨{name}⟩ = {exp.real:.6f}")

    # Example 4: Two-qubit observable on Bell state
    print("\n4. ZZ expectation on Bell state |Φ+⟩:")
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1)
    )

    result = simulator.simulate(circuit)
    observable = cirq.Z(q0) * cirq.Z(q1)
    expectation = observable.expectation_from_state_vector(
        result.final_state_vector,
        qubit_map={q0: 0, q1: 1}
    )

    print(f"  ⟨Φ+|Z₀Z₁|Φ+⟩ = {expectation:.6f}")
    print(f"  Expected: +1 (Bell state has perfect Z-correlation)")


def visualize_measurement_histograms():
    """Visualize measurement histograms for different quantum states."""
    _print_section_header("MEASUREMENT HISTOGRAM VISUALIZATION")

    simulator = cirq.Simulator()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # State 1: Single qubit superposition
    print("\nGenerating histograms for:")
    print("  1. Single-qubit superposition: H|0⟩")

    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(
        cirq.H(q),
        cirq.measure(q, key='result')
    )

    result = simulator.run(circuit, repetitions=1000)
    histogram = result.histogram(key='result')

    axes[0].bar(histogram.keys(), histogram.values(), color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Measurement Outcome', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Single Qubit\nH|0⟩ State', fontsize=12, fontweight='bold')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['|0⟩', '|1⟩'])
    axes[0].grid(axis='y', alpha=0.3)

    # State 2: Bell state
    print("  2. Bell state: (|00⟩ + |11⟩)/√2")

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.measure(q0, q1, key='result')
    )

    result = simulator.run(circuit, repetitions=1000)
    histogram = result.histogram(key='result')

    labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
    counts = [histogram.get(i, 0) for i in range(4)]

    axes[1].bar(range(4), counts, color='coral', alpha=0.7)
    axes[1].set_xlabel('Measurement Outcome', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('Bell State\n(|00⟩ + |11⟩)/√2', fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(labels)
    axes[1].grid(axis='y', alpha=0.3)

    # State 3: GHZ state (three qubits)
    print("  3. GHZ state: (|000⟩ + |111⟩)/√2")

    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.CNOT(q0, q2),
        cirq.measure(q0, q1, q2, key='result')
    )

    result = simulator.run(circuit, repetitions=1000)
    histogram = result.histogram(key='result')

    labels = [f'|{format(i, "03b")}⟩' for i in range(8)]
    counts = [histogram.get(i, 0) for i in range(8)]

    axes[2].bar(range(8), counts, color='mediumseagreen', alpha=0.7)
    axes[2].set_xlabel('Measurement Outcome', fontsize=11)
    axes[2].set_ylabel('Count', fontsize=11)
    axes[2].set_title('GHZ State\n(|000⟩ + |111⟩)/√2', fontsize=12, fontweight='bold')
    axes[2].set_xticks(range(8))
    axes[2].set_xticklabels(labels, rotation=45, ha='right')
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    _notebooks_dir = Path(__file__).parent.parent / 'notebooks'
    _notebooks_dir.mkdir(exist_ok=True)
    plt.savefig(_notebooks_dir / 'measurement_histograms.png', dpi=150, bbox_inches='tight')
    print("\n  Saved to: notebooks/measurement_histograms.png")
    plt.show()


def visualize_state_vectors():
    """Visualize state vector amplitudes and phases."""
    _print_section_header("STATE VECTOR VISUALIZATION")

    simulator = cirq.Simulator()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    states = [
        ("Ground State |0⟩", cirq.Circuit(cirq.I(cirq.LineQubit(0)))),
        ("Excited State |1⟩", cirq.Circuit(cirq.X(cirq.LineQubit(0)))),
        ("Superposition |+⟩", cirq.Circuit(cirq.H(cirq.LineQubit(0)))),
        # H then S creates (|0⟩ + i|1⟩)/√2: equal superposition with π/2 phase
        ("Phase State |i⟩", cirq.Circuit([cirq.H(cirq.LineQubit(0)), cirq.S(cirq.LineQubit(0))])),
        # H on q0 then CNOT(q0,q1) creates (|00⟩ + |11⟩)/√2: maximally entangled
        ("Bell State |Φ+⟩", cirq.Circuit([cirq.H(cirq.LineQubit(0)), cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1))])),
        # W state: symmetric superposition |100⟩ + |010⟩ + |001⟩ (partial entanglement)
        ("W State", cirq.Circuit([cirq.X(cirq.LineQubit(0)), cirq.H(cirq.LineQubit(0)), cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1)), cirq.X(cirq.LineQubit(0))]))
    ]

    print("\nGenerating state vector visualizations...")

    for idx, (name, circuit) in enumerate(states):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        result = simulator.simulate(circuit)
        state_vector = result.final_state_vector

        # Plot amplitudes
        n_basis_states = len(state_vector)
        indices = np.arange(n_basis_states)
        amplitudes = np.abs(state_vector)
        phases = np.angle(state_vector)

        # Color by phase
        colors = plt.cm.hsv(phases / (2 * np.pi) + 0.5)

        bars = ax.bar(indices, amplitudes, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

        ax.set_xlabel('Basis State', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)

        # Set x-tick labels
        n_qubits = int(np.log2(n_basis_states))
        if n_qubits == 1:
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['|0⟩', '|1⟩'])
        elif n_qubits == 2:
            ax.set_xticks(range(4))
            ax.set_xticklabels(['|00⟩', '|01⟩', '|10⟩', '|11⟩'], fontsize=9)
        else:
            ax.set_xticks(range(n_basis_states))
            ax.set_xticklabels([f'|{format(i, f"0{n_qubits}b")}⟩' for i in range(n_basis_states)],
                             rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    _notebooks_dir = Path(__file__).parent.parent / 'notebooks'
    _notebooks_dir.mkdir(exist_ok=True)
    plt.savefig(_notebooks_dir / 'state_vectors.png', dpi=150, bbox_inches='tight')
    print("  Saved to: notebooks/state_vectors.png")
    plt.show()


def compare_run_simulate_performance():
    """Compare run() and simulate() for practical use cases."""
    _print_section_header("WHEN TO USE RUN() VS SIMULATE()")

    print("\nrun() - Use when:")
    print("  ✓ Preparing for real hardware experiments")
    print("  ✓ Need measurement statistics (histograms, probabilities)")
    print("  ✓ Testing sampling-based algorithms (QAOA, quantum ML)")
    print("  ✓ Want to mimic actual quantum computer behavior")

    print("\nsimulate() - Use when:")
    print("  ✓ Debugging circuit logic")
    print("  ✓ Need exact state vector or amplitudes")
    print("  ✓ Calculating expectation values")
    print("  ✓ Verifying theoretical predictions")
    print("  ✓ Teaching and understanding quantum mechanics")

    print("\nKey difference:")
    print("  run()      → Returns classical bits (like real hardware)")
    print("  simulate() → Returns quantum state (impossible on real hardware)")


def main():
    """Run all demonstrations for Section 1.4."""
    print("\n" + "#"*60)
    print("# SECTION 1.4: EXECUTING AND ANALYZING CIRCUITS")
    print("# Simulators, Measurements, and Quantum State Analysis")
    print("#"*60)

    demonstrate_simulator_basics()
    demonstrate_run_vs_simulate()
    demonstrate_bell_state_creation()
    demonstrate_expectation_values()
    compare_run_simulate_performance()
    visualize_measurement_histograms()
    visualize_state_vectors()

    print("\n" + "#"*60)
    print("# Section 1.4 Complete!")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
