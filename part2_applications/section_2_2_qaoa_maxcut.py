# ABOUTME: Implements Quantum Approximate Optimization Algorithm (QAOA) for solving the Max-Cut problem on weighted graphs.
# Demonstrates variational quantum-classical optimization with problem and mixer Hamiltonians.

import cirq
import networkx as nx
import sympy
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# QAOA symbolic parameters used throughout the module
GAMMA_SYMBOL = sympy.Symbol('gamma')
BETA_SYMBOL = sympy.Symbol('beta')


def create_graph() -> nx.Graph:
    """
    Create a weighted graph for the Max-Cut problem.

    Returns:
        NetworkX graph with weighted edges
    """
    G = nx.Graph()
    G.add_weighted_edges_from([
        (0, 1, 5.0),   # Strong connection
        (0, 3, 2.0),
        (1, 2, 3.0),
        (1, 3, 1.0),
        (2, 3, 4.0),
        (2, 4, 6.0),   # Strong connection
        (3, 4, 2.5)
    ])
    return G


def build_qaoa_circuit(graph: nx.Graph, gamma: sympy.Symbol, beta: sympy.Symbol) -> cirq.Circuit:
    """
    Build a single-layer QAOA circuit for Max-Cut.

    The circuit applies:
    1. Hadamard gates to create uniform superposition
    2. Cost unitary exp(-i * gamma * H_C) encoding the Max-Cut problem
    3. Mixer unitary exp(-i * beta * H_M) for state exploration

    Args:
        graph: NetworkX graph defining the Max-Cut problem
        gamma: Symbolic parameter for cost layer
        beta: Symbolic parameter for mixer layer

    Returns:
        Parameterized QAOA circuit
    """
    qubits = sorted([cirq.LineQubit(i) for i in graph.nodes()])
    circuit = cirq.Circuit()

    # Initialize uniform superposition
    circuit.append(cirq.H.on_each(*qubits))

    # Cost layer: encode problem Hamiltonian H_C = sum_edges w_ij * (I - Z_i Z_j) / 2
    # Physics: Edges with qubits in different partitions (Z_i ≠ Z_j) contribute energy w_ij.
    # This encoding makes the Max-Cut value equal to ground state energy of -H_C.
    # Maximizing cut = minimizing -cut = finding ground state of negative weight Hamiltonian.
    # Implements exp(-i * gamma * w_ij * (I - Z_i Z_j) / 2) using CNOT-RZ-CNOT decomposition.
    for u, v, data in graph.edges(data=True):
        weight = data['weight']
        circuit.append([
            cirq.CNOT(qubits[u], qubits[v]),
            cirq.rz(2 * gamma * weight).on(qubits[v]),
            cirq.CNOT(qubits[u], qubits[v])
        ])

    # Mixer layer: implement exp(-i * beta * H_M) where H_M = sum_i X_i
    # Applies RX(-2*beta) to all qubits
    circuit.append(cirq.rx(-2 * beta).on_each(*qubits))

    # Add measurements
    circuit.append(cirq.measure(*qubits, key='result'))

    return circuit


def calculate_cut_value(bitstring: np.ndarray, graph: nx.Graph) -> float:
    """
    Calculate the cut value for a given partition.

    The cut value is the sum of weights of edges crossing the partition.

    Args:
        bitstring: Binary array indicating partition (0 or 1 for each node)
        graph: NetworkX graph

    Returns:
        Total weight of edges crossing the partition
    """
    cut_value = 0.0
    for u, v, data in graph.edges(data=True):
        if bitstring[u] != bitstring[v]:
            cut_value += data['weight']
    return cut_value


def qaoa_objective(params: List[float], graph: nx.Graph, simulator: cirq.Simulator,
                   repetitions: int = 5000) -> float:
    """
    Objective function for QAOA optimization.

    Evaluates the negative average cut value (negative because we minimize but want to maximize cut).

    Args:
        params: [gamma, beta] parameter values
        graph: NetworkX graph
        simulator: Cirq simulator
        repetitions: Number of measurement samples

    Returns:
        Negative average cut value (for minimization)
    """
    gamma_val, beta_val = params

    # Build circuit with resolved parameters
    circuit = build_qaoa_circuit(graph, GAMMA_SYMBOL, BETA_SYMBOL)

    # Resolve parameters
    resolver = cirq.ParamResolver({GAMMA_SYMBOL: gamma_val, BETA_SYMBOL: beta_val})
    resolved_circuit = cirq.resolve_parameters(circuit, resolver)

    # Run circuit and collect measurements
    results = simulator.run(resolved_circuit, repetitions=repetitions)
    measurements = results.measurements['result']

    # Calculate average cut value
    total_cost = 0.0
    for sample in measurements:
        total_cost += calculate_cut_value(sample, graph)

    average_cost = total_cost / repetitions

    # Return negative (we minimize but want to maximize cut)
    return -average_cost


def optimize_variational_parameters(graph: nx.Graph, simulator: cirq.Simulator,
                                    gamma_range: np.ndarray, beta_range: np.ndarray) -> Tuple[float, float, float]:
    """
    Find optimal variational parameters for QAOA via grid search.

    Args:
        graph: NetworkX graph
        simulator: Cirq simulator
        gamma_range: Array of gamma values to try
        beta_range: Array of beta values to try

    Returns:
        Tuple of (best_gamma, best_beta, best_cost)
    """
    print("Performing grid search optimization...")
    print(f"  Grid size: {len(gamma_range)} x {len(beta_range)} = {len(gamma_range) * len(beta_range)} points")

    cost_grid = np.zeros((len(gamma_range), len(beta_range)))

    for i, gamma in enumerate(gamma_range):
        for j, beta in enumerate(beta_range):
            cost_grid[i, j] = -qaoa_objective([gamma, beta], graph, simulator, repetitions=1000)

        # Progress indicator
        print(f"  Progress: {i+1}/{len(gamma_range)} gamma values completed")

    # Find best parameters
    best_idx = np.unravel_index(np.argmax(cost_grid), cost_grid.shape)
    best_gamma = gamma_range[best_idx[0]]
    best_beta = beta_range[best_idx[1]]
    best_cost = np.max(cost_grid)

    return best_gamma, best_beta, best_cost


def extract_solution(graph: nx.Graph, gamma: float, beta: float,
                     simulator: cirq.Simulator, repetitions: int = 5000) -> Tuple[List[int], float]:
    """
    Extract the most likely solution from optimized QAOA circuit.

    Args:
        graph: NetworkX graph
        gamma: Optimized gamma parameter
        beta: Optimized beta parameter
        simulator: Cirq simulator
        repetitions: Number of measurement samples

    Returns:
        Tuple of (best_partition, cut_value)
    """
    # Build and run circuit
    circuit = build_qaoa_circuit(graph, GAMMA_SYMBOL, BETA_SYMBOL)

    resolver = cirq.ParamResolver({GAMMA_SYMBOL: gamma, BETA_SYMBOL: beta})
    resolved_circuit = cirq.resolve_parameters(circuit, resolver)

    results = simulator.run(resolved_circuit, repetitions=repetitions)
    histogram = results.histogram(key='result')

    # Get most common bitstring
    most_common_int, count = histogram.most_common(1)[0]

    # Convert integer to bitstring
    num_qubits = len(graph.nodes())
    bitstring = [int(b) for b in format(most_common_int, f'0{num_qubits}b')]

    # Calculate cut value
    cut_value = calculate_cut_value(np.array(bitstring), graph)

    print(f"\nSolution extraction:")
    print(f"  Most common bitstring: {bitstring}")
    print(f"  Observed frequency: {count}/{repetitions} ({100*count/repetitions:.1f}%)")
    print(f"  Cut value: {cut_value:.2f}")

    return bitstring, cut_value


def visualize_solution(graph: nx.Graph, partition: List[int], cut_value: float):
    """
    Visualize the graph with the Max-Cut partition.

    Args:
        graph: NetworkX graph
        partition: Binary partition of nodes
        cut_value: Total weight of cut edges
    """
    # Set up colors
    colors = ['red' if bit == 0 else 'blue' for bit in partition]

    # Create figure
    plt.figure(figsize=(10, 8))

    # Draw graph
    pos = nx.spring_layout(graph, seed=42)
    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=800, alpha=0.9)
    nx.draw_networkx_labels(graph, pos, font_size=16, font_color='white', font_weight='bold')

    # Draw edges - highlight cut edges
    for u, v, data in graph.edges(data=True):
        weight = data['weight']
        if partition[u] != partition[v]:
            # Cut edge - draw thick and highlighted
            nx.draw_networkx_edges(graph, pos, [(u, v)], width=3, edge_color='green', alpha=0.8)
        else:
            # Non-cut edge - draw thin and faded
            nx.draw_networkx_edges(graph, pos, [(u, v)], width=1, edge_color='gray', alpha=0.3)

    # Draw edge labels
    edge_labels = {(u, v): f"{data['weight']:.1f}" for u, v, data in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=10)

    plt.title(f"QAOA Max-Cut Solution\nCut Value: {cut_value:.2f}", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_optimization_landscape(graph: nx.Graph, simulator: cirq.Simulator,
                                     gamma_range: np.ndarray, beta_range: np.ndarray,
                                     best_gamma: float, best_beta: float):
    """
    Visualize the QAOA optimization landscape.

    Args:
        graph: NetworkX graph
        simulator: Cirq simulator
        gamma_range: Array of gamma values
        beta_range: Array of beta values
        best_gamma: Optimal gamma value
        best_beta: Optimal beta value
    """
    print("\nGenerating optimization landscape visualization...")

    # Create cost grid
    cost_grid = np.zeros((len(gamma_range), len(beta_range)))
    for i, gamma in enumerate(gamma_range):
        for j, beta in enumerate(beta_range):
            cost_grid[i, j] = -qaoa_objective([gamma, beta], graph, simulator, repetitions=500)

    # Create contour plot
    plt.figure(figsize=(10, 8))

    # Create meshgrid
    Gamma, Beta = np.meshgrid(gamma_range, beta_range, indexing='ij')

    # Contour plot
    levels = 15
    contour = plt.contourf(Gamma, Beta, cost_grid, levels=levels, cmap='viridis')
    plt.colorbar(contour, label='Average Cut Value')

    # Overlay contour lines
    plt.contour(Gamma, Beta, cost_grid, levels=levels, colors='white', alpha=0.3, linewidths=0.5)

    # Mark optimal point
    plt.plot(best_gamma, best_beta, 'r*', markersize=20, label='Optimal Parameters')

    plt.xlabel('γ (Gamma)', fontsize=12)
    plt.ylabel('β (Beta)', fontsize=12)
    plt.title('QAOA Optimization Landscape for Max-Cut', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def calculate_approximation_ratio(achieved_cut: float, graph: nx.Graph) -> float:
    """
    Calculate the approximation ratio compared to the maximum possible cut.

    For many graphs, finding the exact maximum cut is NP-hard, so we estimate
    by trying a few heuristic partitions.

    Note: Uses fixed random seed (42) for reproducibility. The approximation ratio
    may vary with different seeds, but this provides a consistent baseline estimate.

    Args:
        achieved_cut: Cut value achieved by QAOA
        graph: NetworkX graph

    Returns:
        Approximation ratio (achieved / estimated_max)
    """
    # Simple heuristic: try a few random partitions and greedy partitioning
    max_cut = achieved_cut
    num_nodes = len(graph.nodes())

    # Try random partitions (caller is responsible for seeding if reproducibility is needed)
    rng = np.random.default_rng(42)
    for _ in range(100):
        partition = rng.integers(0, 2, num_nodes)
        cut_val = calculate_cut_value(partition, graph)
        max_cut = max(max_cut, cut_val)

    # Try greedy algorithm
    partition = np.zeros(num_nodes, dtype=int)
    for node in graph.nodes():
        # Put node in partition that maximizes cut
        partition[node] = 0
        cut_0 = calculate_cut_value(partition, graph)
        partition[node] = 1
        cut_1 = calculate_cut_value(partition, graph)
        partition[node] = 1 if cut_1 > cut_0 else 0

    greedy_cut = calculate_cut_value(partition, graph)
    max_cut = max(max_cut, greedy_cut)

    ratio = achieved_cut / max_cut if max_cut > 0 else 1.0
    return ratio


def main():
    """
    Run complete QAOA Max-Cut demonstration.

    Demonstrates:
    1. Graph construction
    2. QAOA circuit building with parameterized gates
    3. Grid search optimization
    4. Solution extraction and validation
    5. Visualization of solution and optimization landscape
    """
    print("=" * 70)
    print("QAOA for Max-Cut Problem")
    print("=" * 70)

    # Create graph
    print("\n1. Creating weighted graph...")
    graph = create_graph()
    print(f"   Nodes: {graph.number_of_nodes()}")
    print(f"   Edges: {graph.number_of_edges()}")
    print(f"   Total edge weight: {sum(data['weight'] for _, _, data in graph.edges(data=True)):.1f}")

    # Initialize simulator
    simulator = cirq.Simulator()

    # Define parameter ranges for grid search
    print("\n2. Setting up optimization...")
    gamma_range = np.linspace(0, np.pi, 20)
    beta_range = np.linspace(0, np.pi/2, 20)

    # Perform optimization
    print("\n3. Running optimization...")
    best_gamma, best_beta, best_cost = optimize_variational_parameters(
        graph, simulator, gamma_range, beta_range
    )

    print(f"\n   Optimization complete!")
    print(f"   Best parameters: γ = {best_gamma:.4f}, β = {best_beta:.4f}")
    print(f"   Best cut value: {best_cost:.2f}")

    # Extract solution
    print("\n4. Extracting solution...")
    partition, cut_value = extract_solution(graph, best_gamma, best_beta, simulator, repetitions=10000)

    # Calculate approximation ratio
    approx_ratio = calculate_approximation_ratio(cut_value, graph)
    print(f"   Approximation ratio: {approx_ratio:.3f}")

    # Verify solution
    print("\n5. Solution verification:")
    print(f"   Partition: {partition}")
    total_weight = sum(data['weight'] for _, _, data in graph.edges(data=True))
    print(f"   Cut percentage: {100 * cut_value / total_weight:.1f}%")

    # List cut edges
    print(f"\n   Cut edges:")
    for u, v, data in graph.edges(data=True):
        if partition[u] != partition[v]:
            print(f"     Edge ({u}, {v}): weight {data['weight']:.1f}")

    # Visualize solution
    print("\n6. Generating visualizations...")
    visualize_solution(graph, partition, cut_value)

    # Visualize optimization landscape
    visualize_optimization_landscape(graph, simulator, gamma_range, beta_range, best_gamma, best_beta)

    print("\n" + "=" * 70)
    print("QAOA Max-Cut demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
