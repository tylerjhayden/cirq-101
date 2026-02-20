# ABOUTME: Tests for Section 3.1 - Comparative Analysis of variational quantum algorithms
# (VQE, QAOA, QML) covering unified framework, pattern identification, and visualization

import pytest
import numpy as np
import cirq
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


class TestUnifiedFramework:
    """Tests for the unified variational framework abstraction."""

    def test_unified_framework_structure_exists(self):
        """Verify unified framework describes common structure of variational algorithms."""
        from part3_synthesis.section_3_1_comparative_analysis import get_unified_framework_structure

        structure = get_unified_framework_structure()

        assert isinstance(structure, dict)
        assert 'components' in structure
        assert 'quantum_circuit' in structure['components']
        assert 'classical_optimizer' in structure['components']
        assert 'objective_function' in structure['components']

    def test_unified_framework_identifies_quantum_evaluator(self):
        """Verify framework identifies quantum circuit as evaluator component."""
        from part3_synthesis.section_3_1_comparative_analysis import get_unified_framework_structure

        structure = get_unified_framework_structure()

        quantum_component = structure['components']['quantum_circuit']
        assert 'role' in quantum_component
        assert 'evaluator' in quantum_component['role'].lower()

    def test_unified_framework_identifies_classical_optimizer(self):
        """Verify framework identifies classical optimizer component."""
        from part3_synthesis.section_3_1_comparative_analysis import get_unified_framework_structure

        structure = get_unified_framework_structure()

        optimizer_component = structure['components']['classical_optimizer']
        assert 'role' in optimizer_component
        assert 'parameter' in optimizer_component['role'].lower()

    def test_unified_framework_describes_feedback_loop(self):
        """Verify framework describes quantum-classical feedback loop."""
        from part3_synthesis.section_3_1_comparative_analysis import get_unified_framework_structure

        structure = get_unified_framework_structure()

        assert 'loop' in structure or 'feedback' in structure
        loop_info = structure.get('loop') or structure.get('feedback')
        assert loop_info is not None


class TestAlgorithmComparison:
    """Tests for comparing VQE, QAOA, and QML algorithms."""

    def test_comparison_table_has_three_algorithms(self):
        """Verify comparison includes VQE, QAOA, and QML."""
        from part3_synthesis.section_3_1_comparative_analysis import get_algorithm_comparison

        comparison = get_algorithm_comparison()

        assert isinstance(comparison, dict)
        assert 'VQE' in comparison
        assert 'QAOA' in comparison
        assert 'QML' in comparison

    def test_comparison_identifies_vqe_goal(self):
        """Verify VQE goal is identified as eigenvalue/energy minimization."""
        from part3_synthesis.section_3_1_comparative_analysis import get_algorithm_comparison

        comparison = get_algorithm_comparison()
        vqe_info = comparison['VQE']

        assert 'goal' in vqe_info
        goal = vqe_info['goal'].lower()
        assert 'energy' in goal or 'eigenvalue' in goal or 'ground state' in goal

    def test_comparison_identifies_qaoa_goal(self):
        """Verify QAOA goal is combinatorial optimization."""
        from part3_synthesis.section_3_1_comparative_analysis import get_algorithm_comparison

        comparison = get_algorithm_comparison()
        qaoa_info = comparison['QAOA']

        assert 'goal' in qaoa_info
        goal = qaoa_info['goal'].lower()
        assert 'optimization' in goal or 'combinatorial' in goal or 'max' in goal

    def test_comparison_identifies_qml_goal(self):
        """Verify QML goal is classification or machine learning."""
        from part3_synthesis.section_3_1_comparative_analysis import get_algorithm_comparison

        comparison = get_algorithm_comparison()
        qml_info = comparison['QML']

        assert 'goal' in qml_info
        goal = qml_info['goal'].lower()
        assert 'classification' in goal or 'learning' in goal or 'machine learning' in goal

    def test_comparison_includes_quantum_objective(self):
        """Verify each algorithm has quantum objective function defined."""
        from part3_synthesis.section_3_1_comparative_analysis import get_algorithm_comparison

        comparison = get_algorithm_comparison()

        for alg_name in ['VQE', 'QAOA', 'QML']:
            alg_info = comparison[alg_name]
            assert 'quantum_objective' in alg_info or 'objective' in alg_info

    def test_comparison_includes_ansatz_description(self):
        """Verify each algorithm has ansatz/circuit structure described."""
        from part3_synthesis.section_3_1_comparative_analysis import get_algorithm_comparison

        comparison = get_algorithm_comparison()

        for alg_name in ['VQE', 'QAOA', 'QML']:
            alg_info = comparison[alg_name]
            assert 'ansatz' in alg_info or 'circuit' in alg_info


class TestProblemMapping:
    """Tests for problem mapping strategies."""

    def test_problem_mapping_strategies_exist(self):
        """Verify problem mapping strategies are documented."""
        from part3_synthesis.section_3_1_comparative_analysis import get_problem_mapping_strategies

        strategies = get_problem_mapping_strategies()

        assert isinstance(strategies, dict)
        assert len(strategies) >= 3  # At least one per algorithm

    def test_vqe_maps_to_hamiltonian(self):
        """Verify VQE maps chemistry problems to Hamiltonians."""
        from part3_synthesis.section_3_1_comparative_analysis import get_problem_mapping_strategies

        strategies = get_problem_mapping_strategies()

        assert 'VQE' in strategies
        vqe_mapping = strategies['VQE']
        assert 'hamiltonian' in str(vqe_mapping).lower()

    def test_qaoa_maps_to_cost_function(self):
        """Verify QAOA maps optimization to cost Hamiltonian."""
        from part3_synthesis.section_3_1_comparative_analysis import get_problem_mapping_strategies

        strategies = get_problem_mapping_strategies()

        assert 'QAOA' in strategies
        qaoa_mapping = strategies['QAOA']
        mapping_str = str(qaoa_mapping).lower()
        assert 'cost' in mapping_str or 'optimization' in mapping_str

    def test_qml_maps_classical_data_to_quantum(self):
        """Verify QML maps classical data to quantum states."""
        from part3_synthesis.section_3_1_comparative_analysis import get_problem_mapping_strategies

        strategies = get_problem_mapping_strategies()

        assert 'QML' in strategies
        qml_mapping = strategies['QML']
        mapping_str = str(qml_mapping).lower()
        assert 'data' in mapping_str or 'encoding' in mapping_str or 'feature' in mapping_str


class TestAnsatzDesign:
    """Tests for ansatz design principles."""

    def test_ansatz_design_principles_exist(self):
        """Verify ansatz design principles are documented."""
        from part3_synthesis.section_3_1_comparative_analysis import get_ansatz_design_principles

        principles = get_ansatz_design_principles()

        assert isinstance(principles, dict)
        assert len(principles) > 0

    def test_ansatz_principles_include_expressibility(self):
        """Verify principles discuss expressibility."""
        from part3_synthesis.section_3_1_comparative_analysis import get_ansatz_design_principles

        principles = get_ansatz_design_principles()
        principles_str = str(principles).lower()

        assert 'express' in principles_str or 'hilbert' in principles_str

    def test_ansatz_principles_include_hardware_efficiency(self):
        """Verify principles discuss hardware efficiency and depth."""
        from part3_synthesis.section_3_1_comparative_analysis import get_ansatz_design_principles

        principles = get_ansatz_design_principles()
        principles_str = str(principles).lower()

        assert 'hardware' in principles_str or 'depth' in principles_str or 'efficient' in principles_str

    def test_ansatz_principles_include_entanglement(self):
        """Verify principles discuss entanglement structure."""
        from part3_synthesis.section_3_1_comparative_analysis import get_ansatz_design_principles

        principles = get_ansatz_design_principles()
        principles_str = str(principles).lower()

        assert 'entangle' in principles_str or 'correlation' in principles_str


class TestCommonPatterns:
    """Tests for identifying common patterns across algorithms."""

    def test_common_patterns_identified(self):
        """Verify common patterns across algorithms are identified."""
        from part3_synthesis.section_3_1_comparative_analysis import identify_common_patterns

        patterns = identify_common_patterns()

        assert isinstance(patterns, (list, dict))
        assert len(patterns) > 0

    def test_parameterized_circuits_pattern_identified(self):
        """Verify parameterized quantum circuits are recognized as common pattern."""
        from part3_synthesis.section_3_1_comparative_analysis import identify_common_patterns

        patterns = identify_common_patterns()
        patterns_str = str(patterns).lower()

        assert 'parameter' in patterns_str or 'variational' in patterns_str

    def test_classical_optimization_pattern_identified(self):
        """Verify classical optimization loop is recognized as common pattern."""
        from part3_synthesis.section_3_1_comparative_analysis import identify_common_patterns

        patterns = identify_common_patterns()
        patterns_str = str(patterns).lower()

        assert 'classical' in patterns_str and 'optim' in patterns_str

    def test_expectation_value_pattern_identified(self):
        """Verify expectation value evaluation is recognized as common pattern."""
        from part3_synthesis.section_3_1_comparative_analysis import identify_common_patterns

        patterns = identify_common_patterns()
        patterns_str = str(patterns).lower()

        assert 'expectation' in patterns_str or 'measurement' in patterns_str


class TestCostFunctionStructure:
    """Tests for cost function structure comparison."""

    def test_cost_function_structures_documented(self):
        """Verify cost function structures for each algorithm are documented."""
        from part3_synthesis.section_3_1_comparative_analysis import get_cost_function_structures

        structures = get_cost_function_structures()

        assert isinstance(structures, dict)
        assert 'VQE' in structures
        assert 'QAOA' in structures
        assert 'QML' in structures

    def test_vqe_cost_is_hamiltonian_expectation(self):
        """Verify VQE cost function is Hamiltonian expectation value."""
        from part3_synthesis.section_3_1_comparative_analysis import get_cost_function_structures

        structures = get_cost_function_structures()
        vqe_cost = str(structures['VQE']).lower()

        assert 'hamiltonian' in vqe_cost or '<h>' in vqe_cost or 'expectation' in vqe_cost

    def test_qaoa_cost_is_problem_hamiltonian(self):
        """Verify QAOA cost function is problem Hamiltonian expectation."""
        from part3_synthesis.section_3_1_comparative_analysis import get_cost_function_structures

        structures = get_cost_function_structures()
        qaoa_cost = str(structures['QAOA']).lower()

        assert 'cost' in qaoa_cost or 'h_c' in qaoa_cost or 'problem' in qaoa_cost

    def test_qml_cost_includes_loss_function(self):
        """Verify QML cost function includes classical loss on quantum outputs."""
        from part3_synthesis.section_3_1_comparative_analysis import get_cost_function_structures

        structures = get_cost_function_structures()
        qml_cost = str(structures['QML']).lower()

        assert 'loss' in qml_cost or 'label' in qml_cost or 'cross' in qml_cost


class TestOptimizerRequirements:
    """Tests for optimizer requirements analysis."""

    def test_optimizer_requirements_documented(self):
        """Verify optimizer requirements for each algorithm are documented."""
        from part3_synthesis.section_3_1_comparative_analysis import get_optimizer_requirements

        requirements = get_optimizer_requirements()

        assert isinstance(requirements, dict)
        assert 'VQE' in requirements
        assert 'QAOA' in requirements
        assert 'QML' in requirements

    def test_optimizers_handle_noisy_objectives(self):
        """Verify optimizer requirements mention handling noisy objectives."""
        from part3_synthesis.section_3_1_comparative_analysis import get_optimizer_requirements

        requirements = get_optimizer_requirements()
        requirements_str = str(requirements).lower()

        # Variational algorithms use shot-based measurements leading to noise
        assert 'noise' in requirements_str or 'shot' in requirements_str or 'stochastic' in requirements_str

    def test_gradient_free_methods_discussed(self):
        """Verify gradient-free optimization methods are discussed."""
        from part3_synthesis.section_3_1_comparative_analysis import get_optimizer_requirements

        requirements = get_optimizer_requirements()
        requirements_str = str(requirements).lower()

        assert 'gradient' in requirements_str or 'derivative' in requirements_str or 'cobyla' in requirements_str or 'nelder' in requirements_str


class TestVisualization:
    """Tests for comparative visualization functionality."""

    def test_create_comparison_visualization_exists(self):
        """Verify comparison visualization function exists."""
        from part3_synthesis.section_3_1_comparative_analysis import create_comparison_visualization

        assert callable(create_comparison_visualization)

    def test_visualization_creates_figure(self):
        """Verify visualization creates matplotlib figure."""
        from part3_synthesis.section_3_1_comparative_analysis import create_comparison_visualization

        fig = create_comparison_visualization()

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_visualization_has_multiple_subplots(self):
        """Verify visualization includes multiple comparison aspects."""
        from part3_synthesis.section_3_1_comparative_analysis import create_comparison_visualization

        fig = create_comparison_visualization()
        axes = fig.get_axes()

        # Should have at least 2 subplots for different comparison aspects
        assert len(axes) >= 2
        plt.close(fig)

    def test_visualization_includes_algorithm_names(self):
        """Verify visualization labels include algorithm names."""
        from part3_synthesis.section_3_1_comparative_analysis import create_comparison_visualization

        fig = create_comparison_visualization()

        # Convert all text in figure to string
        fig_text = []
        for ax in fig.get_axes():
            # Get all text objects
            for text in ax.texts:
                fig_text.append(text.get_text().upper())
            # Get title
            if ax.get_title():
                fig_text.append(ax.get_title().upper())
            # Get labels
            if ax.get_xlabel():
                fig_text.append(ax.get_xlabel().upper())
            if ax.get_ylabel():
                fig_text.append(ax.get_ylabel().upper())
            # Get legend
            legend = ax.get_legend()
            if legend:
                for text in legend.get_texts():
                    fig_text.append(text.get_text().upper())

        all_text = ' '.join(fig_text)

        # At least two of the three algorithms should appear
        alg_count = sum([
            'VQE' in all_text,
            'QAOA' in all_text,
            'QML' in all_text or 'QUANTUM' in all_text
        ])

        assert alg_count >= 2
        plt.close(fig)


class TestQuantumCircuitValidation:
    """Tests validating actual quantum circuits demonstrating algorithm patterns."""

    def test_minimal_vqe_circuit_produces_valid_statevector(self):
        """Verify minimal VQE example circuit produces normalized statevector."""
        from part3_synthesis.section_3_1_comparative_analysis import build_minimal_vqe_example

        circuit, hamiltonian = build_minimal_vqe_example()

        # Simulate to get statevector
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        statevector = result.final_state_vector

        # Verify normalization (quantum correctness)
        norm = np.sum(np.abs(statevector)**2)
        assert np.isclose(norm, 1.0, atol=1e-10), "Statevector must be normalized"

    def test_minimal_vqe_hamiltonian_is_hermitian(self):
        """Verify VQE example Hamiltonian is Hermitian (H = H†)."""
        from part3_synthesis.section_3_1_comparative_analysis import build_minimal_vqe_example

        circuit, hamiltonian = build_minimal_vqe_example()

        # Convert Hamiltonian to matrix
        n_qubits = len(circuit.all_qubits())
        hamiltonian_matrix = hamiltonian.matrix()

        # Verify Hermiticity: H = H†
        assert np.allclose(hamiltonian_matrix, hamiltonian_matrix.conj().T, atol=1e-10), \
            "Hamiltonian must be Hermitian"

    def test_minimal_vqe_expectation_value_is_real(self):
        """Verify VQE expectation value ⟨ψ|H|ψ⟩ is real for Hermitian H."""
        from part3_synthesis.section_3_1_comparative_analysis import build_minimal_vqe_example

        circuit, hamiltonian = build_minimal_vqe_example()

        # Simulate circuit
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        statevector = result.final_state_vector

        # Compute expectation value
        hamiltonian_matrix = hamiltonian.matrix()
        expectation = np.vdot(statevector, hamiltonian_matrix @ statevector)

        # Hermitian operators have real expectation values
        assert np.abs(expectation.imag) < 1e-10, \
            "Expectation value of Hermitian operator must be real"

    def test_minimal_qaoa_circuit_produces_valid_statevector(self):
        """Verify minimal QAOA example circuit produces normalized statevector."""
        from part3_synthesis.section_3_1_comparative_analysis import build_minimal_qaoa_example

        circuit, cost_hamiltonian = build_minimal_qaoa_example()

        # Simulate to get statevector
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        statevector = result.final_state_vector

        # Verify normalization
        norm = np.sum(np.abs(statevector)**2)
        assert np.isclose(norm, 1.0, atol=1e-10), "Statevector must be normalized"

    def test_minimal_qaoa_cost_hamiltonian_is_hermitian(self):
        """Verify QAOA cost Hamiltonian is Hermitian."""
        from part3_synthesis.section_3_1_comparative_analysis import build_minimal_qaoa_example

        circuit, cost_hamiltonian = build_minimal_qaoa_example()

        # Convert to matrix
        hamiltonian_matrix = cost_hamiltonian.matrix()

        # Verify Hermiticity
        assert np.allclose(hamiltonian_matrix, hamiltonian_matrix.conj().T, atol=1e-10), \
            "Cost Hamiltonian must be Hermitian"

    def test_minimal_qaoa_mixer_unitary_preserves_norm(self):
        """Verify QAOA mixer operation is unitary (U†U = I)."""
        from part3_synthesis.section_3_1_comparative_analysis import build_minimal_qaoa_example

        circuit, cost_hamiltonian = build_minimal_qaoa_example()

        # Get circuit unitary
        simulator = cirq.Simulator()
        n_qubits = len(circuit.all_qubits())
        unitary = cirq.unitary(circuit)

        # Verify unitarity: U†U = I
        identity = np.eye(2**n_qubits)
        product = unitary.conj().T @ unitary
        assert np.allclose(product, identity, atol=1e-10), \
            "QAOA circuit must be unitary"

    def test_minimal_qml_circuit_produces_valid_statevector(self):
        """Verify minimal QML example circuit produces normalized statevector."""
        from part3_synthesis.section_3_1_comparative_analysis import build_minimal_qml_example

        circuit, observables = build_minimal_qml_example()

        # Simulate to get statevector
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)
        statevector = result.final_state_vector

        # Verify normalization
        norm = np.sum(np.abs(statevector)**2)
        assert np.isclose(norm, 1.0, atol=1e-10), "Statevector must be normalized"

    def test_minimal_qml_observables_are_hermitian(self):
        """Verify QML measurement observables are Hermitian."""
        from part3_synthesis.section_3_1_comparative_analysis import build_minimal_qml_example

        circuit, observables = build_minimal_qml_example()

        # Check each observable
        for obs in observables:
            obs_matrix = obs.matrix()
            assert np.allclose(obs_matrix, obs_matrix.conj().T, atol=1e-10), \
                "Measurement observables must be Hermitian"

    def test_minimal_qml_circuit_is_unitary(self):
        """Verify QML parameterized circuit is unitary."""
        from part3_synthesis.section_3_1_comparative_analysis import build_minimal_qml_example

        circuit, observables = build_minimal_qml_example()

        # Get circuit unitary
        n_qubits = len(circuit.all_qubits())
        unitary = cirq.unitary(circuit)

        # Verify unitarity: U†U = I
        identity = np.eye(2**n_qubits)
        product = unitary.conj().T @ unitary
        assert np.allclose(product, identity, atol=1e-10), \
            "QML circuit must be unitary"


class TestIntegration:
    """Integration tests for the complete comparative analysis."""

    def test_main_function_executes(self, capsys):
        """Verify main analysis function executes without errors and captures output."""
        from part3_synthesis.section_3_1_comparative_analysis import main

        # Should not raise any exceptions
        main()

        # Capture and validate output
        captured = capsys.readouterr()
        assert len(captured.out) > 0, "Main should produce console output"
        assert "VQE" in captured.out or "QAOA" in captured.out, \
            "Output should mention algorithms"

    def test_analysis_produces_comprehensive_output(self):
        """Verify analysis produces all expected components."""
        from part3_synthesis.section_3_1_comparative_analysis import (
            get_unified_framework_structure,
            get_algorithm_comparison,
            identify_common_patterns,
            create_comparison_visualization
        )

        # All major functions should work together
        framework = get_unified_framework_structure()
        comparison = get_algorithm_comparison()
        patterns = identify_common_patterns()
        fig = create_comparison_visualization()

        assert framework is not None
        assert comparison is not None
        assert patterns is not None
        assert fig is not None

        plt.close(fig)

    def test_comparative_analysis_consistent(self):
        """Verify comparative analysis is internally consistent."""
        from part3_synthesis.section_3_1_comparative_analysis import (
            get_algorithm_comparison,
            identify_common_patterns
        )

        comparison = get_algorithm_comparison()
        patterns = identify_common_patterns()

        # All algorithms in comparison should share common patterns
        assert len(comparison) == 3
        assert len(patterns) > 0

        # Patterns should be reflected across all algorithms
        for alg_name in comparison:
            alg_info = comparison[alg_name]
            # Each algorithm should have the key characteristics
            assert 'goal' in alg_info or 'objective' in alg_info
