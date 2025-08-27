# Project 1.1: Statistical Validator Validation Suite
# Rigorous testing of A/B testing framework functionality

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import json
from pathlib import Path


def validate_statistical_validator():
    """Comprehensive validation of statistical validator"""

    print("🔬 STATISTICAL VALIDATOR VALIDATION SUITE")
    print("=" * 60)

    results = {}

    # Test 1: Known effect detection
    print("🎯 Test 1: Known effect detection...")

    try:
        validator = StatisticalValidator(confidence_level=0.95, bootstrap_samples=1000)

        # Create data with known large effect (Cohen's d ≈ 1.0)
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 50)  # μ=0, σ=1
        treatment = np.random.normal(1, 1, 50)  # μ=1, σ=1, effect size ≈ 1.0

        result = validator.compare_models(
            baseline_metrics=baseline.tolist(),
            treatment_metrics=treatment.tolist(),
            metric_name="test_metric",
            test_type=TestType.BOOTSTRAP
        )

        # Validate results
        expected_effect_size = 1.0
        effect_size_error = abs(result.effect_size - expected_effect_size)

        tests_passed = {
            'significant_detection': result.is_significant(0.05),
            'effect_size_accuracy': effect_size_error < 0.3,  # Within 30%
            'ci_contains_true_diff': result.confidence_interval[0] <= 1.0 <= result.confidence_interval[1],
            'power_reasonable': result.power and result.power > 0.8
        }

        results['known_effect_detection'] = {
            'passed': all(tests_passed.values()),
            'effect_size_actual': result.effect_size,
            'effect_size_expected': expected_effect_size,
            'effect_size_error': effect_size_error,
            'p_value': result.p_value,
            'power': result.power,
            'individual_tests': tests_passed
        }

        if all(tests_passed.values()):
            print("   ✅ Known effect detection passed")
            print(f"      Effect size: {result.effect_size:.3f} (expected: ~{expected_effect_size:.1f})")
            print(f"      P-value: {result.p_value:.4f}")
        else:
            print("   ❌ Known effect detection failed")
            failed = [k for k, v in tests_passed.items() if not v]
            print(f"      Failed tests: {failed}")

    except Exception as e:
        results['known_effect_detection'] = {'passed': False, 'error': str(e)}
        print(f"   ❌ Known effect detection failed: {e}")

    # Test 2: No effect detection (null hypothesis)
    print("\n🎯 Test 2: Null hypothesis handling...")

    try:
        # Same distribution - should NOT be significant
        np.random.seed(123)
        baseline_null = np.random.normal(0, 1, 100)
        treatment_null = np.random.normal(0, 1, 100)  # Same distribution

        result_null = validator.compare_models(
            baseline_metrics=baseline_null.tolist(),
            treatment_metrics=treatment_null.tolist(),
            metric_name="null_test",
            test_type=TestType.BOOTSTRAP
        )

        null_tests_passed = {
            'not_significant': not result_null.is_significant(0.05),
            'small_effect_size': abs(result_null.effect_size) < 0.3,
            'ci_contains_zero': result_null.confidence_interval[0] <= 0 <= result_null.confidence_interval[1]
        }

        results['null_hypothesis_handling'] = {
            'passed': all(null_tests_passed.values()),
            'p_value': result_null.p_value,
            'effect_size': result_null.effect_size,
            'individual_tests': null_tests_passed
        }

        if all(null_tests_passed.values()):
            print("   ✅ Null hypothesis handling passed")
            print(f"      P-value: {result_null.p_value:.4f} (should be > 0.05)")
            print(f"      Effect size: {result_null.effect_size:.3f} (should be small)")
        else:
            print("   ❌ Null hypothesis handling failed")

    except Exception as e:
        results['null_hypothesis_handling'] = {'passed': False, 'error': str(e)}
        print(f"   ❌ Null hypothesis handling failed: {e}")

    # Test 3: Multiple test methods consistency
    print("\n🎯 Test 3: Test method consistency...")

    try:
        # Test with moderate effect
        np.random.seed(456)
        baseline_multi = np.random.normal(10, 2, 80)
        treatment_multi = np.random.normal(11.5, 2, 80)  # 0.75 effect size

        test_methods = [TestType.TTEST, TestType.BOOTSTRAP, TestType.MANN_WHITNEY]
        method_results = {}

        for method in test_methods:
            try:
                result = validator.compare_models(
                    baseline_metrics=baseline_multi.tolist(),
                    treatment_metrics=treatment_multi.tolist(),
                    metric_name="consistency_test",
                    test_type=method
                )

                method_results[method.value] = {
                    'p_value': result.p_value,
                    'effect_size': result.effect_size,
                    'significant': result.is_significant()
                }
            except Exception as e:
                method_results[method.value] = {'error': str(e)}

        # Check consistency (all should agree on significance)
        significances = [r.get('significant', False) for r in method_results.values() if 'significant' in r]
        consistent_significance = len(set(significances)) <= 1  # All same

        results['method_consistency'] = {
            'passed': consistent_significance and len(method_results) >= 2,
            'method_results': method_results,
            'consistent_significance': consistent_significance
        }

        if consistent_significance and len(method_results) >= 2:
            print("   ✅ Test method consistency passed")
            for method, result in method_results.items():
                if 'p_value' in result:
                    sig_str = "significant" if result['significant'] else "not significant"
                    print(f"      {method}: p={result['p_value']:.4f} ({sig_str})")
        else:
            print("   ❌ Test method consistency failed")

    except Exception as e:
        results['method_consistency'] = {'passed': False, 'error': str(e)}
        print(f"   ❌ Test method consistency failed: {e}")

    # Test 4: Sample size calculations
    print("\n🎯 Test 4: Sample size calculations...")

    try:
        effect_sizes_to_test = [0.2, 0.5, 0.8]  # Small, medium, large
        sample_size_results = {}

        for effect_size in effect_sizes_to_test:
            n1, n2 = validator.sample_size_calculation(effect_size=effect_size, power=0.8)

            # Validate reasonable range
            reasonable_range = 5 <= n1 <= 10000 and 5 <= n2 <= 10000
            larger_effect_smaller_n = True  # We'll check this across effect sizes

            sample_size_results[f"effect_{effect_size}"] = {
                'n1': n1,
                'n2': n2,
                'reasonable_range': reasonable_range
            }

        # Check inverse relationship: larger effect -> smaller sample size needed
        n_small = sample_size_results['effect_0.2']['n1']
        n_large = sample_size_results['effect_0.8']['n1']
        inverse_relationship = n_large < n_small

        results['sample_size_calculations'] = {
            'passed': all(r['reasonable_range'] for r in sample_size_results.values()) and inverse_relationship,
            'results': sample_size_results,
            'inverse_relationship': inverse_relationship
        }

        if results['sample_size_calculations']['passed']:
            print("   ✅ Sample size calculations passed")
            for effect, result in sample_size_results.items():
                print(f"      Effect {effect.split('_')[1]}: n1={result['n1']}, n2={result['n2']}")
        else:
            print("   ❌ Sample size calculations failed")

    except Exception as e:
        results['sample_size_calculations'] = {'passed': False, 'error': str(e)}
        print(f"   ❌ Sample size calculations failed: {e}")

    # Overall assessment
    print("\n" + "=" * 60)

    test_results = [
        results.get('known_effect_detection', {}).get('passed', False),
        results.get('null_hypothesis_handling', {}).get('passed', False),
        results.get('method_consistency', {}).get('passed', False),
        results.get('sample_size_calculations', {}).get('passed', False)
    ]

    overall_passed = all(test_results)
    passed_count = sum(test_results)

    results['overall_assessment'] = {
        'passed': overall_passed,
        'tests_passed': f"{passed_count}/4",
        'individual_results': test_results
    }

    print("🎯 OVERALL VALIDATION RESULT:")
    if overall_passed:
        print("   ✅ STATISTICAL VALIDATOR MEETS ALL QUALITY GATES")
        print("   Ready for rigorous A/B testing in production!")
    else:
        print(f"   ❌ STATISTICAL VALIDATOR PARTIAL SUCCESS ({passed_count}/4)")
        print("   Some tests need review before portfolio integration")

    return results


def create_validation_demo():
    """Create portfolio-quality demo of statistical validator"""

    print("\n" + "=" * 60)
    print("📊 CREATING VALIDATION DEMONSTRATION")
    print("=" * 60)

    # Run the main demo - use the fixed version
    validator, result, demo_fig = demo_fixed_validator()

    # Additional comparison: show different scenarios
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    scenarios = [
        {"name": "Small Effect", "baseline": (0.8, 0.05), "treatment": (0.82, 0.05), "n": 200},
        {"name": "Medium Effect", "baseline": (0.8, 0.05), "treatment": (0.85, 0.05), "n": 100},
        {"name": "Large Effect", "baseline": (0.8, 0.05), "treatment": (0.9, 0.05), "n": 50},
        {"name": "No Effect", "baseline": (0.8, 0.05), "treatment": (0.8, 0.05), "n": 100},
        {"name": "High Variance", "baseline": (0.8, 0.15), "treatment": (0.85, 0.15), "n": 200},
        {"name": "Small Sample", "baseline": (0.8, 0.05), "treatment": (0.9, 0.05), "n": 20}
    ]

    scenario_results = []

    for i, scenario in enumerate(scenarios):
        row, col = i // 3, i % 3
        ax = axes[row, col]

        # Generate data
        np.random.seed(42 + i)
        baseline_data = np.random.normal(scenario["baseline"][0], scenario["baseline"][1], scenario["n"])
        treatment_data = np.random.normal(scenario["treatment"][0], scenario["treatment"][1], scenario["n"])

        # Test
        result = validator.compare_models(
            baseline_metrics=baseline_data.tolist(),
            treatment_metrics=treatment_data.tolist(),
            metric_name=scenario["name"],
            test_type=TestType.BOOTSTRAP
        )

        scenario_results.append({
            'scenario': scenario["name"],
            'p_value': result.p_value,
            'effect_size': result.effect_size,
            'significant': result.is_significant(),
            'power': result.power
        })

        # Plot distributions
        ax.hist(baseline_data, alpha=0.6, label='Baseline', color='red', bins=15, density=True)
        ax.hist(treatment_data, alpha=0.6, label='Treatment', color='green', bins=15, density=True)
        ax.axvline(np.mean(baseline_data), color='red', linestyle='--', alpha=0.8)
        ax.axvline(np.mean(treatment_data), color='green', linestyle='--', alpha=0.8)

        # Add statistics text
        sig_text = "Significant" if result.is_significant() else "Not Significant"
        stats_text = f"{sig_text}\np={result.p_value:.4f}\nEffect: {result.effect_size:.3f}"

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
                                                   facecolor="white", alpha=0.8))

        ax.set_title(f"{scenario['name']} (n={scenario['n']})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('Statistical Validator - Multiple Scenarios', y=1.02, fontsize=16, fontweight='bold')
    plt.show()

    # Summary table
    print(f"\n📋 Scenario Summary:")
    print(f"{'Scenario':<15} {'P-value':<10} {'Effect Size':<12} {'Power':<8} {'Significant'}")
    print("-" * 60)
    for result in scenario_results:
        power_str = f"{result['power']:.3f}" if result['power'] else "N/A"
        sig_str = "Yes" if result['significant'] else "No"
        print(
            f"{result['scenario']:<15} {result['p_value']:<10.4f} {result['effect_size']:<12.3f} {power_str:<8} {sig_str}")

    return fig, scenario_results


def run_statistical_validator_validation():
    """Execute complete statistical validator validation"""

    # Run validation tests
    validation_results = validate_statistical_validator()

    # Create portfolio demo
    demo_fig, scenario_results = create_validation_demo()

    # Save validation report
    report_data = {
        'validation_results': validation_results,
        'scenario_results': scenario_results,
        'timestamp': int(time.time())
    }

    report_path = "/content/cookbook/logs/statistical_validator_validation.json"
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)

    print(f"\n💾 Statistical validator validation saved: {report_path}")

    return validation_results, scenario_results


# Execute validation
print("🚀 Starting Statistical Validator Validation...")
import time

stats_validation_results, demo_scenarios = run_statistical_validator_validation()