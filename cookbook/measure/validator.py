# Project 1.1: The Measurement Suite - Statistical Validator
# A/B testing framework with bootstrap confidence intervals

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
import warnings


class TestType(Enum):
    """Types of statistical tests available"""
    TTEST = "t-test"
    BOOTSTRAP = "bootstrap"
    MANN_WHITNEY = "mann-whitney"
    PERMUTATION = "permutation"
    BAYESIAN = "bayesian"


class EffectSize(Enum):
    """Effect size measures"""
    COHENS_D = "cohen's d"
    HEDGES_G = "hedges' g"
    GLASS_DELTA = "glass' delta"


@dataclass
class StatisticalResult:
    """Container for statistical test results"""
    test_type: str
    p_value: float
    effect_size: float
    effect_size_type: str
    confidence_interval: Tuple[float, float]
    confidence_level: float

    # Additional metrics
    power: Optional[float] = None
    sample_size_a: int = 0
    sample_size_b: int = 0

    # Practical significance
    practically_significant: Optional[bool] = None
    minimum_detectable_effect: Optional[float] = None

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if result is statistically significant"""
        return self.p_value < alpha

    def interpretation(self) -> str:
        """Human-readable interpretation of results"""
        sig_status = "significant" if self.is_significant() else "not significant"

        # Interpret effect size (Cohen's conventions)
        if abs(self.effect_size) < 0.2:
            effect_mag = "negligible"
        elif abs(self.effect_size) < 0.5:
            effect_mag = "small"
        elif abs(self.effect_size) < 0.8:
            effect_mag = "medium"
        else:
            effect_mag = "large"

        direction = "positive" if self.effect_size > 0 else "negative"

        return (f"Result is {sig_status} (p={self.p_value:.4f}) with a "
                f"{effect_mag} {direction} effect size ({self.effect_size:.4f})")


class StatisticalValidator:
    """
    Production-quality A/B testing and statistical validation framework

    Provides rigorous statistical analysis for model comparison with:
    - Multiple testing procedures (t-test, bootstrap, non-parametric)
    - Effect size calculations with confidence intervals
    - Power analysis and sample size recommendations
    - Multiple comparison corrections
    - Practical significance assessment
    """

    def __init__(self,
                 confidence_level: float = 0.95,
                 bootstrap_samples: int = 10000,
                 random_seed: Optional[int] = 42):

        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.bootstrap_samples = bootstrap_samples

        if random_seed is not None:
            np.random.seed(random_seed)

        print(f"📊 Statistical Validator initialized")
        print(f"   Confidence level: {confidence_level * 100}%")
        print(f"   Bootstrap samples: {bootstrap_samples:,}")

    def compare_models(self,
                       baseline_metrics: List[float],
                       treatment_metrics: List[float],
                       metric_name: str = "metric",
                       test_type: TestType = TestType.BOOTSTRAP,
                       effect_size_type: EffectSize = EffectSize.COHENS_D,
                       practical_threshold: Optional[float] = None) -> StatisticalResult:
        """
        Compare two models using rigorous statistical testing

        Args:
            baseline_metrics: List of metric values for baseline model
            treatment_metrics: List of metric values for treatment model
            metric_name: Name of the metric being compared
            test_type: Type of statistical test to use
            effect_size_type: Type of effect size to calculate
            practical_threshold: Minimum effect size for practical significance
        """

        baseline = np.array(baseline_metrics)
        treatment = np.array(treatment_metrics)

        if len(baseline) == 0 or len(treatment) == 0:
            raise ValueError("Both groups must have at least one observation")

        print(f"🔬 Comparing models on {metric_name}")
        print(f"   Baseline: n={len(baseline)}, μ={np.mean(baseline):.4f}, σ={np.std(baseline):.4f}")
        print(f"   Treatment: n={len(treatment)}, μ={np.mean(treatment):.4f}, σ={np.std(treatment):.4f}")

        # Choose test method
        if test_type == TestType.TTEST:
            result = self._t_test(baseline, treatment, effect_size_type)
        elif test_type == TestType.BOOTSTRAP:
            result = self._bootstrap_test(baseline, treatment, effect_size_type)
        elif test_type == TestType.MANN_WHITNEY:
            result = self._mann_whitney_test(baseline, treatment, effect_size_type)
        elif test_type == TestType.PERMUTATION:
            result = self._permutation_test(baseline, treatment, effect_size_type)
        elif test_type == TestType.BAYESIAN:
            result = self._bayesian_test(baseline, treatment, effect_size_type)
        else:
            raise ValueError(f"Unknown test type: {test_type}")

        # Add sample sizes
        result.sample_size_a = len(baseline)
        result.sample_size_b = len(treatment)

        # Calculate power
        result.power = self._calculate_power(baseline, treatment, result.effect_size)

        # Check practical significance
        if practical_threshold is not None:
            result.practically_significant = abs(result.effect_size) >= practical_threshold
            result.minimum_detectable_effect = practical_threshold

        print(f"   {result.interpretation()}")
        if result.power:
            print(f"   Statistical power: {result.power:.3f}")

        return result

    def _t_test(self, baseline: np.ndarray, treatment: np.ndarray,
                effect_size_type: EffectSize) -> StatisticalResult:
        """Perform Welch's t-test (unequal variances)"""

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(treatment, baseline, equal_var=False)

        # Calculate effect size
        effect_size = self._calculate_effect_size(baseline, treatment, effect_size_type)

        # Calculate confidence interval for mean difference
        pooled_se = np.sqrt(np.var(baseline, ddof=1) / len(baseline) +
                            np.var(treatment, ddof=1) / len(treatment))

        # Degrees of freedom for Welch's t-test
        df = ((np.var(baseline, ddof=1) / len(baseline) + np.var(treatment, ddof=1) / len(treatment)) ** 2 /
              ((np.var(baseline, ddof=1) / len(baseline)) ** 2 / (len(baseline) - 1) +
               (np.var(treatment, ddof=1) / len(treatment)) ** 2 / (len(treatment) - 1)))

        t_critical = stats.t.ppf(1 - self.alpha / 2, df)
        mean_diff = np.mean(treatment) - np.mean(baseline)
        ci_lower = mean_diff - t_critical * pooled_se
        ci_upper = mean_diff + t_critical * pooled_se

        return StatisticalResult(
            test_type="t-test",
            p_value=p_value,
            effect_size=effect_size,
            effect_size_type=effect_size_type.value,
            confidence_interval=(ci_lower, ci_upper),
            confidence_level=self.confidence_level
        )

    def _bootstrap_test(self, baseline: np.ndarray, treatment: np.ndarray,
                        effect_size_type: EffectSize) -> StatisticalResult:
        """Perform bootstrap hypothesis test"""

        # Observed difference
        observed_diff = np.mean(treatment) - np.mean(baseline)

        # Bootstrap under null hypothesis (no difference)
        combined = np.concatenate([baseline, treatment])
        n_baseline, n_treatment = len(baseline), len(treatment)

        bootstrap_diffs = []
        for _ in range(self.bootstrap_samples):
            # Resample under null
            resampled = np.random.choice(combined, size=len(combined), replace=True)
            boot_baseline = resampled[:n_baseline]
            boot_treatment = resampled[n_baseline:n_baseline + n_treatment]

            bootstrap_diffs.append(np.mean(boot_treatment) - np.mean(boot_baseline))

        bootstrap_diffs = np.array(bootstrap_diffs)

        # Calculate p-value (two-tailed)
        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))

        # Bootstrap confidence interval for the difference
        ci_lower = np.percentile(bootstrap_diffs, 100 * self.alpha / 2)
        ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - self.alpha / 2))

        # Effect size
        effect_size = self._calculate_effect_size(baseline, treatment, effect_size_type)

        return StatisticalResult(
            test_type="bootstrap",
            p_value=p_value,
            effect_size=effect_size,
            effect_size_type=effect_size_type.value,
            confidence_interval=(ci_lower, ci_upper),
            confidence_level=self.confidence_level
        )

    def _mann_whitney_test(self, baseline: np.ndarray, treatment: np.ndarray,
                           effect_size_type: EffectSize) -> StatisticalResult:
        """Perform Mann-Whitney U test (non-parametric)"""

        u_stat, p_value = stats.mannwhitneyu(treatment, baseline, alternative='two-sided')

        # Effect size (rank-biserial correlation)
        n1, n2 = len(baseline), len(treatment)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)

        # Bootstrap CI for median difference (approximate)
        bootstrap_diffs = []
        for _ in range(1000):  # Fewer samples for speed
            boot_baseline = np.random.choice(baseline, len(baseline), replace=True)
            boot_treatment = np.random.choice(treatment, len(treatment), replace=True)
            bootstrap_diffs.append(np.median(boot_treatment) - np.median(boot_baseline))

        ci_lower = np.percentile(bootstrap_diffs, 100 * self.alpha / 2)
        ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - self.alpha / 2))

        return StatisticalResult(
            test_type="mann-whitney",
            p_value=p_value,
            effect_size=effect_size,
            effect_size_type="rank-biserial correlation",
            confidence_interval=(ci_lower, ci_upper),
            confidence_level=self.confidence_level
        )

    def _permutation_test(self, baseline: np.ndarray, treatment: np.ndarray,
                          effect_size_type: EffectSize) -> StatisticalResult:
        """Perform exact permutation test"""

        observed_diff = np.mean(treatment) - np.mean(baseline)
        combined = np.concatenate([baseline, treatment])
        n_treatment = len(treatment)

        # Generate permutations
        n_permutations = min(10000, 2 ** (len(combined)))  # Cap for efficiency
        permutation_diffs = []

        for _ in range(int(n_permutations)):
            permuted = np.random.permutation(combined)
            perm_treatment = permuted[:n_treatment]
            perm_baseline = permuted[n_treatment:]
            permutation_diffs.append(np.mean(perm_treatment) - np.mean(perm_baseline))

        permutation_diffs = np.array(permutation_diffs)

        # Two-tailed p-value
        p_value = np.mean(np.abs(permutation_diffs) >= np.abs(observed_diff))

        # Confidence interval
        ci_lower = np.percentile(permutation_diffs, 100 * self.alpha / 2)
        ci_upper = np.percentile(permutation_diffs, 100 * (1 - self.alpha / 2))

        effect_size = self._calculate_effect_size(baseline, treatment, effect_size_type)

        return StatisticalResult(
            test_type="permutation",
            p_value=p_value,
            effect_size=effect_size,
            effect_size_type=effect_size_type.value,
            confidence_interval=(ci_lower, ci_upper),
            confidence_level=self.confidence_level
        )

    def _bayesian_test(self, baseline: np.ndarray, treatment: np.ndarray,
                       effect_size_type: EffectSize) -> StatisticalResult:
        """Perform Bayesian hypothesis test (simplified version)"""
        
        # Use normal approximation for simplicity
        # In production, would use MCMC or analytical solutions
        mean_a, var_a = np.mean(baseline), np.var(baseline, ddof=1)
        mean_b, var_b = np.mean(treatment), np.var(treatment, ddof=1)
        
        # Posterior distribution of difference (normal approximation)
        post_mean = mean_b - mean_a
        post_var = var_a/len(baseline) + var_b/len(treatment)
        post_std = np.sqrt(post_var)
        
        # Probability that treatment > baseline (one-sided)
        prob_positive = 1 - stats.norm.cdf(0, post_mean, post_std)
        
        # For two-sided test, use the minimum of both tails
        p_value = 2 * min(prob_positive, 1 - prob_positive)
        
        # Calculate effect size
        effect_size = self._calculate_effect_size(baseline, treatment, effect_size_type)
        
        # Credible interval (Bayesian confidence interval)
        z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
        ci_lower = post_mean - z_score * post_std
        ci_upper = post_mean + z_score * post_std
        
        return StatisticalResult(
            test_type="bayesian",
            p_value=p_value,
            effect_size=effect_size,
            effect_size_type=effect_size_type.value,
            confidence_interval=(ci_lower, ci_upper),
            confidence_level=self.confidence_level
        )

    def _calculate_effect_size(self, baseline: np.ndarray, treatment: np.ndarray,
                               effect_size_type: EffectSize) -> float:
        """Calculate effect size"""

        mean_diff = np.mean(treatment) - np.mean(baseline)

        if effect_size_type == EffectSize.COHENS_D:
            pooled_std = np.sqrt(((len(baseline) - 1) * np.var(baseline, ddof=1) +
                                  (len(treatment) - 1) * np.var(treatment, ddof=1)) /
                                 (len(baseline) + len(treatment) - 2))
            return mean_diff / pooled_std if pooled_std > 0 else 0

        elif effect_size_type == EffectSize.HEDGES_G:
            cohens_d = self._calculate_effect_size(baseline, treatment, EffectSize.COHENS_D)
            correction = 1 - 3 / (4 * (len(baseline) + len(treatment)) - 9)
            return cohens_d * correction

        elif effect_size_type == EffectSize.GLASS_DELTA:
            baseline_std = np.std(baseline, ddof=1)
            return mean_diff / baseline_std if baseline_std > 0 else 0

        else:
            raise ValueError(f"Unknown effect size type: {effect_size_type}")

    def _calculate_power(self, baseline: np.ndarray, treatment: np.ndarray,
                         effect_size: float) -> float:
        """Calculate statistical power (approximate)"""

        try:
            from statsmodels.stats.power import ttest_power

            # Use harmonic mean for unequal sample sizes
            n_harmonic = 2 * len(baseline) * len(treatment) / (len(baseline) + len(treatment))

            power = ttest_power(
                effect_size=abs(effect_size),
                nobs=n_harmonic,  # Fixed: use 'nobs' instead of 'nobs1'/'nobs2'
                alpha=self.alpha
            )
            return power

        except (ImportError, TypeError):
            # Fallback approximation for power calculation
            n_harmonic = 2 * len(baseline) * len(treatment) / (len(baseline) + len(treatment))
            ncp = abs(effect_size) * np.sqrt(n_harmonic / 2)  # Non-centrality parameter

            # Approximate power using normal approximation
            critical_t = stats.t.ppf(1 - self.alpha / 2, len(baseline) + len(treatment) - 2)
            power = 1 - stats.norm.cdf(critical_t - ncp) + stats.norm.cdf(-critical_t - ncp)

            return max(0, min(1, power))

    def multiple_comparisons_correction(self,
                                        p_values: List[float],
                                        method: str = "holm") -> List[float]:
        """Apply multiple comparisons correction"""

        try:
            from statsmodels.stats.multitest import multipletests

            rejected, p_corrected, _, _ = multipletests(
                p_values, alpha=self.alpha, method=method
            )
            return p_corrected.tolist()

        except ImportError:
            # Simple Bonferroni correction as fallback
            warnings.warn("statsmodels not available, using Bonferroni correction")
            return [min(1.0, p * len(p_values)) for p in p_values]

    def sample_size_calculation(self,
                                effect_size: float,
                                power: float = 0.8,
                                ratio: float = 1.0) -> Tuple[int, int]:
        """Calculate required sample size for desired power"""

        try:
            from statsmodels.stats.power import ttest_power

            # Binary search for sample size
            n_low, n_high = 5, 10000

            while n_high - n_low > 1:
                n_mid = (n_low + n_high) // 2
                n1 = n_mid
                n2 = int(n_mid * ratio)

                # Use harmonic mean for power calculation
                n_harmonic = 2 * n1 * n2 / (n1 + n2)

                calculated_power = ttest_power(
                    effect_size=abs(effect_size),
                    nobs=n_harmonic,  # Fixed: use 'nobs' parameter
                    alpha=self.alpha
                )

                if calculated_power < power:
                    n_low = n_mid
                else:
                    n_high = n_mid

            n1 = n_high
            n2 = int(n_high * ratio)

            return n1, n2

        except (ImportError, TypeError):
            # Rough approximation using Cohen's formula
            z_alpha = stats.norm.ppf(1 - self.alpha / 2)
            z_beta = stats.norm.ppf(power)

            n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
            n1 = int(np.ceil(n / (1 + ratio)))
            n2 = int(np.ceil(n * ratio / (1 + ratio)))

            return n1, n2

    def visualize_comparison(self,
                             baseline_metrics: List[float],
                             treatment_metrics: List[float],
                             result: StatisticalResult,
                             metric_name: str = "Metric") -> plt.Figure:
        """Create portfolio-quality visualization of A/B test results"""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Distribution comparison
        ax1.hist(baseline_metrics, alpha=0.7, label='Baseline', bins=20, color='red', density=True)
        ax1.hist(treatment_metrics, alpha=0.7, label='Treatment', bins=20, color='green', density=True)
        ax1.axvline(np.mean(baseline_metrics), color='red', linestyle='--', alpha=0.8)
        ax1.axvline(np.mean(treatment_metrics), color='green', linestyle='--', alpha=0.8)
        ax1.set_title('Distribution Comparison')
        ax1.set_xlabel(metric_name)
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Box plot comparison
        data_to_plot = [baseline_metrics, treatment_metrics]
        labels = ['Baseline', 'Treatment']
        bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('red')
        bp['boxes'][1].set_facecolor('green')
        ax2.set_title('Box Plot Comparison')
        ax2.set_ylabel(metric_name)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Effect size with CI
        effect_sizes = [result.effect_size]
        ci_lower, ci_upper = result.confidence_interval
        ci_size = ci_upper - ci_lower

        ax3.bar(['Effect Size'], effect_sizes, yerr=[[result.effect_size - ci_lower], [ci_upper - result.effect_size]],
                capsize=10, color='blue', alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_title(f'Effect Size ({result.effect_size_type})')
        ax3.set_ylabel('Effect Size')
        ax3.grid(True, alpha=0.3)

        # Add effect size interpretation
        if abs(result.effect_size) < 0.2:
            effect_label = "Negligible"
        elif abs(result.effect_size) < 0.5:
            effect_label = "Small"
        elif abs(result.effect_size) < 0.8:
            effect_label = "Medium"
        else:
            effect_label = "Large"

        ax3.text(0, result.effect_size / 2, effect_label, ha='center', va='center',
                 fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Plot 4: Statistical summary
        ax4.axis('off')

        summary_text = f"""Statistical Test Results

Test Type: {result.test_type}
P-value: {result.p_value:.4f}
Effect Size: {result.effect_size:.4f} ({result.effect_size_type})
Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]
Confidence Level: {result.confidence_level * 100:.0f}%

Sample Sizes:
Baseline: n = {result.sample_size_a}
Treatment: n = {result.sample_size_b}

Statistical Power: {result.power:.3f if result.power else 'N/A'}

Interpretation:
{result.interpretation()}
"""

        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()
        plt.suptitle(f'Statistical Validation: {metric_name}', y=1.02, fontsize=16, fontweight='bold')

        return fig


# Demo and validation functions
def demo_statistical_validator():
    """Demo the statistical validator with synthetic A/B test"""

    print("🧪 STATISTICAL VALIDATOR DEMO")
    print("=" * 50)

    # Create validator
    validator = StatisticalValidator(confidence_level=0.95, bootstrap_samples=5000)

    # Generate synthetic A/B test data
    np.random.seed(42)

    # Baseline model (lower performance)
    baseline_accuracy = np.random.normal(0.85, 0.05, 100)  # 85% ± 5%

    # Treatment model (slightly better)
    treatment_accuracy = np.random.normal(0.88, 0.05, 90)  # 88% ± 5%

    # Run comparison
    result = validator.compare_models(
        baseline_metrics=baseline_accuracy.tolist(),
        treatment_metrics=treatment_accuracy.tolist(),
        metric_name="accuracy",
        test_type=TestType.BOOTSTRAP,
        effect_size_type=EffectSize.COHENS_D,
        practical_threshold=0.02  # 2% improvement threshold
    )

    # Create visualization
    fig = validator.visualize_comparison(
        baseline_accuracy.tolist(),
        treatment_accuracy.tolist(),
        result,
        "Model Accuracy"
    )

    plt.show()

    # Sample size recommendation
    n1, n2 = validator.sample_size_calculation(
        effect_size=result.effect_size,
        power=0.8
    )

    print(f"\n📊 Sample Size Recommendations:")
    print(f"   For 80% power to detect effect size {result.effect_size:.3f}:")
    print(f"   Baseline group: {n1} samples")
    print(f"   Treatment group: {n2} samples")

    return validator, result, fig


# Run demo
if __name__ == "__main__":
    validator, result, figure = demo_statistical_validator()