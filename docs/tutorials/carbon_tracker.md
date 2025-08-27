# Carbon Tracker

The CarbonTracker provides comprehensive carbon footprint monitoring for ML workloads with actionable sustainability insights.

## Overview

The Carbon Tracker provides:
- **Real Carbon Measurement**: CodeCarbon integration with regional carbon intensity
- **Cloud Optimization**: Automatic provider detection and regional recommendations  
- **Impact Visualization**: Intuitive comparisons (phone charges, car driving, tree absorption)
- **Business Reporting**: Corporate ESG reports and carbon budgeting
- **Actionable Insights**: Specific optimization recommendations with quantified savings

## Basic Usage

### Simple Carbon Tracking

```python
from cookbook.measure import CarbonTracker

tracker = CarbonTracker(
    project_name="sustainable-ml",
    experiment_name="carbon-demo"
)

with tracker.start_tracking("model_training"):
    # Your ML training code
    model.fit(X_train, y_train)

# Get carbon metrics
metrics = tracker.stop_tracking("model_training")
print(f"Emissions: {metrics.emissions_kg_co2 * 1000:.2f}g CO2eq")
print(f"Energy: {metrics.energy_consumed_kwh * 1000:.2f} Wh")
```

### Advanced Configuration

```python
tracker = CarbonTracker(
    project_name="enterprise-ml",
    experiment_name="production-training",
    output_dir="./carbon_reports",
    cloud_provider="gcp",           # Force specific provider
    cloud_region="us-west1",        # Low-carbon region
    country_iso_code="US",          # For accurate carbon intensity
    tracking_mode="machine"         # vs "offline" for air-gapped systems
)
```

## Cloud Provider Optimization

### Automatic Detection and Recommendations

```python
# CarbonTracker automatically detects cloud environment
tracker = CarbonTracker(project_name="cloud-optimization")

print(f"Detected provider: {tracker.cloud_provider}")
print(f"Current region: {tracker.cloud_region}")

# Get optimization recommendations
recommendations = tracker.get_regional_recommendations()
for rec in recommendations:
    print(f"💡 {rec}")

# Example output:
# 💡 Consider migrating to us-west1 for 40% lower carbon intensity
# 💡 Schedule training during off-peak hours for additional 15% reduction
```

### Regional Carbon Intensity Comparison

```python
from cookbook.measure import CarbonTracker

# Compare carbon impact across regions
regions = ["us-east1", "us-west1", "europe-north1", "asia-southeast1"]
carbon_comparison = {}

for region in regions:
    tracker = CarbonTracker(
        project_name="regional-analysis",
        cloud_provider="gcp",
        cloud_region=region
    )
    
    with tracker.start_tracking(f"training_in_{region}"):
        # Standardized workload
        simulate_training_workload()
    
    metrics = tracker.stop_tracking()
    carbon_comparison[region] = {
        "emissions_g": metrics.emissions_kg_co2 * 1000,
        "carbon_intensity": metrics.carbon_intensity_g_per_kwh
    }

# Find the most sustainable region
best_region = min(carbon_comparison.keys(), 
                 key=lambda r: carbon_comparison[r]["emissions_g"])
print(f"🌱 Most sustainable region: {best_region}")
```

## Comprehensive Impact Analysis

### Intuitive Comparisons

```python
tracker = CarbonTracker(project_name="impact-analysis")

with tracker.start_tracking("daily_ml_pipeline"):
    # Full ML pipeline
    data = preprocess_data()
    model = train_model(data)
    results = evaluate_model(model)

metrics = tracker.stop_tracking()
comparisons = metrics.emissions_comparison()

print("🌍 Environmental Impact:")
for comparison_type, description in comparisons.items():
    print(f"   {description}")

# Example output:
#   📱 2.3 smartphone charges
#   🚗 450 meters of car driving  
#   🌳 15.2 minutes of tree CO2 absorption
#   💡 4.1 hours of LED bulb usage
```

### Historical Tracking and Trends

```python
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

tracker = CarbonTracker(project_name="trend-analysis")

# Track emissions over time
daily_emissions = []
dates = []

for day in range(7):  # Week of training
    date = datetime.now() - timedelta(days=7-day)
    
    with tracker.start_tracking(f"daily_training_{date.strftime('%Y%m%d')}"):
        # Daily training routine
        run_daily_experiments()
    
    metrics = tracker.stop_tracking()
    daily_emissions.append(metrics.emissions_kg_co2 * 1000)
    dates.append(date)

# Visualize trends
plt.figure(figsize=(12, 6))
plt.plot(dates, daily_emissions, 'o-', linewidth=2, markersize=8)
plt.title('🌱 Daily Carbon Emissions Trend')
plt.xlabel('Date')
plt.ylabel('Emissions (g CO2eq)')
plt.grid(True, alpha=0.3)

# Calculate weekly total and average
weekly_total = sum(daily_emissions)
weekly_avg = weekly_total / 7

plt.axhline(y=weekly_avg, color='red', linestyle='--', 
           label=f'Weekly Average: {weekly_avg:.1f}g CO2eq')
plt.legend()
plt.show()

print(f"📊 Weekly carbon footprint: {weekly_total:.1f}g CO2eq")
```

## Carbon Budgeting

### Team Carbon Budget Management

```python
class TeamCarbonBudget:
    def __init__(self, team_name, monthly_budget_kg):
        self.team_name = team_name
        self.monthly_budget_kg = monthly_budget_kg
        self.current_usage_kg = 0
        self.tracker = CarbonTracker(project_name=f"{team_name}-budget")
    
    def track_experiment(self, experiment_name, experiment_func):
        if self.current_usage_kg >= self.monthly_budget_kg:
            raise Exception(f"⛔ Carbon budget exceeded! Used {self.current_usage_kg:.2f}/{self.monthly_budget_kg:.2f}kg CO2")
        
        with self.tracker.start_tracking(experiment_name):
            result = experiment_func()
        
        metrics = self.tracker.stop_tracking()
        self.current_usage_kg += metrics.emissions_kg_co2
        
        # Budget warnings
        usage_percent = (self.current_usage_kg / self.monthly_budget_kg) * 100
        if usage_percent > 80:
            print(f"⚠️ Carbon budget warning: {usage_percent:.1f}% used")
        
        return result
    
    def get_budget_status(self):
        remaining_kg = self.monthly_budget_kg - self.current_usage_kg
        remaining_percent = (remaining_kg / self.monthly_budget_kg) * 100
        
        return {
            "used_kg": self.current_usage_kg,
            "remaining_kg": remaining_kg,
            "remaining_percent": remaining_percent,
            "days_remaining_at_current_rate": self._estimate_days_remaining()
        }

# Usage
ml_team = TeamCarbonBudget("ml-research-team", monthly_budget_kg=5.0)

# Track team experiments
ml_team.track_experiment("baseline_model", lambda: train_baseline())
ml_team.track_experiment("optimized_model", lambda: train_optimized())

# Check budget status
status = ml_team.get_budget_status()
print(f"Budget status: {status['remaining_percent']:.1f}% remaining")
```

### Personal Developer Carbon Tracking

```python
import os
from datetime import datetime

class DeveloperCarbonTracker:
    def __init__(self):
        self.developer = os.getenv('USER', 'unknown')
        self.tracker = CarbonTracker(
            project_name=f"developer-{self.developer}",
            experiment_name=f"daily-work-{datetime.now().strftime('%Y%m%d')}"
        )
        self.daily_emissions = []
    
    def track_work_session(self, session_name):
        return self.tracker.start_tracking(f"{self.developer}_{session_name}")
    
    def end_work_day(self):
        # Generate daily carbon report
        report = self.tracker.get_summary_report()
        
        # Log to personal carbon journal
        self._log_daily_carbon(report)
        
        # Personal sustainability insights
        self._generate_personal_insights(report)
    
    def _generate_personal_insights(self, report):
        daily_emissions = report['summary']['total_emissions_kg_co2'] * 1000
        
        print(f"🌱 Your daily ML carbon footprint: {daily_emissions:.2f}g CO2")
        
        # Personal context
        monthly_estimate = daily_emissions * 22  # Working days
        annual_estimate = monthly_estimate * 12
        
        print(f"📅 Monthly estimate: {monthly_estimate:.1f}g CO2")
        print(f"📈 Annual estimate: {annual_estimate/1000:.2f}kg CO2")
        
        # Personal benchmarking
        sustainable_dev_target = 400  # kg CO2 per year (industry target)
        if annual_estimate/1000 < sustainable_dev_target:
            print(f"✅ Below sustainable developer target!")
        else:
            reduction_needed = (annual_estimate/1000 - sustainable_dev_target)
            print(f"⚠️ {reduction_needed:.1f}kg CO2/year above sustainable target")

# Usage
dev_tracker = DeveloperCarbonTracker()

# Track work sessions
with dev_tracker.track_work_session("morning_experiments"):
    run_hyperparameter_search()

with dev_tracker.track_work_session("model_evaluation"):
    evaluate_all_models()

# End of day summary
dev_tracker.end_work_day()
```

## Corporate ESG Reporting

### Automated Sustainability Reports

```python
from datetime import datetime, timedelta
import json

class CorporateCarbonReporter:
    def __init__(self, organization_name):
        self.organization = organization_name
        self.tracker = CarbonTracker(project_name=f"{organization_name}-esg")
    
    def generate_quarterly_report(self, team_trackers):
        """Generate comprehensive ESG sustainability report"""
        
        report_data = {
            "report_metadata": {
                "organization": self.organization,
                "report_period": "Q4 2025",
                "generated_date": datetime.now().isoformat(),
                "scope": "ML Engineering Carbon Footprint"
            },
            "executive_summary": self._calculate_executive_summary(team_trackers),
            "team_breakdown": self._analyze_team_breakdown(team_trackers),
            "sustainability_targets": self._assess_sustainability_targets(),
            "optimization_impact": self._calculate_optimization_impact(),
            "recommendations": self._generate_corporate_recommendations()
        }
        
        # Save report
        report_path = f"esg_carbon_report_q4_2025.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return report_data
    
    def _calculate_executive_summary(self, team_trackers):
        total_emissions = sum(tracker.get_total_emissions() for tracker in team_trackers)
        total_developers = len(team_trackers) * 5  # Assume 5 devs per team
        
        return {
            "total_emissions_tonnes": total_emissions / 1000,
            "emissions_per_developer_kg": total_emissions / total_developers,
            "equivalent_car_km": total_emissions / 0.0002,
            "trees_for_offset": total_emissions / 21.7,
            "carbon_budget_utilization_percent": 75.3  # Example
        }
    
    def _generate_corporate_recommendations(self):
        return [
            {
                "category": "Infrastructure Optimization",
                "priority": "High",
                "recommendation": "Migrate 80% of training workloads to low-carbon regions",
                "estimated_reduction": "30-50% emissions reduction",
                "timeline": "Q1 2026",
                "investment_required": "$50k migration costs"
            },
            {
                "category": "Process Optimization", 
                "priority": "Medium",
                "recommendation": "Implement carbon-aware job scheduling",
                "estimated_reduction": "10-25% emissions reduction",
                "timeline": "Q2 2026",
                "investment_required": "Engineering time: 2 person-months"
            },
            {
                "category": "Technology Adoption",
                "priority": "High", 
                "recommendation": "Deploy model compression techniques enterprise-wide",
                "estimated_reduction": "40-60% emissions reduction",
                "timeline": "Q3 2026",
                "investment_required": "Training and tooling: $75k"
            }
        ]

# Usage
esg_reporter = CorporateCarbonReporter("TechCorp AI Division")
quarterly_report = esg_reporter.generate_quarterly_report(team_trackers)

print(f"📊 Corporate carbon footprint: {quarterly_report['executive_summary']['total_emissions_tonnes']:.2f} tonnes CO2")
```

## Advanced Optimization Techniques

### Carbon-Aware Hyperparameter Optimization

```python
from cookbook.measure import CarbonTracker
import optuna

def carbon_aware_objective(trial):
    """Optimize for both accuracy and carbon efficiency"""
    
    # Suggest hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 128, step=16)
    epochs = trial.suggest_int('epochs', 5, 50)
    
    tracker = CarbonTracker(project_name="carbon-aware-hpo")
    
    with tracker.start_tracking(f"trial_{trial.number}"):
        # Train model with suggested parameters
        model = create_model()
        accuracy = train_model(model, lr=lr, batch_size=batch_size, epochs=epochs)
    
    carbon_metrics = tracker.stop_tracking()
    carbon_cost = carbon_metrics.emissions_kg_co2 * 1000  # Convert to grams
    
    # Multi-objective: maximize accuracy, minimize carbon
    # Carbon penalty: each gram of CO2 costs 0.01 accuracy points
    carbon_penalty = carbon_cost * 0.01
    carbon_aware_score = accuracy - carbon_penalty
    
    # Log for analysis
    trial.set_user_attr("accuracy", accuracy)
    trial.set_user_attr("carbon_g_co2", carbon_cost)
    trial.set_user_attr("carbon_penalty", carbon_penalty)
    
    return carbon_aware_score

# Run carbon-aware optimization
study = optuna.create_study(direction='maximize')
study.optimize(carbon_aware_objective, n_trials=50)

# Analyze results
best_trial = study.best_trial
print(f"🏆 Best carbon-aware hyperparameters:")
print(f"   Accuracy: {best_trial.user_attrs['accuracy']:.4f}")
print(f"   Carbon cost: {best_trial.user_attrs['carbon_g_co2']:.2f}g CO2")
print(f"   Parameters: {best_trial.params}")
```

### Sustainable Model Architecture Search

```python
def sustainable_architecture_search():
    """Compare model architectures for sustainability"""
    
    architectures = {
        "efficientnet_b0": {"params": 5.3e6, "flops": 0.39e9},
        "resnet50": {"params": 25.6e6, "flops": 4.1e9},
        "mobilenet_v2": {"params": 3.4e6, "flops": 0.3e9},
        "vision_transformer": {"params": 86e6, "flops": 17.6e9}
    }
    
    results = {}
    tracker = CarbonTracker(project_name="sustainable-architecture-search")
    
    for arch_name, arch_specs in architectures.items():
        print(f"🔍 Testing {arch_name}...")
        
        with tracker.start_tracking(f"architecture_{arch_name}"):
            # Simulate training with this architecture
            accuracy = simulate_architecture_training(arch_specs)
        
        carbon_metrics = tracker.stop_tracking()
        
        # Calculate sustainability metrics
        results[arch_name] = {
            "accuracy": accuracy,
            "carbon_g_co2": carbon_metrics.emissions_kg_co2 * 1000,
            "parameters_millions": arch_specs["params"] / 1e6,
            "flops_billions": arch_specs["flops"] / 1e9,
            "carbon_efficiency": accuracy / (carbon_metrics.emissions_kg_co2 * 1000),  # acc per gram CO2
            "parameter_efficiency": accuracy / (arch_specs["params"] / 1e6)  # acc per million params
        }
    
    # Rank by sustainability
    sustainability_ranking = sorted(results.items(), 
                                   key=lambda x: x[1]["carbon_efficiency"], 
                                   reverse=True)
    
    print(f"\n🌱 SUSTAINABILITY RANKING:")
    for i, (arch_name, metrics) in enumerate(sustainability_ranking, 1):
        print(f"{i}. {arch_name}")
        print(f"   Carbon Efficiency: {metrics['carbon_efficiency']:.4f} acc/g CO2")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Carbon Cost: {metrics['carbon_g_co2']:.2f}g CO2")
        print()
    
    return sustainability_ranking

# Find most sustainable architecture
best_sustainable_arch = sustainable_architecture_search()
print(f"🏆 Most sustainable architecture: {best_sustainable_arch[0][0]}")
```

## Integration with ML Pipelines

### Kubeflow Pipeline Integration

```python
from kfp import dsl
from cookbook.measure import CarbonTracker

@dsl.component
def carbon_aware_training_component(
    model_config: str,
    carbon_budget_g: float = 1000.0
) -> dict:
    """Kubeflow component with carbon tracking"""
    
    tracker = CarbonTracker(
        project_name="kubeflow-pipeline",
        experiment_name="carbon-aware-training"
    )
    
    with tracker.start_tracking("pipeline_training"):
        # Load model configuration
        config = json.loads(model_config)
        
        # Train model
        model = create_model(config)
        accuracy = train_model(model)
    
    carbon_metrics = tracker.stop_tracking()
    carbon_used_g = carbon_metrics.emissions_kg_co2 * 1000
    
    # Check carbon budget
    if carbon_used_g > carbon_budget_g:
        raise Exception(f"Carbon budget exceeded: {carbon_used_g:.2f}g > {carbon_budget_g:.2f}g")
    
    return {
        "accuracy": accuracy,
        "carbon_used_g": carbon_used_g,
        "carbon_remaining_g": carbon_budget_g - carbon_used_g,
        "model_path": save_model(model)
    }
```

## Best Practices

### 1. Establish Carbon Baselines

```python
def establish_carbon_baseline():
    """Create carbon baseline for future optimization"""
    
    tracker = CarbonTracker(project_name="baseline-establishment")
    baseline_metrics = {}
    
    # Standard workloads
    standard_workloads = [
        ("data_preprocessing", preprocess_data),
        ("model_training", train_baseline_model),
        ("model_evaluation", evaluate_model),
        ("hyperparameter_search", run_hp_search)
    ]
    
    for workload_name, workload_func in standard_workloads:
        print(f"📊 Establishing baseline for {workload_name}...")
        
        # Run workload multiple times for statistical significance
        emissions_samples = []
        for run in range(5):
            with tracker.start_tracking(f"{workload_name}_run_{run}"):
                workload_func()
            metrics = tracker.stop_tracking()
            emissions_samples.append(metrics.emissions_kg_co2 * 1000)
        
        baseline_metrics[workload_name] = {
            "mean_emissions_g": np.mean(emissions_samples),
            "std_emissions_g": np.std(emissions_samples),
            "samples": emissions_samples
        }
    
    # Save baseline for future comparison
    with open("carbon_baseline.json", "w") as f:
        json.dump(baseline_metrics, f, indent=2, default=str)
    
    return baseline_metrics
```

### 2. Continuous Carbon Monitoring

```python
class ContinuousCarbonMonitor:
    def __init__(self, baseline_path="carbon_baseline.json"):
        self.baseline = self.load_baseline(baseline_path)
        self.tracker = CarbonTracker(project_name="continuous-monitoring")
        
    def monitor_workload(self, workload_name, workload_func):
        """Monitor workload and alert on carbon regression"""
        
        with self.tracker.start_tracking(workload_name):
            result = workload_func()
        
        metrics = self.tracker.stop_tracking()
        current_emissions = metrics.emissions_kg_co2 * 1000
        
        # Compare to baseline
        if workload_name in self.baseline:
            baseline_mean = self.baseline[workload_name]["mean_emissions_g"]
            regression_threshold = baseline_mean * 1.2  # 20% increase
            
            if current_emissions > regression_threshold:
                self.alert_carbon_regression(workload_name, current_emissions, baseline_mean)
        
        return result
    
    def alert_carbon_regression(self, workload_name, current, baseline):
        increase_percent = ((current - baseline) / baseline) * 100
        print(f"🚨 CARBON REGRESSION ALERT:")
        print(f"   Workload: {workload_name}")
        print(f"   Current: {current:.2f}g CO2 (+{increase_percent:.1f}%)")
        print(f"   Baseline: {baseline:.2f}g CO2")
```

---

**Next**: Learn about [Statistical Validation](statistical_validator.md) for rigorous model comparison.
