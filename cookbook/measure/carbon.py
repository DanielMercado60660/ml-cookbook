# Project 1.1: The Measurement Suite - CodeCarbon Integration
# carbon footprint tracking for ML workloads
#nooodle noodle demo word for github


import os
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings

# CodeCarbon imports with graceful fallbacks
try:
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker
    from codecarbon.core.units import Energy, Power

    CODECARBON_AVAILABLE = True
    print("✅ CodeCarbon available - carbon tracking enabled")
except ImportError:
    CODECARBON_AVAILABLE = False
    print("⚠️  CodeCarbon not available - install with: pip install codecarbon")


    # Mock classes for development
    class EmissionsTracker:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            return 0.001  # Mock emissions in kg CO2

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass


    class OfflineEmissionsTracker:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def stop(self):
            return 0.001

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass


@dataclass
class CarbonMetrics:
    """Container for comprehensive carbon footprint metrics"""

    # Core emissions data
    emissions_kg_co2: float = 0.0
    energy_consumed_kwh: float = 0.0
    duration_seconds: float = 0.0

    # Hardware details
    cpu_energy_kwh: float = 0.0
    gpu_energy_kwh: float = 0.0
    ram_energy_kwh: float = 0.0

    # Geographic and grid data
    country_name: str = "Unknown"
    country_iso_code: str = "Unknown"
    region: str = "Unknown"
    cloud_provider: str = "Unknown"
    cloud_region: str = "Unknown"

    # Grid carbon intensity
    carbon_intensity_g_per_kwh: float = 0.0

    # Economic impact
    energy_cost_usd: Optional[float] = None
    carbon_offset_cost_usd: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    def emissions_comparison(self) -> Dict[str, str]:
        """Provide intuitive comparisons for carbon emissions"""
        kg_co2 = self.emissions_kg_co2

        if kg_co2 == 0:
            return {"comparison": "No emissions tracked"}

        # Various comparison metrics
        comparisons = {}

        # Car driving equivalent (average car: ~0.2 kg CO2/km)
        km_driving = kg_co2 / 0.2
        comparisons["car_driving_km"] = f"{km_driving:.2f} km of car driving"

        # Phone charging equivalent (~8.5g CO2 per charge)
        phone_charges = kg_co2 * 1000 / 8.5
        comparisons["phone_charges"] = f"{phone_charges:.1f} smartphone charges"

        # Tree absorption equivalent (~21.7 kg CO2/year per tree)
        tree_minutes = (kg_co2 / 21.7) * 365 * 24 * 60
        if tree_minutes < 60:
            comparisons["tree_absorption"] = f"{tree_minutes:.1f} minutes of tree CO2 absorption"
        elif tree_minutes < 1440:  # Less than a day
            comparisons["tree_absorption"] = f"{tree_minutes / 60:.1f} hours of tree CO2 absorption"
        else:
            comparisons["tree_absorption"] = f"{tree_minutes / 1440:.1f} days of tree CO2 absorption"

        # Energy equivalent comparisons
        if self.energy_consumed_kwh > 0:
            # LED bulb hours (10W LED = 0.01 kWh)
            led_hours = self.energy_consumed_kwh / 0.01
            comparisons["led_bulb_hours"] = f"{led_hours:.1f} hours of LED bulb usage"

            # Laptop hours (50W laptop = 0.05 kWh)
            laptop_hours = self.energy_consumed_kwh / 0.05
            comparisons["laptop_hours"] = f"{laptop_hours:.1f} hours of laptop usage"

        return comparisons


class CarbonTracker:
    """
    Advanced carbon footprint tracker for ML workloads

    Integrates with CodeCarbon for real measurements and provides:
    - Automatic hardware detection
    - Cloud provider optimization
    - Regional carbon intensity data
    - Economic impact estimates
    - Portfolio-quality reporting
    """

    def __init__(self,
                 project_name: str = "ml-cookbook",
                 experiment_name: str = "carbon-tracking",
                 output_dir: str = "./carbon_logs",
                 cloud_provider: Optional[str] = None,
                 cloud_region: Optional[str] = None,
                 country_iso_code: Optional[str] = None,
                 tracking_mode: str = "machine"):

        self.project_name = project_name
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.cloud_provider = cloud_provider
        self.cloud_region = cloud_region
        self.country_iso_code = country_iso_code
        self.tracking_mode = tracking_mode

        # State
        self.current_tracker = None
        self.metrics_history: List[CarbonMetrics] = []

        # Auto-detect cloud environment if not specified
        if not self.cloud_provider:
            self.cloud_provider, self.cloud_region = self._detect_cloud_environment()

        print(f"🌱 Carbon Tracker initialized")
        print(f"   Project: {project_name}")
        print(f"   Output: {output_dir}")
        print(f"   Cloud: {self.cloud_provider}/{self.cloud_region}")

    def _detect_cloud_environment(self) -> tuple:
        """Auto-detect cloud provider and region"""

        # Google Cloud Platform detection
        try:
            import requests
            metadata_server = "http://metadata.google.internal/computeMetadata/v1/"
            metadata_flavor = {"Metadata-Flavor": "Google"}

            # Check if we're on GCP
            response = requests.get(metadata_server, headers=metadata_flavor, timeout=1)
            if response.status_code == 200:
                # Get zone information
                zone_response = requests.get(
                    metadata_server + "instance/zone",
                    headers=metadata_flavor,
                    timeout=1
                )
                if zone_response.status_code == 200:
                    zone = zone_response.text.split("/")[-1]
                    region = "-".join(zone.split("-")[:-1])
                    return "gcp", region
        except:
            pass

        # AWS detection
        try:
            import requests
            response = requests.get(
                "http://169.254.169.254/latest/meta-data/placement/region",
                timeout=1
            )
            if response.status_code == 200:
                region = response.text
                return "aws", region
        except:
            pass

        # Azure detection
        try:
            import requests
            response = requests.get(
                "http://169.254.169.254/metadata/instance/compute/location?api-version=2020-06-01",
                headers={"Metadata": "true"},
                timeout=1
            )
            if response.status_code == 200:
                region = response.text
                return "azure", region
        except:
            pass

        # Default to unknown
        return "unknown", "unknown"

    def start_tracking(self,
                       task_name: str = "ml_task",
                       measure_power_secs: int = 15,
                       save_to_file: bool = True) -> 'CarbonTracker':
        """Start carbon tracking for a specific task"""

        if not CODECARBON_AVAILABLE:
            print("⚠️  Using mock carbon tracking (install codecarbon for real metrics)")

        # Configure tracker based on environment
        tracker_kwargs = {
            "project_name": self.project_name,
            "experiment_id": f"{self.experiment_name}_{task_name}",
            "output_dir": str(self.output_dir),
            "measure_power_secs": measure_power_secs,
            "save_to_file": save_to_file
        }

        # Add cloud-specific configuration
        if self.cloud_provider != "unknown":
            tracker_kwargs["cloud_provider"] = self.cloud_provider
            if self.cloud_region != "unknown":
                tracker_kwargs["cloud_region"] = self.cloud_region

        if self.country_iso_code:
            tracker_kwargs["country_iso_code"] = self.country_iso_code

        # Create appropriate tracker
        if CODECARBON_AVAILABLE:
            if self.tracking_mode == "offline":
                self.current_tracker = OfflineEmissionsTracker(**tracker_kwargs)
            else:
                self.current_tracker = EmissionsTracker(**tracker_kwargs)
        else:
            self.current_tracker = EmissionsTracker(**tracker_kwargs)

        # Start tracking
        self.current_tracker.start()
        self.start_time = time.time()

        print(f"🌱 Started carbon tracking: {task_name}")
        return self

    def stop_tracking(self, task_name: str = "ml_task") -> CarbonMetrics:
        """Stop carbon tracking and return comprehensive metrics"""

        if not self.current_tracker:
            raise RuntimeError("No active carbon tracking session")

        # Stop tracking and get emissions
        emissions_kg = self.current_tracker.stop()
        duration = time.time() - self.start_time

        print(f"🛑 Stopped carbon tracking: {task_name}")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Emissions: {emissions_kg * 1000:.2f}g CO2eq")

        # Create comprehensive metrics
        metrics = CarbonMetrics(
            emissions_kg_co2=emissions_kg,
            duration_seconds=duration,
            country_name=self._get_country_name(),
            cloud_provider=self.cloud_provider,
            cloud_region=self.cloud_region
        )

        # Try to extract detailed metrics from CodeCarbon data
        try:
            self._enrich_metrics_from_codecarbon_logs(metrics, task_name)
        except Exception as e:
            print(f"⚠️  Could not extract detailed metrics: {e}")

        # Add to history
        self.metrics_history.append(metrics)

        # Reset tracker
        self.current_tracker = None

        return metrics

    def _get_country_name(self) -> str:
        """Get country name from ISO code"""
        country_mapping = {
            "US": "United States",
            "GB": "United Kingdom",
            "DE": "Germany",
            "FR": "France",
            "CA": "Canada",
            "AU": "Australia",
            "JP": "Japan",
            "CN": "China",
            "IN": "India",
            "BR": "Brazil"
        }
        return country_mapping.get(self.country_iso_code, "Unknown")

    def _enrich_metrics_from_codecarbon_logs(self, metrics: CarbonMetrics, task_name: str):
        """Extract detailed metrics from CodeCarbon log files"""

        # Look for CodeCarbon CSV output
        csv_files = list(self.output_dir.glob("emissions.csv"))
        if not csv_files:
            return

        # Read the most recent entry
        import pandas as pd
        try:
            df = pd.read_csv(csv_files[0])
            if len(df) > 0:
                latest = df.iloc[-1]

                # Update metrics with detailed data
                metrics.energy_consumed_kwh = latest.get("energy_consumed", 0)
                metrics.cpu_energy_kwh = latest.get("cpu_energy", 0)
                metrics.gpu_energy_kwh = latest.get("gpu_energy", 0)
                metrics.ram_energy_kwh = latest.get("ram_energy", 0)
                metrics.country_name = latest.get("country_name", "Unknown")
                metrics.country_iso_code = latest.get("country_iso_code", "Unknown")
                metrics.region = latest.get("region", "Unknown")
                metrics.carbon_intensity_g_per_kwh = latest.get("carbon_intensity", 0)

        except Exception as e:
            print(f"⚠️  Could not parse CodeCarbon CSV: {e}")

    def __enter__(self):
        """Context manager entry"""
        return self.start_tracking()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_tracking()

    def get_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive carbon footprint report"""

        if not self.metrics_history:
            return {"error": "No tracking data available"}

        # Aggregate metrics
        total_emissions = sum(m.emissions_kg_co2 for m in self.metrics_history)
        total_energy = sum(m.energy_consumed_kwh for m in self.metrics_history)
        total_duration = sum(m.duration_seconds for m in self.metrics_history)

        latest_metrics = self.metrics_history[-1]
        comparisons = latest_metrics.emissions_comparison()

        report = {
            "summary": {
                "total_experiments": len(self.metrics_history),
                "total_emissions_kg_co2": total_emissions,
                "total_energy_kwh": total_energy,
                "total_duration_hours": total_duration / 3600,
                "average_emissions_per_experiment": total_emissions / len(self.metrics_history),
                "emissions_rate_g_per_hour": (total_emissions * 1000) / (total_duration / 3600)
            },
            "comparisons": comparisons,
            "hardware_breakdown": {
                "cpu_energy_percent": (latest_metrics.cpu_energy_kwh / max(latest_metrics.energy_consumed_kwh,
                                                                           0.001)) * 100,
                "gpu_energy_percent": (latest_metrics.gpu_energy_kwh / max(latest_metrics.energy_consumed_kwh,
                                                                           0.001)) * 100,
                "ram_energy_percent": (latest_metrics.ram_energy_kwh / max(latest_metrics.energy_consumed_kwh,
                                                                           0.001)) * 100
            },
            "location_info": {
                "country": latest_metrics.country_name,
                "region": latest_metrics.region,
                "cloud_provider": latest_metrics.cloud_provider,
                "cloud_region": latest_metrics.cloud_region,
                "carbon_intensity_g_per_kwh": latest_metrics.carbon_intensity_g_per_kwh
            },
            "recommendations": self._generate_recommendations()
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate sustainability recommendations"""

        recommendations = []

        if not self.metrics_history:
            return ["No data available for recommendations"]

        latest = self.metrics_history[-1]

        # GPU usage recommendations
        if latest.gpu_energy_kwh > latest.cpu_energy_kwh * 2:
            recommendations.append(
                "🖥️  High GPU usage detected - consider mixed precision training to reduce energy consumption")

        # Regional recommendations
        high_carbon_regions = ["us-east-1", "us-east-2", "ap-south-1"]
        if latest.cloud_region in high_carbon_regions:
            recommendations.append(
                "🌍 Consider switching to a lower-carbon cloud region (e.g., us-west-1, europe-north-1)")

        # Time-based recommendations
        if latest.duration_seconds > 3600:  # > 1 hour
            recommendations.append(
                "⏱️  Long training runs - consider checkpointing and spot instances to reduce waste from interruptions")

        # Energy efficiency
        if latest.emissions_kg_co2 > 0.1:  # > 100g CO2
            recommendations.append(
                "⚡ Significant emissions detected - consider model compression, early stopping, or efficient architectures")

        # Default recommendations
        if not recommendations:
            recommendations.extend([
                "✅ Good job on efficient ML! Continue monitoring carbon footprint",
                "💡 Consider carbon offsetting for production workloads",
                "📊 Track metrics over time to identify optimization opportunities"
            ])

        return recommendations

    def save_report(self, filename: Optional[str] = None) -> str:
        """Save comprehensive report to JSON file"""

        if filename is None:
            timestamp = int(time.time())
            filename = f"carbon_report_{timestamp}.json"

        report_path = self.output_dir / filename

        report_data = {
            "report": self.get_summary_report(),
            "raw_metrics": [m.to_dict() for m in self.metrics_history],
            "metadata": {
                "project_name": self.project_name,
                "experiment_name": self.experiment_name,
                "tracking_mode": self.tracking_mode,
                "codecarbon_available": CODECARBON_AVAILABLE,
                "generation_timestamp": time.time()
            }
        }

        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        print(f"💾 Carbon report saved: {report_path}")
        return str(report_path)


# Enhanced Performance Profiler with CodeCarbon Integration
class CarbonAwareProfiler:
    """
    Enhanced profiler that combines performance metrics with carbon tracking
    """

    def __init__(self,
                 project_name: str = "ml-cookbook",
                 experiment_name: str = "carbon-aware-profiling",
                 track_carbon: bool = True):

        # Import our existing profiler
        try:
            from .profiler import PerformanceProfiler
            self.performance_profiler = PerformanceProfiler(
                track_gpu=True,
                track_carbon=False  # We'll handle carbon tracking separately
            )
            self.profiler_available = True
        except ImportError:
            self.profiler_available = False
            print("⚠️  PerformanceProfiler not available")

        # Carbon tracker
        if track_carbon:
            self.carbon_tracker = CarbonTracker(
                project_name=project_name,
                experiment_name=experiment_name
            )
        else:
            self.carbon_tracker = None

    def profile_with_carbon(self, description: str = "ML Task"):
        """Context manager for combined performance and carbon profiling"""

        return CombinedProfilingContext(
            description=description,
            performance_profiler=self.performance_profiler if self.profiler_available else None,
            carbon_tracker=self.carbon_tracker
        )


class CombinedProfilingContext:
    """Context manager for combined performance and carbon profiling"""

    def __init__(self, description: str, performance_profiler, carbon_tracker):
        self.description = description
        self.performance_profiler = performance_profiler
        self.carbon_tracker = carbon_tracker
        self.results = {}

    def __enter__(self):
        # Start both trackers
        if self.performance_profiler:
            self.perf_context = self.performance_profiler.profile(self.description)
            self.perf_context.__enter__()

        if self.carbon_tracker:
            self.carbon_tracker.start_tracking(self.description)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop both trackers and collect results
        if self.performance_profiler:
            self.perf_context.__exit__(exc_type, exc_val, exc_tb)
            self.results['performance'] = self.performance_profiler.metrics.to_dict()

        if self.carbon_tracker:
            carbon_metrics = self.carbon_tracker.stop_tracking(self.description)
            self.results['carbon'] = carbon_metrics.to_dict()
            self.results['comparisons'] = carbon_metrics.emissions_comparison()

        # Print combined summary
        self._print_combined_summary()

    def _print_combined_summary(self):
        """Print combined performance and carbon summary"""

        print("\n" + "=" * 70)
        print("🔬 COMBINED PERFORMANCE & CARBON ANALYSIS")
        print("=" * 70)

        if 'performance' in self.results:
            perf = self.results['performance']
            print(f"🧠 Performance:")
            print(f"   Peak RAM: {perf['memory']['peak_ram_mb']:.1f} MB")
            print(f"   Wall Time: {perf['timing']['wall_time_s']:.3f}s")
            if perf['compute']['estimated_flops'] > 0:
                print(f"   Est. FLOPs: {perf['compute']['estimated_flops']:,}")

        if 'carbon' in self.results:
            carbon = self.results['carbon']
            print(f"🌱 Carbon Impact:")
            print(f"   Emissions: {carbon['emissions_kg_co2'] * 1000:.2f}g CO2eq")
            print(f"   Energy: {carbon['energy_consumed_kwh'] * 1000:.2f} Wh")

            if 'comparisons' in self.results:
                comp = self.results['comparisons']
                if 'phone_charges' in comp:
                    print(f"   Equivalent: {comp['phone_charges']}")

        print("=" * 70)


# Demo functions
def demo_carbon_tracking():
    """Demonstrate carbon tracking capabilities"""

    print("🧪 CARBON TRACKING DEMO")
    print("=" * 50)

    # Create carbon tracker
    tracker = CarbonTracker(
        project_name="ml-cookbook-demo",
        experiment_name="carbon_demo",
        output_dir="/content/cookbook/carbon_logs"
    )

    # Demo 1: Basic carbon tracking
    print("\n🌱 Demo 1: Basic carbon tracking")
    with tracker.start_tracking("demo_task_1"):
        # Simulate ML workload
        import numpy as np

        # Simulate training computation
        for i in range(5):
            # Matrix multiplication to use CPU/GPU
            a = np.random.rand(1000, 1000)
            b = np.random.rand(1000, 1000)
            c = np.dot(a, b)

            print(f"   Training step {i + 1}/5 completed")
            time.sleep(0.2)

    print("\n🌱 Demo 2: Multiple task tracking")

    # Demo 2: Track multiple tasks
    tasks = ["data_preprocessing", "model_training", "evaluation"]

    for task in tasks:
        with tracker.start_tracking(task):
            # Different compute patterns for different tasks
            if "preprocessing" in task:
                # Light CPU work
                time.sleep(0.1)
            elif "training" in task:
                # Heavy compute simulation
                time.sleep(0.3)
            else:
                # Evaluation
                time.sleep(0.1)

        print(f"   ✅ {task} completed")

    # Generate comprehensive report
    print("\n📊 Generating comprehensive carbon report...")
    report = tracker.get_summary_report()
    report_path = tracker.save_report()

    # Display key findings
    print("\n🎯 Key Sustainability Insights:")
    summary = report['summary']
    print(f"   Total Experiments: {summary['total_experiments']}")
    print(f"   Total Emissions: {summary['total_emissions_kg_co2'] * 1000:.2f}g CO2eq")
    print(f"   Emissions Rate: {summary['emissions_rate_g_per_hour']:.2f}g CO2/hour")

    print("\n💡 Sustainability Recommendations:")
    for rec in report['recommendations']:
        print(f"   {rec}")

    return tracker, report


def demo_carbon_aware_profiler():
    """Demonstrate combined carbon and performance profiling"""

    print("\n🧪 CARBON-AWARE PROFILER DEMO")
    print("=" * 50)

    # Create combined profiler
    profiler = CarbonAwareProfiler(
        project_name="ml-cookbook-demo",
        experiment_name="combined_profiling"
    )

    # Profile a simulated ML training loop
    with profiler.profile_with_carbon("ML Training Loop") as session:
        # Simulate realistic ML workload
        print("   🚀 Starting simulated training...")

        # Simulate data loading
        time.sleep(0.1)
        print("   📊 Data loading completed")

        # Simulate model forward/backward passes
        import numpy as np
        for epoch in range(3):
            # Forward pass simulation
            x = np.random.rand(32, 784)  # Batch of data
            weights = np.random.rand(784, 10)
            y = np.dot(x, weights)

            # Backward pass simulation
            grad = np.random.rand(784, 10) * 0.01
            weights -= grad

            print(f"   🔄 Epoch {epoch + 1}/3 completed")
            time.sleep(0.1)

        print("   ✅ Training completed")

    print("\n📈 Combined analysis completed!")

    return profiler, session.results


if __name__ == "__main__":
    # Run both demos
    carbon_tracker, carbon_report = demo_carbon_tracking()
    combined_profiler, combined_results = demo_carbon_aware_profiler()

    print("\n🎉 Carbon integration demo completed!")
    print(f"🌱 Real sustainability tracking now available in your ML Cookbook!")