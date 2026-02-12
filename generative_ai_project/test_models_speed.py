"""
Temperature Grid Search for UniBS LLM Cluster
Tests all models with temperatures from 0.0 to 1.0 (step 0.1)
to find optimal settings for each model.

Features:
- Timeout: 25 seconds per test (skips slow models)
- Auto-save: Results saved after each test
- Resume capability: Can resume interrupted runs
- Quality metrics: Detects correct answers and gibberish

Author: Ransomware Analysis Project
Date: February 2026
"""

import os
import sys
import time
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from openai import OpenAI

# ═══════════════════════════════════════════════════════════════════
# LOAD ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════

try:
    from dotenv import load_dotenv
    BASE_DIR = Path(__file__).parent
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        print(f"✅ Loaded environment from: {env_path}")
except ImportError:
    print("⚠️  Warning: python-dotenv not installed")

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

BASE_URL = "https://gpustack.ing.unibs.it/v1"
API_KEY = os.environ.get("GPUSTACK_API_KEY")

if not API_KEY:
    print("❌ ERROR: GPUSTACK_API_KEY not found!")
    sys.exit(1)

# Models to test
MODELS = [
    "qwen3",
    "phi4-mini",
    "phi4",
    "llama3.2",
    "gpt-oss",
    "granite3.3",
    "gemma3"
]

# Temperature range: 0.0 to 1.0, step 0.1
TEMPERATURES = [round(t * 0.1, 1) for t in range(0, 11)]  # [0.0, 0.1, ..., 1.0]

# Timeout per test (seconds)
TIMEOUT_SECONDS = 25

# Test prompt (simple to evaluate quality easily)
TEST_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant. Answer briefly."},
    {"role": "user", "content": "What is 2+2? Answer with just the number."}
]

# Output directory
OUTPUT_DIR = BASE_DIR / "temperature_test_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    """Single test result for model + temperature combination."""
    model: str
    temperature: float
    success: bool
    response: Optional[str]
    time_seconds: Optional[float]
    error: Optional[str]
    is_correct: Optional[bool]  # Did it answer "4"?
    is_gibberish: Optional[bool]  # Is response nonsense?
    timed_out: bool = False  # Was test skipped due to timeout?
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════
# TEST CLASS
# ═══════════════════════════════════════════════════════════════════

class TemperatureGridSearch:
    """
    Systematic temperature testing for all LLM models.
    Tests each model with temperatures from 0.0 to 1.0.
    """
    
    def __init__(self):
        self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        self.results: List[TestResult] = []
        self.results_file = OUTPUT_DIR / "temperature_grid_results.json"
        self.csv_file = OUTPUT_DIR / "temperature_grid_results.csv"
        
        # Load previous results if exist (for resume capability)
        self.load_previous_results()
    
    def load_previous_results(self):
        """Load results from previous interrupted run."""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    data = json.load(f)
                    self.results = [TestResult(**r) for r in data]
                print(f"📂 Loaded {len(self.results)} previous results")
            except Exception as e:
                print(f"⚠️  Could not load previous results: {e}")
    
    def already_tested(self, model: str, temperature: float) -> bool:
        """Check if this combination was already tested."""
        return any(
            r.model == model and r.temperature == temperature
            for r in self.results
        )
    
    def test_single(self, model: str, temperature: float) -> TestResult:
        """
        Test a single model with specific temperature.
        TIMEOUT: 25 seconds max per test.
        
        Args:
            model: Model name
            temperature: Temperature value (0.0 to 1.0)
            
        Returns:
            TestResult object with timing and quality metrics
        """
        result = TestResult(
            model=model,
            temperature=temperature,
            success=False,
            response=None,
            time_seconds=None,
            error=None,
            is_correct=None,
            is_gibberish=None,
            timed_out=False
        )
        
        try:
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=model,
                messages=TEST_MESSAGES,
                temperature=temperature,
                top_p=0.95,
                max_tokens=100,  # Short answer expected
                frequency_penalty=0,
                presence_penalty=0,
                timeout=TIMEOUT_SECONDS,  # ← TIMEOUT: 25 secondi
            )
            
            elapsed = time.time() - start_time
            answer = response.choices[0].message.content.strip()
            
            result.success = True
            result.response = answer
            result.time_seconds = round(elapsed, 2)
            
            # Evaluate quality
            result.is_correct = self._is_correct_answer(answer)
            result.is_gibberish = self._is_gibberish(answer)
            
        except Exception as e:
            elapsed = time.time() - start_time
            result.time_seconds = round(elapsed, 2)
            
            # Check if it was a timeout
            error_str = str(e).lower()
            if "timeout" in error_str or "timed out" in error_str or elapsed >= (TIMEOUT_SECONDS - 0.5):
                result.error = f"TIMEOUT (>{elapsed:.1f}s)"
                result.timed_out = True
            else:
                result.error = str(e)[:200]
        
        return result
    
    @staticmethod
    def _is_correct_answer(response: str) -> bool:
        """Check if response is correct (contains '4')."""
        if not response:
            return False
        # Simple check: does it contain "4"?
        response_clean = response.strip().lower()
        return "4" in response and len(response) < 50
    
    @staticmethod
    def _is_gibberish(response: str) -> bool:
        """Detect if response is gibberish."""
        if not response or len(response) > 200:
            return True
        
        # Heuristics for gibberish
        gibberish_indicators = [
            "000000" in response,
            "111111" in response,
            response.count(">") > 10,
            response.count("{") > 10,
            "inscript" in response.lower(),
            "aboriter" in response.lower(),
            response.count("ier") > 10,
            response.count("\\") > 10,
        ]
        
        return any(gibberish_indicators)
    
    def run_grid_search(self):
        """
        Run complete grid search: all models × all temperatures.
        Total: 7 models × 11 temperatures = 77 tests
        """
        total_tests = len(MODELS) * len(TEMPERATURES)
        completed = len(self.results)
        
        print("\n" + "="*70)
        print("🔬 TEMPERATURE GRID SEARCH")
        print("="*70)
        print(f"📊 Models: {len(MODELS)}")
        print(f"🌡️  Temperatures: {TEMPERATURES}")
        print(f"⏱️  Timeout per test: {TIMEOUT_SECONDS}s")
        print(f"🧪 Total tests: {total_tests}")
        print(f"✅ Already completed: {completed}")
        print(f"⏳ Remaining: {total_tests - completed}")
        print(f"⏱️  Max estimated time: ~{(total_tests - completed) * (TIMEOUT_SECONDS + 1) / 60:.1f} minutes")
        print("="*70 + "\n")
        
        test_count = completed
        
        for model in MODELS:
            print(f"\n{'='*70}")
            print(f"🤖 Model: {model}")
            print(f"{'='*70}")
            
            for temp in TEMPERATURES:
                # Skip if already tested
                if self.already_tested(model, temp):
                    test_count += 1
                    print(f"  ⏭️  [{test_count}/{total_tests}] T={temp:.1f} - Already tested")
                    continue
                
                test_count += 1
                print(f"  🧪 [{test_count}/{total_tests}] T={temp:.1f}...", end=" ", flush=True)
                
                result = self.test_single(model, temp)
                self.results.append(result)
                
                # Print result
                if result.success:
                    quality = "✅" if result.is_correct else "❌"
                    gibberish = "🗑️" if result.is_gibberish else "📝"
                    print(f"{quality} {gibberish} {result.time_seconds:.2f}s")
                elif result.timed_out:
                    print(f"⏱️  TIMEOUT (>{result.time_seconds:.1f}s)")
                else:
                    error_preview = result.error[:40] if result.error else "Unknown"
                    print(f"❌ FAILED: {error_preview}")
                
                # Save after each test (in case of interruption)
                self.save_results()
                
                # Brief pause
                time.sleep(0.5)
        
        print("\n" + "="*70)
        print("✅ GRID SEARCH COMPLETE!")
        print("="*70 + "\n")
    
    def save_results(self):
        """Save results to JSON and CSV."""
        # JSON (full data)
        with open(self.results_file, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        
        # CSV (for Excel/analysis)
        with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=self.results[0].to_dict().keys())
                writer.writeheader()
                writer.writerows([r.to_dict() for r in self.results])
    
    def print_summary(self):
        """Print comprehensive summary with recommendations."""
        if not self.results:
            print("No results to summarize!")
            return
        
        print("\n" + "="*70)
        print("📊 SUMMARY - BEST TEMPERATURE PER MODEL")
        print("="*70 + "\n")
        
        # Group by model
        models_data = {}
        for result in self.results:
            if result.model not in models_data:
                models_data[result.model] = []
            models_data[result.model].append(result)
        
        # Find best temperature for each model
        print(f"{'Model':<15} | {'Best T':<8} | {'Time':<8} | {'Quality':<10} | {'Timeouts':<10}")
        print("-" * 70)
        
        best_configs = []
        
        for model, results in sorted(models_data.items()):
            # Filter successful and correct responses
            good_results = [
                r for r in results 
                if r.success and r.is_correct and not r.is_gibberish
            ]
            
            timeout_count = sum(1 for r in results if r.timed_out)
            
            if good_results:
                # Sort by time (fastest first)
                best = min(good_results, key=lambda r: r.time_seconds)
                quality_pct = len(good_results) / len(results) * 100
                
                print(f"{model:<15} | {best.temperature:<8.1f} | {best.time_seconds:<6.2f}s | {quality_pct:<8.0f}% | {timeout_count:<10}")
                
                best_configs.append({
                    "model": model,
                    "temperature": best.temperature,
                    "time": best.time_seconds,
                    "quality": quality_pct
                })
            else:
                print(f"{model:<15} | {'NONE':<8} | {'N/A':<8} | {'0%':<10} | {timeout_count:<10}")
        
        print("\n" + "="*70)
        print("💡 RECOMMENDED CONFIGURATION")
        print("="*70 + "\n")
        
        if best_configs:
            # Sort by time (fastest first)
            best_configs.sort(key=lambda x: x['time'])
            
            print("# ============================================================")
            print("# Paste this in your model_config.yaml:")
            print("# ============================================================\n")
            print("model_specific_params:")
            for cfg in best_configs:
                print(f"  {cfg['model']}:")
                print(f"    temperature: {cfg['temperature']}")
                print(f"    # Avg time: {cfg['time']:.2f}s | Quality: {cfg['quality']:.0f}%\n")
            
            print("\n# ============================================================")
            print("# Recommended ensemble (top 3 fastest with correct output):")
            print("# ============================================================\n")
            print("ensemble_models:")
            for cfg in best_configs[:3]:
                print(f"  - {cfg['model']:<15}  # T={cfg['temperature']:.1f}, {cfg['time']:.2f}s")
            
            # Show models to avoid
            slow_or_bad = [
                model for model in MODELS 
                if not any(c['model'] == model for c in best_configs)
            ]
            if slow_or_bad:
                print("\n# ⚠️  Models to avoid (too slow or poor quality):")
                for model in slow_or_bad:
                    print(f"  # - {model}")
        
        print("\n" + "="*70)
        print(f"📁 Full results saved to:")
        print(f"   - JSON: {self.results_file}")
        print(f"   - CSV:  {self.csv_file}")
        print("="*70 + "\n")
        
        # Statistics
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        correct = sum(1 for r in self.results if r.is_correct)
        timeouts = sum(1 for r in self.results if r.timed_out)
        
        print("📈 OVERALL STATISTICS:")
        print(f"   Total tests: {total}")
        print(f"   Successful: {successful} ({successful/total*100:.1f}%)")
        print(f"   Correct answers: {correct} ({correct/total*100:.1f}%)")
        print(f"   Timeouts: {timeouts} ({timeouts/total*100:.1f}%)")
        print("="*70 + "\n")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    """Main entry point."""
    try:
        searcher = TemperatureGridSearch()
        searcher.run_grid_search()
        searcher.print_summary()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user.")
        print("💾 Results saved. You can resume by running the script again.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
