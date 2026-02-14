"""
Ransomware Negotiation Analysis Pipeline

A multi-model analysis pipeline for processing ransomware negotiation chat transcripts.
Implements task-first parallel processing with adaptive few-shot learning and automatic
text chunking for large conversations.

Key Features:
- Multi-model ensemble processing with parallel execution
- Adaptive few-shot example selection based on conversation length
- Automatic text chunking using recursive splitting for large chats
- Robust JSON validation and cleaning
- Comprehensive logging and statistics tracking
- Consensus analysis across multiple models

Author: Brilant Gashi
Institution: University of Brescia
Academic Year: 2024-2025
"""

import sys
import json
import yaml
import logging
import time
import re
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from collections import defaultdict
from threading import Lock
from io import StringIO
from contextlib import contextmanager
from typing import List, Dict, Any, Tuple, Optional
import argparse
from src.utils.sampling import (
    stratified_sample_chats,
    save_sample_manifest,
    load_sample_manifest,
    filter_db_by_sample
)

# Progress bar imports with graceful fallback
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): 
        """Fallback when tqdm is not installed."""
        return iterable


# === PATH CONFIGURATION ===
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR / "src"))


# === LOGGING CONFIGURATION ===
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


class TqdmLoggingHandler(logging.Handler):
    """
    Custom logging handler that integrates with tqdm progress bars.
    
    Prevents logging output from interfering with progress bar display
    by using tqdm.write() instead of standard print.
    """
    
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# File handler - captures all log levels
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(file_formatter)
root_logger.addHandler(file_handler)

# Console handler - only errors to avoid cluttering output
console_handler = TqdmLoggingHandler()
console_handler.setLevel(logging.ERROR)
console_formatter = logging.Formatter('ERROR: %(message)s')
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

logger = logging.getLogger("RansomPipeline")


# === STDOUT SUPPRESSION UTILITY ===
@contextmanager
def suppress_stdout():
    """
    Context manager to temporarily suppress stdout.
    
    Useful for preventing unwanted output during client initialization
    or other operations that should not interfere with progress bars.
    """
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


# === MODULE IMPORTS ===
try:
    from src.llm.unibs_client import LLMClient
    from src.utils.data_loader import download_and_load_messages_db, clean_message_list
    from src.analysis.consensus import ConsensusManager
except ImportError as e:
    logger.critical(f"Failed to import required modules: {e}")
    sys.exit(1)


# ============================================================================
# COMMAND LINE ARGUMENT PARSING
# ============================================================================


def parse_arguments():
    """
    Parse command line arguments for pipeline execution options.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Execute ransomware negotiation analysis pipeline'
    )
    
    parser.add_argument(
        '--use-sampling',
        action='store_true',
        help='Apply stratified sampling based on message count distribution'
    )
    
    parser.add_argument(
        '--use-existing-sample',
        action='store_true',
        help='Reuse existing sample_manifest.json if available'
    )
    
    parser.add_argument(
        '--new-sample',
        action='store_true',
        help='Force generation of new sample'
    )
    
    parser.add_argument(
        '--max-chats',
        type=int,
        default=None,
        help='Maximum number of chats to process'
    )
    
    parser.add_argument(
        '--sample-seed',
        type=int,
        default=42,
        help='Random seed for sampling reproducibility (default: 42)'
    )
    
    return parser.parse_args()


# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================


class ChunkConfig:
    """
    Configuration parameters for text chunking.
    
    These values control when and how large conversations are split
    into smaller chunks to fit within model token limits.
    
    Attributes:
        MAX_PROMPT_CHARS: Maximum characters allowed in a single prompt
        LONG_CHAT_THRESHOLD_MESSAGES: Threshold to reduce few-shot examples
    """
    
    MAX_PROMPT_CHARS = 5000
    """Maximum characters in prompt before triggering chunking."""
    
    LONG_CHAT_THRESHOLD_MESSAGES = 30
    """Number of messages above which few-shot examples are reduced."""


# ============================================================================
# PERFORMANCE TRACKING
# ============================================================================


class PerformanceTracker:
    """
    Tracks computational costs and execution metrics.
    
    Records execution time per chat stratified by message count,
    enabling cost analysis and performance optimization.
    """
    
    def __init__(self):
        """Initialize performance tracker with empty metrics."""
        self.metrics = []
        self._lock = Lock()
    
    def record_execution(
        self, 
        chat_id: str, 
        group: str, 
        message_count: int, 
        model: str,
        task: str,
        execution_time: float
    ):
        """
        Record execution metrics for a single chat processing operation.
        
        Args:
            chat_id: Unique identifier for the chat
            group: Ransomware group name
            message_count: Number of messages in conversation
            model: Model name used for processing
            task: Task name
            execution_time: Elapsed time in seconds
        """
        with self._lock:
            self.metrics.append({
                'chat_id': chat_id,
                'group': group,
                'message_count': message_count,
                'model': model,
                'task': task,
                'execution_time_seconds': execution_time,
                'timestamp': datetime.now().isoformat()
            })
    
    def save_report(self, output_path: Path):
        """
        Save performance report to JSON file.
        
        Args:
            output_path: Path where report will be saved
        """
        if not self.metrics:
            return
        
        # Stratify by message count bins
        bins = {'10-30': [], '30-60': [], '60-100': [], '100-150': [], '>150': []}
        
        for m in self.metrics:
            msg_count = m['message_count']
            exec_time = m['execution_time_seconds']
            
            if 10 <= msg_count <= 30:
                bins['10-30'].append(exec_time)
            elif 30 < msg_count <= 60:
                bins['30-60'].append(exec_time)
            elif 60 < msg_count <= 100:
                bins['60-100'].append(exec_time)
            elif 100 < msg_count <= 150:
                bins['100-150'].append(exec_time)
            else:
                bins['>150'].append(exec_time)
        
        # Generate report
        report = {
            'total_operations': len(self.metrics),
            'total_time_seconds': sum(m['execution_time_seconds'] for m in self.metrics),
            'by_length_bin': {}
        }
        
        for bin_name, times in bins.items():
            if times:
                report['by_length_bin'][bin_name] = {
                    'count': len(times),
                    'mean_seconds': sum(times) / len(times),
                    'min_seconds': min(times),
                    'max_seconds': max(times),
                    'total_seconds': sum(times)
                }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Performance report saved: {output_path}")


# ============================================================================
# STATISTICS TRACKER (Thread-Safe)
# ============================================================================


class PipelineStats:
    """
    Thread-safe statistics tracker for pipeline execution.
    
    Tracks various metrics including:
    - Chat processing completion rates
    - Chunk statistics for large conversations
    - Model performance and success rates
    - Warnings and errors encountered
    - Few-shot example usage
    
    All methods are thread-safe for use in parallel processing.
    """
    
    def __init__(self):
        """Initialize statistics tracker with default values."""
        self.start_time = datetime.now()
        self.total_chats = 0
        self.completed_chats = 0
        self.chunked_chats = 0
        self.total_chunks_processed = 0
        self.chat_warnings = defaultdict(list)
        self.chat_errors = defaultdict(list)
        self.model_stats = defaultdict(
            lambda: {'valid': 0, 'invalid': 0, 'tasks': 0}
        )
        self.few_shot_stats = defaultdict(int)
        self.chunk_stats = defaultdict(int)
        self.task_completion = defaultdict(int)
        self._lock = Lock()
    
    def add_warning(self, chat_id: str, message: str, model: Optional[str] = None):
        """
        Record a warning for a specific chat.
        
        Args:
            chat_id: Unique identifier for the chat
            message: Warning message
            model: Optional model name that generated the warning
        """
        with self._lock:
            self.chat_warnings[chat_id].append(message)
            logger.warning(f"{chat_id}: {message}")
            if model:
                self.model_stats[model]['invalid'] += 1
    
    def add_error(self, chat_id: str, error: str):
        """
        Record an error for a specific chat.
        
        Args:
            chat_id: Unique identifier for the chat
            error: Error message
        """
        with self._lock:
            self.chat_errors[chat_id].append(error)
            logger.error(f"{chat_id}: {error}")
    
    def add_success(self, model: str):
        """
        Record a successful operation for a model.
        
        Args:
            model: Model name
        """
        with self._lock:
            self.model_stats[model]['valid'] += 1
    
    def increment_task(self, model: str, task_name: Optional[str] = None):
        """
        Increment task counter for a model.
        
        Args:
            model: Model name
            task_name: Optional task name for granular tracking
        """
        with self._lock:
            self.model_stats[model]['tasks'] += 1
            if task_name:
                self.task_completion[task_name] += 1
    
    def add_few_shot_loaded(self, task_name: str, count: int):
        """
        Record few-shot examples loaded for a task.
        
        Args:
            task_name: Name of the task
            count: Number of examples loaded
        """
        with self._lock:
            self.few_shot_stats[task_name] = count
    
    def add_chunked_chat(self, chat_id: str, num_chunks: int):
        """
        Record that a chat was chunked.
        
        Args:
            chat_id: Unique identifier for the chat
            num_chunks: Number of chunks created
        """
        with self._lock:
            self.chunked_chats += 1
            self.total_chunks_processed += num_chunks
            self.chunk_stats[chat_id] = num_chunks
    
    def duration(self) -> str:
        """
        Calculate elapsed time since pipeline start.
        
        Returns:
            Formatted duration string (HH:MM:SS)
        """
        elapsed = datetime.now() - self.start_time
        return str(elapsed).split('.')[0]
    
    def print_summary(self):
        """
        Print comprehensive summary of pipeline execution statistics.
        
        Displays:
        - Overall completion metrics
        - Task completion breakdown
        - Few-shot example usage
        - Chunking statistics
        - Model performance metrics
        - Warnings and errors summary
        """
        duration = self.duration()
        
        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION COMPLETE")
        print("=" * 70)
        print(f"Duration: {duration}")
        tqdm.write()
        
        # Overall summary
        print("SUMMARY")
        print(f"  Total Chats:        {self.total_chats}")
        print(f"  Completed:          {self.completed_chats} "
              f"({self.completed_chats/self.total_chats*100:.1f}%)")
        print(f"  Chunked:            {self.chunked_chats} chats "
              f"-> {self.total_chunks_processed} chunks")
        print(f"  With Warnings:      {len(self.chat_warnings)}")
        print(f"  Errors:             {len(self.chat_errors)}")
        
        # Task completion
        if self.task_completion:
            tqdm.write()
            print("TASK COMPLETION")
            for task_name, count in sorted(self.task_completion.items()):
                print(f"  {task_name:30s}: {count:4d} completions")
        
        # Few-shot statistics
        if self.few_shot_stats:
            tqdm.write()
            print("FEW-SHOT EXAMPLES LOADED")
            for task_name, count in sorted(self.few_shot_stats.items()):
                print(f"  {task_name:25s}: {count} examples")
        
        # Top chunked chats
        if self.chunk_stats:
            tqdm.write()
            print("TOP CHUNKED CHATS")
            sorted_chunks = sorted(
                self.chunk_stats.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for chat_id, num_chunks in sorted_chunks[:5]:
                print(f"  {chat_id:25s}: {num_chunks} chunks")
        
        # Model performance
        if self.model_stats:
            tqdm.write()
            print("MODEL PERFORMANCE")
            for model, stats in sorted(self.model_stats.items()):
                total = stats['valid'] + stats['invalid']
                if total == 0:
                    continue
                success_rate = stats['valid'] / total * 100 if total > 0 else 0
                print(f"  {model:12s}: {stats['tasks']:3d} tasks, "
                      f"{stats['valid']:3d}/{total:3d} valid JSON "
                      f"({success_rate:5.1f}%)")
        
        # Warnings summary
        if self.chat_warnings:
            tqdm.write()
            print(f"WARNINGS ({len(self.chat_warnings)} chats)")
            for chat_id, warnings in sorted(self.chat_warnings.items())[:5]:
                print(f"  {chat_id}: {len(warnings)} issue(s)")
            if len(self.chat_warnings) > 5:
                print(f"  ... +{len(self.chat_warnings)-5} more (see logs)")
        
        # Errors summary
        if self.chat_errors:
            tqdm.write()
            print(f"ERRORS ({len(self.chat_errors)} chats)")
            for chat_id, errors in sorted(self.chat_errors.items())[:3]:
                print(f"  {chat_id}: {errors[0][:60]}")
            if len(self.chat_errors) > 3:
                print(f"  ... +{len(self.chat_errors)-3} more (see logs)")
        
        tqdm.write()
        print("=" * 70)
        print(f"Outputs: data/outputs/")
        print(f"Logs:    {LOG_FILE.name}")
        print("=" * 70 + "\n")


# ============================================================================
# RECURSIVE TEXT CHUNKING STRATEGY
# ============================================================================


class RecursiveChunker:
    """
    Implements recursive text chunking for large conversations.
    
    Uses a recursive 3-way splitting strategy to divide conversations
    that exceed the maximum prompt size. The algorithm:
    
    1. Renders the full prompt to check actual character count
    2. If under limit, returns as single chunk
    3. If over limit, splits into 3 roughly equal parts
    4. Recursively processes each part until all fit within limits
    
    This approach is simple, effective, and handles edge cases well.
    """
    
    @staticmethod
    def chunk_chat(
        dialogue: List[Dict[str, Any]], 
        system_prompt: str, 
        user_template: str
    ) -> List[List[Dict[str, Any]]]:
        """
        Split a conversation into chunks that fit within prompt limits.
        
        Args:
            dialogue: List of message dictionaries
            system_prompt: System prompt content
            user_template: User message template containing {{chat_json}}
        
        Returns:
            List of dialogue chunks, each guaranteed to fit in prompt
        """
        # Calculate actual rendered prompt size
        chat_json = json.dumps(dialogue, indent=2, ensure_ascii=False)
        user_msg = user_template.replace("{{chat_json}}", chat_json)
        combined_length = len(system_prompt) + len(user_msg)
        
        # Base case: fits in one chunk or cannot split further
        if combined_length <= ChunkConfig.MAX_PROMPT_CHARS or len(dialogue) <= 1:
            return [dialogue]
        
        # Recursive case: split into 3 parts
        third = max(1, len(dialogue) // 3)
        two_third = max(third + 1, 2 * len(dialogue) // 3)
        
        logger.debug(
            f"Splitting {len(dialogue)} messages into 3 chunks: "
            f"[0:{third}], [{third}:{two_third}], [{two_third}:{len(dialogue)}]"
        )
        
        # Recursively process each part
        return (
            RecursiveChunker.chunk_chat(
                dialogue[:third], system_prompt, user_template
            ) +
            RecursiveChunker.chunk_chat(
                dialogue[third:two_third], system_prompt, user_template
            ) +
            RecursiveChunker.chunk_chat(
                dialogue[two_third:], system_prompt, user_template
            )
        )
    
    @staticmethod
    def merge_chunk_results(chunk_results: List[str]) -> str:
        """
        Merge JSON results from multiple chunks.
        
        Parses each chunk's JSON output and concatenates arrays.
        Handles both array and object responses.
        
        Args:
            chunk_results: List of JSON strings from each chunk
        
        Returns:
            Merged JSON string
        """
        if not chunk_results:
            return "[]"
        
        if len(chunk_results) == 1:
            return chunk_results[0]
        
        # Parse and concatenate all chunks
        all_items = []
        for result in chunk_results:
            try:
                parsed = json.loads(result)
                if isinstance(parsed, list):
                    all_items.extend(parsed)
                elif isinstance(parsed, dict):
                    all_items.append(parsed)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse chunk result: {e}")
                continue
        
        return json.dumps(all_items, indent=2, ensure_ascii=False)


# ============================================================================
# PIPELINE BANNER
# ============================================================================


def print_banner(
    config: dict, 
    models: List[str], 
    max_chats: Optional[int], 
    num_tasks: int, 
    max_workers: int
):
    """
    Display pipeline configuration banner.
    
    Args:
        config: Pipeline configuration dictionary
        models: List of model names
        max_chats: Maximum number of chats to process (None = all)
        num_tasks: Number of tasks to execute
        max_workers: Maximum parallel workers
    """
    print()
    print("=" * 70)
    print("RANSOMWARE NEGOTIATION ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"Date:           {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Target Chats:   {max_chats if max_chats else 'All'}")
    print(f"Models:         {', '.join(models)}")
    print(f"Parallelism:    {max_workers}/{len(models)} concurrent models")
    print(f"Tasks:          {num_tasks} (processed sequentially)")
    print(f"Few-Shot:       Adaptive (reduced for long chats)")
    print(f"Chunking:       Recursive 3-way (max {ChunkConfig.MAX_PROMPT_CHARS:,} chars)")
    print(f"Strategy:       Task-first (all models on one task, then next)")
    print(f"Log File:       {LOG_FILE.name}")
    print("=" * 70)
    print()


# ============================================================================
# MAIN PIPELINE CLASS
# ============================================================================


class RansomwarePipeline:
    """
    Main pipeline orchestrator for ransomware negotiation analysis.
    
    Coordinates multi-model analysis across multiple tasks with:
    - Task-first parallel processing strategy
    - Dynamic worker allocation based on available resources
    - Adaptive few-shot learning
    - Automatic text chunking for large conversations
    - Robust error handling and logging
    - Consensus analysis across models
    
    The pipeline processes conversations through configurable tasks,
    with each task executed by multiple models in parallel. Worker threads
    are dynamically distributed among models to maximize throughput within
    API rate limits.
    """
    
    def __init__(self, args=None):
        """Initialize pipeline with directory structure and configuration."""
        self.args = args
        self.base_dir = BASE_DIR
        self.config_dir = self.base_dir / "config"
        self.output_dir = self.base_dir / "data" / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model configuration
        self.model_config_path = self.config_dir / "model_config.yaml"
        with open(self.model_config_path, 'r', encoding='utf-8') as f:
            self.model_config = yaml.safe_load(f)
        
        # Extract model list
        self.models_list = self.model_config.get(
            'ensemble_models', 
            [self.model_config.get('active_model')]
        )
        
        # Get max workers from config (default to number of models)
        self.max_workers = self.model_config.get('processing', {}).get(
            'max_workers', 
            len(self.models_list)
        )
        
        # Calculate dynamic worker allocation per model
        self._calculate_worker_allocation()
        
        # Initialize components
        self.consensus_manager = ConsensusManager(self.base_dir)
        self.prompts_config = {}
        self.full_dataset = {}
        self.stats = PipelineStats()
        self.performance_tracker = PerformanceTracker()
        self.few_shot_cache = {}
        self.chunker = RecursiveChunker()
        
        # Global API rate limiter (safety semaphore)
        self.api_semaphore = threading.Semaphore(self.max_workers)
    
    def _calculate_worker_allocation(self):
        """
        Dynamically calculate worker allocation per model.
        
        Distributes max_workers among models as evenly as possible.
        Extra workers are assigned to the first models in the list.
        """
        num_models = len(self.models_list)
        if num_models == 0:
            self.workers_per_model = {}
            return

        # Base allocation
        base_workers = self.max_workers // num_models
        extra_workers = self.max_workers % num_models
        
        # Build allocation map
        self.workers_per_model = {}
        for idx, model_name in enumerate(self.models_list):
            # First 'extra_workers' models get +1 worker
            workers = base_workers + (1 if idx < extra_workers else 0)
            self.workers_per_model[model_name] = max(1, workers)  # At least 1 worker
        
        logger.info(
            f"Dynamic worker allocation: {self.max_workers} total workers "
            f"distributed among {num_models} models"
        )
        for model, workers in self.workers_per_model.items():
            logger.info(f"  - {model}: {workers} worker(s)")

    def load_resources(self):
        """
        Load all required resources for pipeline execution.
        
        Loads:
        - Task prompt templates from separate YAML files
        - Dataset of ransomware negotiation chats
        - Model configurations
        
        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If dataset is empty or invalid
        """
        # Load task prompt templates
        task_files = {
            'Speech Act Analysis': 'prompt_speech_act_analysis.yaml',
            'Psychological Profiling': 'prompt_psychological_profiling.yaml',
            'Tactical Extraction': 'prompt_tactical_extraction.yaml'
        }
        
        self.prompts_config = {'tasks': {}}
        
        for task_name, filename in task_files.items():
            prompt_file_path = self.config_dir / filename
            
            try:
                with open(prompt_file_path, 'r', encoding='utf-8') as f:
                    task_data = yaml.safe_load(f)
                
                self.prompts_config['tasks'][task_name] = {
                    'output_format': task_data.get('output_format', 'json'),
                    'system_prompt': task_data.get('system_prompt', ''),
                    'user_template': task_data.get('user_prompt', '')
                }
                
                logger.info(f"Loaded prompt template: {task_name}")
            
            except FileNotFoundError:
                logger.error(f"Prompt file not found: {filename}")
                raise
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")
                raise
        
        logger.info(
            f"Loaded {len(self.prompts_config['tasks'])} task templates: "
            f"{list(self.prompts_config['tasks'].keys())}"
        )
        
        # Load dataset
        try:
            raw_rel_path = self.model_config['paths']['raw_data']
            data_path = self.base_dir / raw_rel_path
            logger.info(f"Loading dataset from {data_path}")
            
            self.full_dataset = download_and_load_messages_db(str(data_path))
            
            if not self.full_dataset:
                raise ValueError("Dataset is empty")
            
            logger.info(f"Dataset loaded. Groups: {len(self.full_dataset)}")
        
        except Exception as e:
            logger.critical(f"Failed to load dataset: {e}")
            raise
        
        # Apply sampling if requested
        self._apply_sampling()
    
    def _apply_sampling(self):
        """Apply stratified sampling to reduce dataset size."""
        if not self.args or not self.args.use_sampling:
            return
        
        manifest_path = self.base_dir / "data" / "sample_manifest.json"
        
        # Load or generate sample
        if (self.args.use_existing_sample and 
            manifest_path.exists() and 
            not self.args.new_sample):
            
            print("\nLoading existing sample manifest...")
            sampled_chats = load_sample_manifest(manifest_path)
        else:
            print("\nGenerating stratified sample...")
            sampled_chats = stratified_sample_chats(
                self.full_dataset,
                random_seed=self.args.sample_seed,
                verbose=True
            )
            save_sample_manifest(sampled_chats, manifest_path)
        
        # Filter dataset
        self.full_dataset = filter_db_by_sample(self.full_dataset, sampled_chats)
        
        sampled_count = sum(len(chats) for chats in self.full_dataset.values())
        logger.info(f"Dataset filtered to {sampled_count} sampled chats")
    
    def _load_few_shot_examples(
        self, 
        task_name: str, 
        max_examples: Optional[int] = None
    ) -> List[Dict]:
        """
        Load few-shot examples for a specific task.
        
        Implements caching to avoid repeated file reads. Supports
        limiting the number of examples for long conversations.
        
        Args:
            task_name: Name of the task
            max_examples: Maximum number of examples to return (None = all)
        
        Returns:
            List of few-shot example dictionaries
        """
        # Check cache first
        if task_name in self.few_shot_cache:
            examples = self.few_shot_cache[task_name]
        else:
            few_shot_path = (
                self.config_dir / "few_shot_examples" / f"{task_name}.json"
            )
            
            if not few_shot_path.exists():
                logger.warning(f"No few-shot file found: {few_shot_path}")
                self.few_shot_cache[task_name] = []
                return []
            
            try:
                with open(few_shot_path, 'r', encoding='utf-8') as f:
                    few_shot_data = json.load(f)
                
                examples = few_shot_data.get('examples', [])
                
                if not examples:
                    logger.warning(
                        f"No examples in few-shot file for {task_name}"
                    )
                    self.few_shot_cache[task_name] = []
                    return []
                
                logger.info(
                    f"Loaded {len(examples)} few-shot examples for {task_name}"
                )
                self.stats.add_few_shot_loaded(task_name, len(examples))
                
                # Cache the results
                self.few_shot_cache[task_name] = examples
            
            except Exception as e:
                logger.error(f"Failed to load few-shot for {task_name}: {e}")
                self.few_shot_cache[task_name] = []
                return []
        
        # Limit examples if requested
        if max_examples is not None and len(examples) > max_examples:
            return examples[:max_examples]
        
        return examples
    
    def _clean_json_output(self, text: Any) -> Optional[str]:
        """
        Robust JSON parser and cleaner.
        
        Attempts multiple strategies to extract valid JSON from model output:
        1. Direct parsing if already valid
        2. Extract JSON array from text
        3. Remove markdown code blocks and retry
        
        Args:
            text: Raw model output (string, list, or dict)
        
        Returns:
            Cleaned JSON string, or None if cannot be parsed
        """
        # Handle already-parsed JSON
        if isinstance(text, (list, dict)):
            return json.dumps(text, indent=2, ensure_ascii=False)
        
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        
        text = text.strip()
        
        # Try direct parsing
        try:
            parsed = json.loads(text)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON array
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return json.dumps(parsed, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                pass
        
        # Remove markdown code blocks and retry
        text = re.sub(r'```json\s*|\s*```', '', text)
        try:
            parsed = json.loads(text)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
        
        logger.debug(f"JSON unrepairable: {text[:200]}")
        return None
    
    def _build_messages_with_few_shot(
        self, 
        task_name: str, 
        task_cfg: Dict, 
        chat_json_str: str, 
        user_template: str,
        dialogue_length: int
    ) -> List[Dict]:
        """
        Build message array with adaptive few-shot examples.
        
        Automatically reduces few-shot examples for long conversations
        to stay within token limits. Few-shot examples are added as
        user/assistant pairs before the actual query.
        
        Args:
            task_name: Name of the task
            task_cfg: Task configuration dictionary
            chat_json_str: JSON string of the conversation
            user_template: Template for user message
            dialogue_length: Number of messages in conversation
        
        Returns:
            List of message dictionaries for the API
        """
        sys_msg = task_cfg['system_prompt']
        messages = [{"role": "system", "content": sys_msg}]
        
        # Adaptive few-shot loading based on conversation size
        if dialogue_length > ChunkConfig.LONG_CHAT_THRESHOLD_MESSAGES:
            max_few_shot = 1
            logger.debug(
                f"Long chat detected ({dialogue_length} msgs), "
                f"limiting few-shot to {max_few_shot}"
            )
        else:
            max_few_shot = None
        
        few_shot_examples = self._load_few_shot_examples(
            task_name, 
            max_examples=max_few_shot
        )
        
        # Add few-shot examples as user/assistant pairs
        for example in few_shot_examples:
            # Format input
            input_data = example.get('input', [])
            if isinstance(input_data, list):
                input_json = json.dumps(input_data, indent=2, ensure_ascii=False)
                user_example = user_template.replace("{{chat_json}}", input_json)
            else:
                user_example = str(input_data)
            
            # Format output
            output_data = example.get('output')
            if isinstance(output_data, (dict, list)):
                assistant_response = json.dumps(
                    output_data, indent=2, ensure_ascii=False
                )
            else:
                assistant_response = str(output_data)
            
            # Add to messages
            messages.append({"role": "user", "content": user_example})
            messages.append({"role": "assistant", "content": assistant_response})
        
        # Add actual query
        final_prompt = user_template.replace("{{chat_json}}", chat_json_str)
        messages.append({"role": "user", "content": final_prompt})
        
        logger.debug(
            f"Built message with {len(few_shot_examples)} "
            f"few-shot examples for {task_name}"
        )
        
        return messages
    
    def _process_single_chat(
        self, 
        client: LLMClient, 
        model_name: str,
        group_name: str, 
        chat_id: str, 
        dialogue: List[Dict],
        task_name: str, 
        task_cfg: Dict
    ) -> Tuple[bool, str]:
        """
        Process a single chat for a specific task with automatic chunking.
        
        Implements the complete processing pipeline for one chat:
        1. Clean and validate dialogue
        2. Determine if chunking is needed
        3. Split into chunks if necessary
        4. Process each chunk with LLM
        5. Merge results if chunked
        
        Args:
            client: LLM client instance
            model_name: Name of the model being used
            group_name: Group/category of the chat
            chat_id: Unique identifier for the chat
            dialogue: List of message dictionaries
            task_name: Name of the task being performed
            task_cfg: Task configuration dictionary
        
        Returns:
            Tuple of (success: bool, content: str)
        """
        start_time = time.time()
        
        # Clean dialogue
        dialogue = clean_message_list(dialogue)
        original_length = len(dialogue)
        
        # Get prompts for chunking decision
        system_prompt = task_cfg['system_prompt']
        user_template = task_cfg['user_template']
        
        # Apply recursive chunking strategy
        chunks = self.chunker.chunk_chat(dialogue, system_prompt, user_template)
        
        if len(chunks) > 1:
            logger.info(
                f"[{model_name}] Chat {chat_id} chunked: "
                f"{original_length} msgs -> {len(chunks)} chunks"
            )
            self.stats.add_chunked_chat(chat_id, len(chunks))
        
        # Process each chunk
        chunk_results = []
        
        for idx, chunk in enumerate(chunks):
            try:
                chunk_json_str = json.dumps(chunk, indent=2, ensure_ascii=False)
                
                # Build messages with adaptive few-shot
                messages = self._build_messages_with_few_shot(
                    task_name, task_cfg, chunk_json_str, 
                    user_template, len(chunk)
                )
                
                # Call LLM
                response_obj = client.generate_response(messages)
                
                if isinstance(response_obj, dict):
                    resp_text = response_obj.get('content', '')
                else:
                    resp_text = response_obj
                
                chunk_results.append(resp_text)
                
                if len(chunks) > 1:
                    logger.info(
                        f"[{model_name}] Completed {chat_id}/{task_name} "
                        f"chunk {idx+1}/{len(chunks)}"
                    )
                else:
                    logger.info(f"[{model_name}] Completed {chat_id}/{task_name}")
                
                time.sleep(0.5)  # Rate limiting
            
            except Exception as e:
                logger.error(
                    f"[{model_name}] Error processing chunk {idx+1}: {e}"
                )
                chunk_results.append("")  # Maintain order
        
        # Record performance metrics
        execution_time = time.time() - start_time
        self.performance_tracker.record_execution(
            chat_id, group_name, original_length, 
            model_name, task_name, execution_time
        )
        
        # Merge chunk results
        if len(chunks) > 1:
            merged_content = self.chunker.merge_chunk_results(chunk_results)
            return True, merged_content
        else:
            return True, chunk_results[0] if chunk_results else ""

    def _process_single_chat_wrapper(
        self,
        client: LLMClient,
        model_name: str,
        group_name: str,
        chat_id: str,
        dialogue: List[Dict],
        task_name: str,
        task_cfg: Dict,
        out_file: Path,
        pbar
    ):
        """
        Wrapper for processing a single chat with global rate limiting.
        Used by the internal thread pool executor.
        
        Args:
            client: LLM client instance
            model_name: Name of the model being used
            group_name: Group/category of the chat
            chat_id: Unique identifier for the chat
            dialogue: List of message dictionaries
            task_name: Name of the task being performed
            task_cfg: Task configuration dictionary
            out_file: Path where to save the output
            pbar: Progress bar instance
        """
        try:
            # Global rate limiting (prevents API overload across all models)
            with self.api_semaphore:
                # Process chat with automatic chunking
                success, content = self._process_single_chat(
                    client, model_name, group_name, chat_id, 
                    dialogue, task_name, task_cfg
                )
            
            if not success:
                pbar.update(1)
                return
            
            # Validate JSON if required
            if task_cfg.get('output_format') == 'json':
                cleaned_json = self._clean_json_output(content)
                
                if cleaned_json is None or cleaned_json == "[]":
                    warning = f"Invalid JSON in {task_name}"
                    self.stats.add_warning(chat_id, warning, model_name)
                    pbar.update(1)
                    return
                else:
                    content = cleaned_json
                    self.stats.add_success(model_name)
            
            # Write output
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            time.sleep(0.5)  # Rate limiting between calls
        
        except Exception as e:
            raise  # Re-raise to be caught by future.result()
        
        finally:
            pbar.update(1)

    def _process_chats_parallel(
        self,
        client: LLMClient,
        model_name: str,
        task_name: str,
        task_cfg: Dict,
        chat_queue: List,
        pbar,
        num_workers: int
    ):
        """
        Process multiple chats in parallel for a single model.
        
        Uses an internal ThreadPoolExecutor with dynamically allocated workers.
        
        Args:
            client: LLM client instance
            model_name: Name of the model being used
            task_name: Name of the task being performed
            task_cfg: Task configuration dictionary
            chat_queue: List of (group_name, chat_id, chat_content) tuples
            pbar: Progress bar instance
            num_workers: Number of workers allocated to this model
        """
        with ThreadPoolExecutor(max_workers=num_workers) as chat_executor:
            futures = []
            
            for group_name, chat_id, chat_content in chat_queue:
                dialogue = chat_content.get('dialogue', [])
                if not dialogue:
                    pbar.update(1)
                    continue
                
                self.stats.increment_task(model_name, task_name)
                
                # Prepare output directory
                task_out_dir = (
                    self.output_dir / task_name / model_name / group_name
                )
                task_out_dir.mkdir(parents=True, exist_ok=True)
                
                out_file = task_out_dir / (
                    f"{chat_id}.{task_cfg.get('output_format', 'txt')}"
                )
                
                # Skip if already processed
                if out_file.exists():
                    logger.debug(f"[{model_name}] Skip {chat_id}/{task_name}")
                    pbar.update(1)
                    continue
                
                # Submit chat processing to internal pool
                future = chat_executor.submit(
                    self._process_single_chat_wrapper,
                    client, model_name, group_name, chat_id,
                    dialogue, task_name, task_cfg, out_file, pbar
                )
                futures.append((future, chat_id))
            
            # Wait for all chats to complete
            for future, chat_id in futures:
                try:
                    future.result()
                except Exception as e:
                    error_msg = f"[{model_name}] {task_name}: {str(e)[:60]}"
                    self.stats.add_error(chat_id, error_msg)
                    logger.error(f"{chat_id}: {e}", exc_info=True)

    def _process_chats_sequential(
        self,
        client: LLMClient,
        model_name: str,
        task_name: str,
        task_cfg: Dict,
        chat_queue: List,
        pbar
    ):
        """
        Process chats sequentially (fallback when allocated only 1 worker).
        
        Kept for efficiency when parallelism is not beneficial.
        """
        for group_name, chat_id, chat_content in chat_queue:
            dialogue = chat_content.get('dialogue', [])
            if not dialogue:
                pbar.update(1)
                continue
            
            self.stats.increment_task(model_name, task_name)
            
            # Prepare output directory
            task_out_dir = (
                self.output_dir / task_name / model_name / group_name
            )
            task_out_dir.mkdir(parents=True, exist_ok=True)
            
            out_file = task_out_dir / (
                f"{chat_id}.{task_cfg.get('output_format', 'txt')}"
            )
            
            # Skip if already processed
            if out_file.exists():
                logger.debug(f"[{model_name}] Skip {chat_id}/{task_name}")
                pbar.update(1)
                continue
            
            try:
                # Global rate limiting
                with self.api_semaphore:
                    # Process chat with automatic chunking
                    success, content = self._process_single_chat(
                        client, model_name, group_name, chat_id, 
                        dialogue, task_name, task_cfg
                    )
                
                if not success:
                    pbar.update(1)
                    continue
                
                # Validate JSON if required
                if task_cfg.get('output_format') == 'json':
                    cleaned_json = self._clean_json_output(content)
                    
                    if cleaned_json is None or cleaned_json == "[]":
                        warning = f"Invalid JSON in {task_name}"
                        self.stats.add_warning(chat_id, warning, model_name)
                        pbar.update(1)
                        continue
                    else:
                        content = cleaned_json
                        self.stats.add_success(model_name)
                
                # Write output
                with open(out_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                time.sleep(0.5)  # Rate limiting
            
            except Exception as e:
                error_msg = f"[{model_name}] {task_name}: {str(e)[:60]}"
                self.stats.add_error(chat_id, error_msg)
                logger.error(f"{chat_id}: {e}", exc_info=True)
            
            pbar.update(1)

    def _process_model_for_task(
        self, 
        model_name: str, 
        task_name: str, 
        task_cfg: Dict, 
        chat_queue: List, 
        pbar
    ):
        """
        Process all chats for one task using one model.
        
        This is the core worker function called within the task loop.
        It manages the lifecycle of the LLM client and delegates
        chat processing to either parallel or sequential handlers.
        
        Args:
            model_name: Name of the model to use
            task_name: Name of the task to perform
            task_cfg: Task configuration dictionary
            chat_queue: List of (group_name, chat_id, chat_content) tuples
            pbar: Progress bar instance
        """
        # Initialize client with stdout suppressed
        with suppress_stdout():
            client = LLMClient(
                config_path=str(self.model_config_path), 
                model_override=model_name
            )
        
        # Get dynamic worker allocation for this model
        workers_for_model = self.workers_per_model.get(model_name, 1)
        
        logger.info(
            f"[{model_name}] Started task '{task_name}' "
            f"for {len(chat_queue)} chats with {workers_for_model} worker(s)"
        )
        
        # Process chats with dynamic parallelism
        if workers_for_model > 1:
            self._process_chats_parallel(
                client, model_name, task_name, task_cfg, 
                chat_queue, pbar, workers_for_model
            )
        else:
            # Fall back to sequential processing
            self._process_chats_sequential(
                client, model_name, task_name, task_cfg, 
                chat_queue, pbar
            )
        
        logger.info(f"[{model_name}] Completed task '{task_name}'")
    
    def run(self, max_chats: Optional[int] = None):
        """
        Execute the complete pipeline with task-first strategy.
        
        Processing flow:
        FOR each task:
            FOR each model (in parallel):
                Process all chats (using internal parallel pool)
        
        This ensures all models complete one task before moving to the next,
        facilitating immediate consensus analysis per task.
        
        Args:
            max_chats: Maximum number of chats to process (None = all)
        """
        tasks = self.prompts_config.get('tasks', {})
        
        # Build chat queue
        all_jobs = []
        for group_name, chats in self.full_dataset.items():
            for chat_id, chat_content in chats.items():
                all_jobs.append((group_name, chat_id, chat_content))
        
        # Limit if requested
        if max_chats and len(all_jobs) > max_chats:
            all_jobs = all_jobs[:max_chats]
        
        self.stats.total_chats = len(all_jobs)
        
        # Display configuration banner
        print_banner(
            self.prompts_config, 
            self.models_list, 
            max_chats, 
            len(tasks), 
            self.max_workers
        )
        
        # Calculate total operations for progress tracking
        total_operations = len(all_jobs) * len(self.models_list) * len(tasks)
        
        # Task-first processing loop
        with tqdm(
            total=total_operations, 
            desc="Processing", 
            unit="op", 
            bar_format="{desc}: {percentage:3.0f}%|{bar:25}| "
                      "{n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            colour="cyan", 
            ncols=120
        ) as pbar:
            
            for task_idx, (task_name, task_cfg) in enumerate(tasks.items(), 1):
                # Update progress bar description
                task_display = task_name[:30] 
                pbar.set_description(f"Task {task_idx}/{len(tasks)}: {task_display}")
                
                # Run all models in parallel for this task
                # Note: We use max_workers for the outer pool to ensure we have
                # enough threads to spawn the model handlers. The actual work
                # limiting is done by the internal pools and the global semaphore.
                outer_workers = len(self.models_list)
                
                with ThreadPoolExecutor(max_workers=outer_workers) as executor:
                    futures = []
                    
                    for model_name in self.models_list:
                        future = executor.submit(
                            self._process_model_for_task,
                            model_name, task_name, task_cfg, all_jobs, pbar
                        )
                        futures.append((future, model_name))
                    
                    # Wait for all models to complete this task
                    for future, model_name in futures:
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(
                                f"{model_name} failed on {task_name}: {e}", 
                                exc_info=True
                            )
                
                tqdm.write(f"Task '{task_name}' completed by all models")
        
        # Run consensus analysis if using multiple models
        if len(self.models_list) > 1:
            print("\nRunning consensus analysis...")
            for group_name, chat_id, _ in all_jobs:
                try:
                    self.consensus_manager.run_consensus_pipeline(
                        group_name, chat_id, self.models_list
                    )
                except Exception as e:
                    logger.error(f"Consensus error {chat_id}: {e}")
        
        # Finalize statistics
        self.stats.completed_chats = len(all_jobs)
        self.stats.print_summary()
        
        # Save performance report
        perf_report_path = self.base_dir / "data" / "performance_report.json"
        self.performance_tracker.save_report(perf_report_path)
        print(f"\nPerformance report saved: {perf_report_path.name}")



# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    """
    Main entry point for pipeline execution.
    
    Handles initialization, execution, and graceful shutdown with
    proper error handling and user interrupt support.
    """
    args = parse_arguments()
    pipeline = RansomwarePipeline(args)
    
    try:
        pipeline.load_resources()
        pipeline.run(max_chats=args.max_chats)
    
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(0)
    
    except Exception as e:
        print(f"\nFatal Error: {e}")
        logger.critical(e, exc_info=True)
        sys.exit(1)
