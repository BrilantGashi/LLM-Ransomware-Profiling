"""
Ransomware Negotiation Analysis Pipeline - v2.3.0 TASK-FIRST STRATEGY
All models process all chats with ONE prompt, then move to next prompt
Reads from 3 separate YAML files - CLEAN OUTPUT + FEW-SHOT + SMART CHUNKING
"""

import sys
import json
import yaml
import logging
import time
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from collections import defaultdict
from threading import Lock
from io import StringIO
from contextlib import contextmanager
from typing import List, Dict, Any, Tuple

# --- TQDM IMPORT ---
try:
    from tqdm import tqdm
except ImportError:
    print("⚠️  Warning: tqdm not installed.")
    def tqdm(iterable, **kwargs): 
        return iterable

# --- SETUP PATHS ---
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR / "src"))

# --- LOGGING SETUP ---
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

class TqdmLoggingHandler(logging.Handler):
    """Logging handler that plays nice with tqdm."""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)

# Configure Root Logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# File Handler - TUTTO nei log
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
root_logger.addHandler(file_handler)

# Console Handler - SOLO ERRORI
console_handler = TqdmLoggingHandler()
console_handler.setLevel(logging.ERROR)
console_formatter = logging.Formatter('❌ %(levelname)s: %(message)s')
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

logger = logging.getLogger("RansomPipeline")

# --- STDOUT SUPPRESSOR ---
@contextmanager
def suppress_stdout():
    """Temporarily suppress stdout to avoid tqdm interference."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout

# --- IMPORT MODULES ---
try:
    from src.llm.unibs_client import UniBSLLMClient
    from src.utils.data_loader import download_and_load_messages_db, clean_message_list
    from src.analysis.consensus import ConsensusManager
except ImportError as e:
    logger.critical(f"Error importing modules: {e}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════
# CHUNKING CONFIGURATION (Professor's Approach)
# ═══════════════════════════════════════════════════════════════════

class ChunkConfig:
    """Configuration for professor's chunking approach."""
    MAX_PROMPT_CHARS = 10_000  # Safe limit based on actual prompt size
    LONG_CHAT_THRESHOLD_MESSAGES = 50  # When to reduce few-shot examples

# ═══════════════════════════════════════════════════════════════════
# STATISTICS TRACKER (Thread-Safe)
# ═══════════════════════════════════════════════════════════════════

class PipelineStats:
    def __init__(self):
        self.start_time = datetime.now()
        self.total_chats = 0
        self.completed_chats = 0
        self.chunked_chats = 0
        self.total_chunks_processed = 0
        self.chat_warnings = defaultdict(list)
        self.chat_errors = defaultdict(list)
        self.model_stats = defaultdict(lambda: {'valid': 0, 'invalid': 0, 'tasks': 0})
        self.few_shot_stats = defaultdict(int)
        self.chunk_stats = defaultdict(int)
        self.task_completion = defaultdict(int)
        self._lock = Lock()
        
    def add_warning(self, chat_id: str, message: str, model: str = None):
        with self._lock:
            self.chat_warnings[chat_id].append(message)
            logger.warning(f"{chat_id}: {message}")
            if model:
                self.model_stats[model]['invalid'] += 1
    
    def add_error(self, chat_id: str, error: str):
        with self._lock:
            self.chat_errors[chat_id].append(error)
            logger.error(f"{chat_id}: {error}")
    
    def add_success(self, model: str):
        with self._lock:
            self.model_stats[model]['valid'] += 1
    
    def increment_task(self, model: str, task_name: str = None):
        with self._lock:
            self.model_stats[model]['tasks'] += 1
            if task_name:
                self.task_completion[task_name] += 1
    
    def add_few_shot_loaded(self, task_name: str, count: int):
        with self._lock:
            self.few_shot_stats[task_name] = count
    
    def add_chunked_chat(self, chat_id: str, num_chunks: int):
        with self._lock:
            self.chunked_chats += 1
            self.total_chunks_processed += num_chunks
            self.chunk_stats[chat_id] = num_chunks
    
    def duration(self):
        elapsed = datetime.now() - self.start_time
        return str(elapsed).split('.')[0]
    
    def print_summary(self):
        duration = self.duration()
        
        print("\n" + "━" * 70)
        print("✅ PIPELINE COMPLETE")
        print("━" * 70)
        print(f"⏱️  Duration: {duration}")
        print()
        
        print("📊 SUMMARY")
        print(f"  ├─ Total Chats:     {self.total_chats}")
        print(f"  ├─ ✅ Completed:     {self.completed_chats} ({self.completed_chats/self.total_chats*100:.1f}%)")
        print(f"  ├─ 🔪 Chunked:       {self.chunked_chats} chats → {self.total_chunks_processed} chunks")
        print(f"  ├─ ⚠️  With Warnings:  {len(self.chat_warnings)}")
        print(f"  └─ ❌ Errors:        {len(self.chat_errors)}")
        
        if self.task_completion:
            print()
            print("📋 TASK COMPLETION")
            for task_name, count in sorted(self.task_completion.items()):
                print(f"  ├─ {task_name:30s}: {count:4d} completions")
        
        if self.few_shot_stats:
            print()
            print("📚 FEW-SHOT EXAMPLES LOADED")
            for task_name, count in sorted(self.few_shot_stats.items()):
                print(f"  ├─ {task_name:25s}: {count} examples")
        
        if self.chunk_stats:
            print()
            print(f"🔪 TOP CHUNKED CHATS")
            for chat_id, num_chunks in sorted(self.chunk_stats.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  ├─ {chat_id:25s}: {num_chunks} chunks")
        
        if self.model_stats:
            print()
            print("🤖 MODEL PERFORMANCE")
            for model, stats in sorted(self.model_stats.items()):
                total = stats['valid'] + stats['invalid']
                if total == 0:
                    continue
                success_rate = stats['valid'] / total * 100 if total > 0 else 0
                star = " ⭐" if success_rate >= 95 else ""
                print(f"  ├─ {model:12s}: {stats['tasks']:3d} tasks, {stats['valid']:3d}/{total:3d} valid JSON ({success_rate:5.1f}%){star}")
        
        if self.chat_warnings:
            print()
            print(f"⚠️  WARNINGS ({len(self.chat_warnings)} chats)")
            for chat_id, warnings in sorted(self.chat_warnings.items())[:5]:
                print(f"  ├─ {chat_id}: {len(warnings)} issue{'s' if len(warnings) > 1 else ''}")
            if len(self.chat_warnings) > 5:
                print(f"  └─ ... +{len(self.chat_warnings)-5} more (see logs)")
        
        if self.chat_errors:
            print()
            print(f"❌ ERRORS ({len(self.chat_errors)} chats)")
            for chat_id, errors in sorted(self.chat_errors.items())[:3]:
                print(f"  ├─ {chat_id}: {errors[0][:60]}")
            if len(self.chat_errors) > 3:
                print(f"  └─ ... +{len(self.chat_errors)-3} more (see logs)")
        
        print()
        print("━" * 70)
        print(f"📁 Outputs: data/outputs/")
        print(f"📝 Logs:    {LOG_FILE.name}")
        print("━" * 70 + "\n")

# ═══════════════════════════════════════════════════════════════════
# PROFESSOR'S CHUNKING STRATEGY
# ═══════════════════════════════════════════════════════════════════

class ProfessorChunker:
    """
    Professor's recursive 3-way chunking approach.
    Splits chat until the ACTUAL rendered prompt fits under MAX_PROMPT_CHARS.
    Simple, effective, and battle-tested.
    """
    
    @staticmethod
    def chunk_chat(
        dialogue: List[Dict[str, Any]], 
        system_prompt: str, 
        user_template: str
    ) -> List[List[Dict[str, Any]]]:
        """
        Split a chat into smaller chunks until the rendered prompt fits.
        Uses recursive 3-way split (crude but effective).
        
        Args:
            dialogue: List of message dictionaries
            system_prompt: System message content
            user_template: User message template (contains {{chat_json}})
        
        Returns:
            List of dialogue chunks
        """
        # Render the full prompt to check size
        chat_json = json.dumps(dialogue, ensure_ascii=False)
        user_msg = user_template.replace("{{chat_json}}", chat_json)
        combined_length = len(system_prompt) + len(user_msg)
        
        # Base case: fits in one chunk or can't split further
        if combined_length <= ChunkConfig.MAX_PROMPT_CHARS or len(dialogue) <= 1:
            return [dialogue]
        
        # Recursive case: split into 3 parts
        third = max(1, len(dialogue) // 3)
        two_third = max(third + 1, 2 * len(dialogue) // 3)
        
        logger.debug(f"Splitting {len(dialogue)} messages into 3 chunks: "
                    f"[0:{third}], [{third}:{two_third}], [{two_third}:{len(dialogue)}]")
        
        return (
            ProfessorChunker.chunk_chat(dialogue[:third], system_prompt, user_template) +
            ProfessorChunker.chunk_chat(dialogue[third:two_third], system_prompt, user_template) +
            ProfessorChunker.chunk_chat(dialogue[two_third:], system_prompt, user_template)
        )
    
    @staticmethod
    def merge_chunk_results(chunk_results: List[str]) -> str:
        """
        Merge JSON arrays from multiple chunks.
        Simple concatenation - works for speech act labeling.
        
        Args:
            chunk_results: List of JSON strings from each chunk
        
        Returns:
            Merged JSON string
        """
        if not chunk_results:
            return "[]"
        
        if len(chunk_results) == 1:
            return chunk_results[0]
        
        # Parse all chunks and concatenate
        all_items = []
        for result in chunk_results:
            try:
                parsed = json.loads(result)
                if isinstance(parsed, list):
                    all_items.extend(parsed)
                elif isinstance(parsed, dict):
                    # If individual chunks return dicts, collect them
                    all_items.append(parsed)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse chunk result: {e}")
                continue
        
        return json.dumps(all_items, indent=2, ensure_ascii=False)

# ═══════════════════════════════════════════════════════════════════
# BANNER
# ═══════════════════════════════════════════════════════════════════

def print_banner(config, models, max_chats, num_tasks, max_workers):
    print()
    print("━" * 70)
    print("🔬  RANSOMWARE NEGOTIATION PIPELINE  │  v2.3.0 TASK-FIRST STRATEGY")
    print("━" * 70)
    print(f"📅  Date:       {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"📊  Target:     {max_chats if max_chats else 'All'} chats")
    print(f"🤖  Models:     {', '.join(models)}")
    print(f"⚡  Parallelism: {max_workers}/{len(models)} concurrent models")
    print(f"🧪  Tasks:      {num_tasks} (processed sequentially)")
    print(f"📚  Few-Shot:   ✅ ADAPTIVE (reduced for long chats)")
    print(f"🔪  Chunking:   ✅ RECURSIVE 3-WAY (max {ChunkConfig.MAX_PROMPT_CHARS:,} chars)")
    print(f"⚡  Strategy:   TASK-FIRST (all models on one prompt, then next)")
    print(f"📝  Log:        {LOG_FILE.name}")
    print("━" * 70)
    print()

# ═══════════════════════════════════════════════════════════════════
# PIPELINE CLASS
# ═══════════════════════════════════════════════════════════════════

class RansomwarePipeline:
    """v2.3.0 TASK-FIRST WITH PROFESSOR'S RECURSIVE CHUNKING"""
    
    def __init__(self):
        self.base_dir = BASE_DIR
        self.config_dir = self.base_dir / "config"
        self.output_dir = self.base_dir / "data" / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_config_path = self.config_dir / "model_config.yaml"
        with open(self.model_config_path, 'r', encoding='utf-8') as f:
            self.model_config = yaml.safe_load(f)

        self.models_list = self.model_config.get('ensemble_models', [self.model_config.get('active_model')])
        
        # Legge max_workers dal config, fallback a numero di modelli
        self.max_workers = self.model_config.get('processing', {}).get('max_workers', len(self.models_list))
        
        self.consensus_manager = ConsensusManager(self.base_dir)
        self.prompts_config = {}
        self.full_dataset = {}
        self.stats = PipelineStats()
        
        # Cache for few-shot examples
        self.few_shot_cache = {}
        self.chunker = ProfessorChunker()

    def load_resources(self):
        """Load prompts from separate YAML files and dataset."""
        
        # Carica i 3 file di prompt separati
        task_files = {
            'speech_act_analysis': 'prompt_speech_act_analysis.yaml',
            'psychological_profiling': 'prompt_psychological_profiling.yaml',
            'tactical_extraction': 'prompt_tactical_extraction.yaml'
        }
        
        self.prompts_config = {'tasks': {}}
        
        for task_name, filename in task_files.items():
            prompt_file_path = self.config_dir / filename
            
            try:
                with open(prompt_file_path, 'r', encoding='utf-8') as f:
                    task_data = yaml.safe_load(f)
                
                # Estrai le informazioni dal file YAML
                self.prompts_config['tasks'][task_name] = {
                    'output_format': task_data.get('output_format', 'json'),
                    'system_prompt': task_data.get('system_prompt', ''),
                    'user_template': task_data.get('user_prompt', '')
                }
                
                logger.info(f"✅ Loaded prompt template: {task_name}")
            
            except FileNotFoundError:
                logger.error(f"❌ Prompt file not found: {filename}")
                raise
            except Exception as e:
                logger.error(f"❌ Failed to load {filename}: {e}")
                raise
        
        logger.info(f"Loaded {len(self.prompts_config['tasks'])} task templates: {list(self.prompts_config['tasks'].keys())}")

        # Carica il dataset
        try:
            raw_rel_path = self.model_config['paths']['raw_data']
            data_path = self.base_dir / raw_rel_path
            logger.info(f"Loading dataset from {data_path}")
            self.full_dataset = download_and_load_messages_db(str(data_path))
            
            if not self.full_dataset:
                raise ValueError("Dataset is empty.")
            
            logger.info(f"Dataset loaded. Groups: {len(self.full_dataset)}")
        except Exception as e:
            logger.critical(f"Failed to load dataset: {e}")
            raise
    
    def _load_few_shot_examples(self, task_name: str, max_examples: int = None) -> list:
        """
        Load few-shot examples for a specific task (with caching and limiting).
        
        Args:
            task_name: Name of the task
            max_examples: Maximum number of examples to return (None = all)
        """
        # Check cache first
        if task_name in self.few_shot_cache:
            examples = self.few_shot_cache[task_name]
        else:
            few_shot_path = self.config_dir / "few_shot_examples" / f"{task_name}.json"
            
            if not few_shot_path.exists():
                logger.warning(f"⚠️  No few-shot file found: {few_shot_path}")
                self.few_shot_cache[task_name] = []
                return []
            
            try:
                with open(few_shot_path, 'r', encoding='utf-8') as f:
                    few_shot_data = json.load(f)
                
                examples = few_shot_data.get('examples', [])
                
                if not examples:
                    logger.warning(f"⚠️  No examples in few-shot file for {task_name}")
                    self.few_shot_cache[task_name] = []
                    return []
                
                logger.info(f"✅ Loaded {len(examples)} few-shot examples for {task_name}")
                self.stats.add_few_shot_loaded(task_name, len(examples))
                
                # Cache the results
                self.few_shot_cache[task_name] = examples
            
            except Exception as e:
                logger.error(f"❌ Failed to load few-shot for {task_name}: {e}")
                self.few_shot_cache[task_name] = []
                return []
        
        # Limit examples if requested
        if max_examples is not None and len(examples) > max_examples:
            return examples[:max_examples]
        
        return examples

    def _clean_json_output(self, text):
        """Robust JSON parser."""
        if isinstance(text, (list, dict)):
            return json.dumps(text, ensure_ascii=False)
            
        if not isinstance(text, str):
            return str(text) if text is not None else ""

        text = text.strip()
        
        try:
            parsed = json.loads(text)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON array
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                pass
        
        # Remove markdown code blocks
        text = re.sub(r'```json\s*|\s*```', '', text)
        try:
            parsed = json.loads(text)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
                
        logger.debug(f"JSON unrepairable: {text[:200]}")
        return None

    def _build_messages_with_few_shot(
        self, 
        task_name: str, 
        task_cfg: dict, 
        chat_json_str: str, 
        user_template: str,
        dialogue_length: int
    ) -> list:
        """
        Build messages array with adaptive few-shot examples.
        
        Args:
            task_name: Name of the task
            task_cfg: Task configuration
            chat_json_str: JSON string of the chat
            user_template: Template for user message
            dialogue_length: Number of messages in dialogue
        
        Returns:
            List of message dictionaries for the API
        """
        sys_msg = task_cfg['system_prompt']
        messages = [{"role": "system", "content": sys_msg}]
        
        # Adaptive few-shot loading based on chat size
        if dialogue_length > ChunkConfig.LONG_CHAT_THRESHOLD_MESSAGES:
            # Long chat - use minimal few-shot
            max_few_shot = 1
            logger.debug(f"Long chat detected ({dialogue_length} msgs), limiting few-shot to {max_few_shot}")
        else:
            # Normal chat - full few-shot
            max_few_shot = None
        
        few_shot_examples = self._load_few_shot_examples(task_name, max_examples=max_few_shot)
        
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
                assistant_response = json.dumps(output_data, indent=2, ensure_ascii=False)
            else:
                assistant_response = str(output_data)
            
            # Add to messages
            messages.append({"role": "user", "content": user_example})
            messages.append({"role": "assistant", "content": assistant_response})
        
        # Add actual query
        final_prompt = user_template.replace("{{chat_json}}", chat_json_str)
        messages.append({"role": "user", "content": final_prompt})
        
        logger.debug(f"Built message with {len(few_shot_examples)} few-shot examples for {task_name}")
        
        return messages

    def _process_single_chat(
        self, 
        client: UniBSLLMClient, 
        model_name: str,
        group_name: str, 
        chat_id: str, 
        dialogue: List[Dict],
        task_name: str, 
        task_cfg: dict
    ) -> Tuple[bool, str]:
        """
        Process a single chat for a specific task, with automatic chunking.
        Uses professor's recursive 3-way split based on actual prompt size.
        
        Returns:
            (success: bool, content: str)
        """
        # Clean dialogue
        dialogue = clean_message_list(dialogue)
        original_length = len(dialogue)
        
        # Get prompts for chunking decision
        system_prompt = task_cfg['system_prompt']
        user_template = task_cfg['user_template']
        
        # Use professor's chunking strategy
        chunks = self.chunker.chunk_chat(dialogue, system_prompt, user_template)
        
        if len(chunks) > 1:
            logger.info(f"[{model_name}] Chat {chat_id} chunked: {original_length} msgs → {len(chunks)} chunks")
            self.stats.add_chunked_chat(chat_id, len(chunks))
        
        # Process each chunk
        chunk_results = []
        
        for idx, chunk in enumerate(chunks):
            try:
                chunk_json_str = json.dumps(chunk, ensure_ascii=False)
                
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
                    logger.info(f"[{model_name}] ✓ {chat_id}/{task_name} chunk {idx+1}/{len(chunks)}")
                else:
                    logger.info(f"[{model_name}] ✓ {chat_id}/{task_name}")
                
                time.sleep(0.5)
            
            except Exception as e:
                logger.error(f"[{model_name}] Error processing chunk {idx+1}: {e}")
                chunk_results.append("")  # Add empty to maintain order
        
        # Merge chunk results
        if len(chunks) > 1:
            merged_content = self.chunker.merge_chunk_results(chunk_results)
            return True, merged_content
        else:
            return True, chunk_results[0] if chunk_results else ""

    def _process_model_for_task(self, model_name: str, task_name: str, task_cfg: dict, chat_queue: list, pbar):
        """
        ONE MODEL processes ALL chats for ONE TASK.
        This is called within a task loop.
        """
        # ✅ Suppress stdout during client creation
        with suppress_stdout():
            client = UniBSLLMClient(
                config_path=str(self.model_config_path), 
                model_override=model_name
            )
        
        logger.info(f"[{model_name}] Started task '{task_name}' for {len(chat_queue)} chats")
        
        for group_name, chat_id, chat_content in chat_queue:
            dialogue = chat_content.get('dialogue', [])
            if not dialogue:
                pbar.update(1)
                continue
            
            self.stats.increment_task(model_name, task_name)
            
            task_out_dir = self.output_dir / task_name / model_name / group_name
            task_out_dir.mkdir(parents=True, exist_ok=True)
            
            out_file = task_out_dir / f"{chat_id}.{task_cfg.get('output_format', 'txt')}"

            if out_file.exists():
                logger.debug(f"[{model_name}] Skip {chat_id}/{task_name}")
                pbar.update(1)
                continue

            try:
                # Process chat (with automatic chunking if needed)
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
                    
                    if cleaned_json is None:
                        warning = f"Invalid JSON in {task_name}"
                        self.stats.add_warning(chat_id, warning, model_name)
                        content = content  # Keep original
                    else:
                        content = cleaned_json
                        self.stats.add_success(model_name)

                # Write output
                with open(out_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                time.sleep(0.5)

            except Exception as e:
                error_msg = f"[{model_name}] {task_name}: {str(e)[:60]}"
                self.stats.add_error(chat_id, error_msg)
                logger.error(f"{chat_id}: {e}", exc_info=True)
            
            pbar.update(1)
        
        logger.info(f"[{model_name}] Completed task '{task_name}'")

    def run(self, max_chats=None):
        """
        Run pipeline with TASK-FIRST strategy.
        
        Flow:
        FOR each task:
            FOR each model (in parallel):
                Process all chats
        """
        tasks = self.prompts_config.get('tasks', {})
        
        # Build chat queue
        all_jobs = []
        for group_name, chats in self.full_dataset.items():
            for chat_id, chat_content in chats.items():
                all_jobs.append((group_name, chat_id, chat_content))

        if max_chats and len(all_jobs) > max_chats:
            all_jobs = all_jobs[:max_chats]

        self.stats.total_chats = len(all_jobs)
        
        print_banner(self.prompts_config, self.models_list, max_chats, len(tasks), self.max_workers)

        total_operations = len(all_jobs) * len(self.models_list) * len(tasks)
        
        # TASK-FIRST LOOP
        with tqdm(total=total_operations, desc="⚡ Processing", unit="operation", 
                  bar_format="{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                  colour="cyan", ncols=80) as pbar:
            
            for task_idx, (task_name, task_cfg) in enumerate(tasks.items(), 1):
                print(f"\n🧪 Task {task_idx}/{len(tasks)}: {task_name}")
                print(f"   Processing with {len(self.models_list)} models ({self.max_workers} parallel)...")
                
                # Run all models in parallel for THIS task
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
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
                            logger.error(f"❌ {model_name} failed on {task_name}: {e}", exc_info=True)
                
                print(f"   ✅ Task '{task_name}' completed by all models")
        
        # Consensus
        if len(self.models_list) > 1:
            print("\n🔄 Running consensus analysis...")
            for group_name, chat_id, _ in all_jobs:
                try:
                    self.consensus_manager.run_consensus_pipeline(group_name, chat_id, self.models_list)
                except Exception as e:
                    logger.error(f"Consensus error {chat_id}: {e}")
        
        self.stats.completed_chats = len(all_jobs)
        self.stats.print_summary()

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pipeline = RansomwarePipeline()
    try:
        pipeline.load_resources()
        pipeline.run(max_chats=10)  # Test with 5 chats first
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌  Fatal Error: {e}")
        logger.critical(e, exc_info=True)
        sys.exit(1)
