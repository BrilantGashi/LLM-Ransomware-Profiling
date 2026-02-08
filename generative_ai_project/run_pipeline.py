"""
Ransomware Negotiation Analysis Pipeline - v1.9.1 MODEL-FIRST CLEAN
Each model processes ONE chat at a time (all 3 tasks sequentially)
Max 3 concurrent requests (one per model) - CLEAN OUTPUT
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

# Console Handler - SOLO ERRORI (blocca INFO!)
console_handler = TqdmLoggingHandler()
console_handler.setLevel(logging.ERROR)  # ← CRITICAL: Blocca INFO/WARNING
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
# STATISTICS TRACKER (Thread-Safe)
# ═══════════════════════════════════════════════════════════════════

class PipelineStats:
    def __init__(self):
        self.start_time = datetime.now()
        self.total_chats = 0
        self.completed_chats = 0
        self.chat_warnings = defaultdict(list)
        self.chat_errors = defaultdict(list)
        self.model_stats = defaultdict(lambda: {'valid': 0, 'invalid': 0, 'tasks': 0})
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
    
    def increment_task(self, model: str):
        with self._lock:
            self.model_stats[model]['tasks'] += 1
    
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
        print(f"  ├─ ⚠️  With Warnings:  {len(self.chat_warnings)}")
        print(f"  └─ ❌ Errors:        {len(self.chat_errors)}")
        
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
# BANNER
# ═══════════════════════════════════════════════════════════════════

def print_banner(config, models, max_chats):
    print()
    print("━" * 70)
    print("🔬  RANSOMWARE NEGOTIATION PIPELINE  │  v1.9.1 CLEAN")
    print("━" * 70)
    print(f"📅  Date:       {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"📊  Target:     {max_chats if max_chats else 'All'} chats")
    print(f"🤖  Models:     {', '.join(models)}")
    print(f"🧪  Tasks:      {len(config.get('tasks', {}))} per chat")
    print(f"⚡  Strategy:   MODEL-FIRST (max 3 concurrent requests)")
    print(f"📝  Log:        {LOG_FILE.name}")
    print("━" * 70)
    print()

# ═══════════════════════════════════════════════════════════════════
# PIPELINE CLASS
# ═══════════════════════════════════════════════════════════════════

class RansomwarePipeline:
    """v1.9.1 MODEL-FIRST CLEAN: Clean output with suppressed logs"""
    
    def __init__(self):
        self.base_dir = BASE_DIR
        self.config_dir = self.base_dir / "config"
        self.output_dir = self.base_dir / "data" / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_config_path = self.config_dir / "model_config.yaml"
        with open(self.model_config_path, 'r', encoding='utf-8') as f:
            self.model_config = yaml.safe_load(f)

        self.models_list = self.model_config.get('ensemble_models', [self.model_config.get('active_model')])
        self.consensus_manager = ConsensusManager(self.base_dir)
        self.prompts_config = {}
        self.full_dataset = {}
        self.stats = PipelineStats()

    def load_resources(self):
        """Load prompts and dataset."""
        prompts_path = self.config_dir / "prompt_templates.yaml"
        try:
            with open(prompts_path, 'r', encoding='utf-8') as f:
                self.prompts_config = yaml.safe_load(f)
            logger.info(f"Loaded templates: {list(self.prompts_config.get('tasks', {}).keys())}")
        except Exception as e:
            logger.error(f"Failed to load prompts: {e}")
            raise

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

        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                pass
        
        text = re.sub(r'```json\s*|\s*```', '', text)
        try:
            parsed = json.loads(text)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
                
        logger.debug(f"JSON unrepairable: {text[:200]}")
        return None

    def _process_model_pipeline(self, model_name: str, chat_queue: list, tasks: dict, pbar):
        """ONE MODEL processes ALL chats sequentially."""
        
        # ✅ Suppress stdout during client creation
        with suppress_stdout():
            client = UniBSLLMClient(
                config_path=str(self.model_config_path), 
                model_override=model_name
            )
        
        logger.info(f"[{model_name}] Started processing {len(chat_queue)} chats")
        
        for group_name, chat_id, chat_content in chat_queue:
            dialogue = chat_content.get('dialogue', [])
            if not dialogue:
                pbar.update(1)
                continue
            
            dialogue = clean_message_list(dialogue)
            chat_json_str = json.dumps(dialogue, indent=2, ensure_ascii=False)
            
            # Process ALL tasks sequentially
            for task_name, task_cfg in tasks.items():
                self.stats.increment_task(model_name)
                
                task_out_dir = self.output_dir / task_name / model_name / group_name
                task_out_dir.mkdir(parents=True, exist_ok=True)
                
                out_file = task_out_dir / f"{chat_id}.{task_cfg.get('output_format', 'txt')}"

                if out_file.exists():
                    logger.debug(f"[{model_name}] Skip {chat_id}/{task_name}")
                    continue

                try:
                    sys_msg = task_cfg['system_prompt']
                    user_template = task_cfg['user_template']
                    final_prompt = user_template.replace("{{chat_json}}", chat_json_str)

                    messages = [
                        {"role": "system", "content": sys_msg}, 
                        {"role": "user", "content": final_prompt}
                    ]

                    response_obj = client.generate_response(messages)
                    
                    if isinstance(response_obj, dict):
                        resp_text = response_obj.get('content', '')
                    else:
                        resp_text = response_obj
                    
                    content = resp_text
                    
                    # Validate JSON
                    if task_cfg.get('output_format') == 'json':
                        cleaned_json = self._clean_json_output(resp_text)
                        
                        if cleaned_json is None:
                            warning = f"Invalid JSON in {task_name}"
                            self.stats.add_warning(chat_id, warning, model_name)
                            content = resp_text
                        else:
                            content = cleaned_json
                            self.stats.add_success(model_name)

                    with open(out_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    logger.info(f"[{model_name}] ✓ {chat_id}/{task_name}")
                    time.sleep(0.5)

                except Exception as e:
                    error_msg = f"[{model_name}] {task_name}: {str(e)[:60]}"
                    self.stats.add_error(chat_id, error_msg)
                    logger.error(f"{chat_id}: {e}", exc_info=True)
            
            pbar.update(1)
        
        logger.info(f"[{model_name}] Completed all chats")

    def run(self, max_chats=None):
        """Run pipeline with MODEL-FIRST strategy."""
        tasks = self.prompts_config.get('tasks', {})
        all_jobs = []
        
        for group_name, chats in self.full_dataset.items():
            for chat_id, chat_content in chats.items():
                all_jobs.append((group_name, chat_id, chat_content))

        if max_chats and len(all_jobs) > max_chats:
            all_jobs = all_jobs[:max_chats]

        self.stats.total_chats = len(all_jobs)
        
        print_banner(self.prompts_config, self.models_list, max_chats)

        total_tasks = len(all_jobs) * len(self.models_list)
        
        with tqdm(total=total_tasks, desc="⚡ Processing", unit="chat·model", 
                  bar_format="{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                  colour="cyan", ncols=80) as pbar:
            
            with ThreadPoolExecutor(max_workers=len(self.models_list)) as executor:
                futures = []
                
                for model_name in self.models_list:
                    future = executor.submit(
                        self._process_model_pipeline,
                        model_name, all_jobs, tasks, pbar
                    )
                    futures.append((future, model_name))
                
                for future, model_name in futures:
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"❌ {model_name} failed: {e}", exc_info=True)
        
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
        pipeline.run(max_chats=10)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌  Fatal Error: {e}")
        logger.critical(e, exc_info=True)
        sys.exit(1)
