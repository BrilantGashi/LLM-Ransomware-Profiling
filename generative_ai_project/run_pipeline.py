"""
Ransomware Negotiation Analysis Pipeline
Main execution script with multi-model ensemble support and robust error handling.
Compliant with UniBS Cluster Handbook (February 2026)

Author: Brilant Gashi
Supervisors: Prof. Federico Cerutti, Prof. Pietro Baroni
University of Brescia - 2025/2026
"""

import sys
import json
import yaml
import logging.config
import time
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


# --- TQDM IMPORT (Progress Bar) ---
try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not installed. Install with 'pip install tqdm' for progress bars.")
    def tqdm(iterable, **kwargs): return iterable


# --- SETUP PATHS ---
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR / "src"))


# --- LOGGING SETUP (Dual Mode: Console Clean / File Verbose) ---
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


class TqdmLoggingHandler(logging.Handler):
    """Avoid tqdm interference with logging."""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


# Configure Root Logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# 1. File Handler (Verbose - DEBUG level)
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
root_logger.addHandler(file_handler)

# 2. Console Handler (Clean - WARNING+ level only)
console_handler = TqdmLoggingHandler()
console_handler.setLevel(logging.WARNING) 
console_formatter = logging.Formatter('⚠️  %(message)s')
console_handler.setFormatter(console_formatter)
root_logger.addHandler(console_handler)

# Specific logger for this script
logger = logging.getLogger("RansomPipeline")


# --- IMPORT MODULES ---
try:
    from src.llm.unibs_client import UniBSLLMClient
    from src.utils.data_loader import download_and_load_messages_db, clean_message_list
    from src.analysis.consensus import ConsensusManager
    from src.handlers.error_handler import UniBSErrorHandler
except ImportError as e:
    logger.critical(f"Error importing modules: {e}")
    sys.exit(1)


def print_banner(config, models, max_chats):
    """Prints a professional academic-style banner."""
    print("\n" + "="*70)
    print(f"🔬  RANSOMWARE NEGOTIATION ANALYSIS PIPELINE  |  v1.3.0")
    print("="*70)
    print(f"📅  Date:      {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"📊  Target:    {max_chats if max_chats else 'Full Dataset'} chats")
    print(f"🤖  Ensemble:  {', '.join(models)}")
    print(f"⚙️   Workers:   {config.get('processing', {}).get('max_workers', 4)}")
    print(f"📝  Logs:      {LOG_FILE.name}")
    print("="*70 + "\n")


class RansomwarePipeline:
    """
    Main pipeline orchestrator for ransomware negotiation analysis.
    
    Supports multi-model ensemble processing with consensus validation,
    automatic chunking for long dialogues, and robust error handling.
    """
    
    def __init__(self):
        """Initialize pipeline with configuration files and directory structure."""
        self.base_dir = BASE_DIR
        self.config_dir = self.base_dir / "config"
        self.output_dir = self.base_dir / "data" / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Model Config with UTF-8 encoding
        self.model_config_path = self.config_dir / "model_config.yaml"
        with open(self.model_config_path, 'r', encoding='utf-8') as f:
            self.model_config = yaml.safe_load(f)

        # Use ensemble list if present, otherwise single active model
        self.models_list = self.model_config.get('ensemble_models', [self.model_config.get('active_model')])
        self.max_workers = self.model_config.get('processing', {}).get('max_workers', 4)
        self.chunk_max_chars = self.model_config.get('processing', {}).get('chunk_max_chars', 10000)
        
        # Initialize error handler (Handbook Section 9 compliant)
        retry_config = self.model_config.get('processing', {}).get('retry', {})
        self.error_handler = UniBSErrorHandler(
            max_retries=retry_config.get('max_attempts', 3),
            backoff_factor=retry_config.get('backoff_factor', 2)
        )
        
        self.consensus_manager = ConsensusManager(self.base_dir)
        self.prompts_config = {}
        self.full_dataset = {}
        self.few_shot_cache = {}
        
        # Feature flags
        self.save_reasoning = self.model_config.get('logging', {}).get('save_reasoning', True)
        self.validate_json = self.model_config.get('features', {}).get('validate_json_output', True)

    def load_resources(self):
        """
        Load all required resources: prompts, dataset, and configurations.
        Logs details to file, prints minimal info to console.
        """
        # Load Prompts with UTF-8 encoding
        prompts_path = self.config_dir / "prompt_templates.yaml"
        try:
            with open(prompts_path, 'r', encoding='utf-8') as f:
                self.prompts_config = yaml.safe_load(f)
            logger.info(f"Loaded templates: {list(self.prompts_config.get('tasks', {}).keys())}")
        except Exception as e:
            logger.error(f"Failed to load prompts: {e}")
            raise

        # Load Dataset
        try:
            raw_rel_path = self.model_config['paths']['raw_data']
            data_path = self.base_dir / raw_rel_path
            print(f"📂 Loading dataset from local source...") 
            self.full_dataset = download_and_load_messages_db(str(data_path))
            
            if not self.full_dataset:
                raise ValueError("Dataset is empty.")
            
            logger.info(f"Dataset loaded. Groups: {len(self.full_dataset)}")
        except Exception as e:
            logger.critical(f"Failed to load dataset: {e}")
            raise

    def _load_few_shot_examples(self, task_name: str) -> str:
        """
        Load few-shot examples from config directory.
        
        Args:
            task_name: Name of the analysis task
            
        Returns:
            Formatted few-shot examples string or empty string if unavailable
        """
        if task_name in self.few_shot_cache: 
            return self.few_shot_cache[task_name]
        
        example_file = self.config_dir / "few_shot_examples" / f"{task_name}.json"
        if not example_file.exists(): 
            return ""
        
        try:
            with open(example_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            examples = data.get('examples', [])
            if not examples: 
                return ""
            
            formatted = "\n\n" + "="*60 + "\n📚 FEW-SHOT EXAMPLES:\n" + "="*60 + "\n"
            for i, ex in enumerate(examples, 1):
                formatted += f"\n🔹 Example {i}:\nINPUT:\n{json.dumps(ex['input'], indent=2)}\nOUTPUT:\n{json.dumps(ex['output'], indent=2)}\n" + "-"*60
            
            formatted += "\nNow analyze the actual chat below:\n" + "="*60 + "\n"
            self.few_shot_cache[task_name] = formatted
            logger.info(f"Loaded {len(examples)} shots for {task_name}")
            return formatted
        except Exception as e:
            logger.warning(f"Failed to load few-shot examples for {task_name}: {e}")
            return ""

    def _chunk_dialogue_if_needed(self, dialogue, max_chars=None):
        """
        Split long dialogues to fit within model context window.
        Uses recursive 3-way split for balanced chunks.
        
        Args:
            dialogue: List of message dictionaries
            max_chars: Maximum characters before splitting (default from config)
        
        Returns:
            List of dialogue chunks
        """
        if max_chars is None:
            max_chars = self.chunk_max_chars
        
        chat_json = json.dumps(dialogue, ensure_ascii=False)
        
        if len(chat_json) <= max_chars or len(dialogue) <= 1:
            return [dialogue]
        
        # Recursive 3-way split
        third = max(1, len(dialogue) // 3)
        two_third = max(third + 1, 2 * len(dialogue) // 3)
        
        return (
            self._chunk_dialogue_if_needed(dialogue[:third], max_chars) +
            self._chunk_dialogue_if_needed(dialogue[third:two_third], max_chars) +
            self._chunk_dialogue_if_needed(dialogue[two_third:], max_chars)
        )

    def _validate_and_repair_json(self, text):
        """
        Validate and repair malformed JSON output from LLM.
        
        Applies multiple repair strategies:
        1. Quick validation (already valid)
        2. Regex extraction of JSON arrays/objects
        3. Markdown code block removal
        
        Args:
            text: Raw model output (string, dict, or list)
        
        Returns:
            str: Valid JSON string or None if unrepairable
        """
        # Protection: If input is already a list or dict, serialize it
        if isinstance(text, (list, dict)):
            return json.dumps(text, ensure_ascii=False)
        
        if not isinstance(text, str):
            return str(text) if text is not None else None
        
        text = text.strip()
        
        # 1. Quick validation: Already valid JSON
        try:
            parsed = json.loads(text)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
        
        # 2. Regex extraction of JSON array
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return json.dumps(parsed, ensure_ascii=False)
            except json.JSONDecodeError:
                pass
        
        # 3. Remove markdown code blocks
        text = re.sub(r'```json\s*|\s*```', '', text)
        try:
            parsed = json.loads(text)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
        
        logger.error(f"JSON unrepairable: {text[:200]}")
        return None

    def _process_single_chat(self, group_name, chat_id, chat_content, tasks):
        """
        Process a single chat with ALL models in ensemble.
        
        Workflow:
        1. Clean and validate dialogue
        2. Chunk if exceeds context window
        3. Process with each model in ensemble
        4. Apply JSON validation if enabled
        5. Save reasoning traces if enabled
        6. Run consensus if multi-model
        
        Args:
            group_name: Ransomware group name
            chat_id: Unique chat identifier
            chat_content: Chat data with 'dialogue' key
            tasks: Dictionary of tasks to execute
        
        Returns:
            str: "SUCCESS", "SKIPPED_EMPTY", or "ERROR"
        """
        dialogue = chat_content.get('dialogue', [])
        if not dialogue: 
            return "SKIPPED_EMPTY"

        dialogue = clean_message_list(dialogue)
        
        # Chunk dialogue if too long
        dialogue_chunks = self._chunk_dialogue_if_needed(dialogue)
        if len(dialogue_chunks) > 1:
            logger.info(f"Chat {chat_id} split into {len(dialogue_chunks)} chunks")
        
        # Iterate Models in Ensemble
        for model_name in self.models_list:
            client = UniBSLLMClient(config_path=str(self.model_config_path), model_override=model_name)
            
            for task_name, task_cfg in tasks.items():
                task_out_dir = self.output_dir / task_name / model_name / group_name
                task_out_dir.mkdir(parents=True, exist_ok=True)
                
                out_file = task_out_dir / f"{chat_id}.{task_cfg.get('output_format', 'txt')}"

                # Skip if already processed
                if out_file.exists(): 
                    continue 

                try:
                    sys_msg = task_cfg['system_prompt']
                    user_template = task_cfg['user_template']
                    examples = self._load_few_shot_examples(task_name)
                    
                    # Process all chunks and aggregate results
                    all_results = []
                    
                    for chunk_idx, chunk in enumerate(dialogue_chunks):
                        chat_json_str = json.dumps(chunk, indent=2, ensure_ascii=False)
                        final_prompt = user_template.replace("{{chat_json}}", chat_json_str)
                        
                        # Add few-shot examples
                        if examples:
                            marker = "Chat to analyze:"
                            if marker in final_prompt:
                                final_prompt = final_prompt.replace(marker, examples + "\n" + marker)
                            else:
                                final_prompt = examples + "\n\n" + final_prompt

                        messages = [
                            {"role": "system", "content": sys_msg}, 
                            {"role": "user", "content": final_prompt}
                        ]
                        
                        # Use error handler with retry logic (Handbook compliant)
                        response_obj = self.error_handler.with_retry(
                            client.generate_response,
                            messages
                        )
                        
                        # Handle reasoning_content (Handbook page 2)
                        resp_text = response_obj.get('content', '')
                        reasoning = response_obj.get('reasoning', None)
                        
                        # Save reasoning if enabled and available
                        if self.save_reasoning and reasoning:
                            reasoning_file = task_out_dir / f"{chat_id}_chunk{chunk_idx}_reasoning.txt"
                            reasoning_file.write_text(reasoning, encoding='utf-8')
                            logger.debug(f"Saved reasoning for {chat_id} chunk {chunk_idx}")
                        
                        all_results.append(resp_text)
                    
                    # Aggregate chunks (simple concatenation for now)
                    content = "\n\n".join(all_results) if len(all_results) > 1 else all_results[0]
                    
                    # Validate and clean JSON output
                    if task_cfg.get('output_format') == 'json' and self.validate_json:
                        cleaned_json = self._validate_and_repair_json(content)
                        
                        if cleaned_json is None:
                            logger.warning(f"⚠️ Invalid JSON from {model_name} on {chat_id}, skipping save")
                            continue
                        
                        content = cleaned_json

                    # Save final output with UTF-8 encoding
                    with open(out_file, 'w', encoding='utf-8') as f:
                        f.write(content)

                except Exception as e:
                    logger.error(f"Error {task_name}/{model_name}/{chat_id}: {e}", exc_info=True)
                    return "ERROR"

        # Run Consensus (if enabled and applicable)
        if len(self.models_list) > 1 and 'speech_act_analysis' in tasks:
            try:
                self.consensus_manager.run_consensus_pipeline(group_name, chat_id, self.models_list)
            except Exception as e:
                logger.error(f"Consensus error {chat_id}: {e}")

        return "SUCCESS"

    def run(self, max_chats=None):
        """
        Execute the full pipeline with parallel processing.
        
        Args:
            max_chats: Maximum number of chats to process (None = all)
        """
        print_banner(self.model_config, self.models_list, max_chats)
        
        tasks = self.prompts_config.get('tasks', {})
        all_jobs = []
        
        # Flatten the dataset into a list of jobs
        for group_name, chats in self.full_dataset.items():
            for chat_id, chat_content in chats.items():
                all_jobs.append((group_name, chat_id, chat_content))

        # Slice dataset if max_chats is set
        if max_chats and len(all_jobs) > max_chats:
            all_jobs = all_jobs[:max_chats]

        print(f"🚀  Initialization complete. Processing {len(all_jobs)} chats...\n")

        success_count = 0
        skip_count = 0
        error_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chat = {
                executor.submit(self._process_single_chat, job[0], job[1], job[2], tasks): job[1] 
                for job in all_jobs
            }

            pbar = tqdm(
                as_completed(future_to_chat), 
                total=len(all_jobs), 
                unit="chat",
                bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                colour="green"
            )
            
            for future in pbar:
                chat_id = future_to_chat[future]
                try:
                    status = future.result()
                    if status == "SUCCESS": 
                        success_count += 1
                    elif status == "SKIPPED_EMPTY": 
                        skip_count += 1
                    else: 
                        error_count += 1
                except Exception as e:
                    logger.error(f"Thread execution failed for {chat_id}: {e}", exc_info=True)
                    error_count += 1

                pbar.set_description(f"🔍 Processing")

        # Print execution summary
        print("\n" + "="*70)
        print("✅  EXECUTION SUMMARY")
        print("="*70)
        print(f"🟢  Completed:   {success_count}")
        print(f"🟡  Skipped:     {skip_count}")
        print(f"🔴  Errors:      {error_count}")
        print(f"📂  Output Dir:  {self.output_dir}")
        print(f"📝  Full Log:    {LOG_FILE}")
        
        # Print error report (Handbook Section 9)
        error_report = self.error_handler.get_error_report()
        if error_report and "Nessun errore" not in error_report and "No errors" not in error_report:
            print("\n⚠️  ERRORS DETECTED - See error_report.txt")
            error_report_file = LOG_DIR / "error_report.txt"
            error_report_file.write_text(error_report, encoding='utf-8')
        
        print("="*70 + "\n")


if __name__ == "__main__":
    pipeline = RansomwarePipeline()
    try:
        pipeline.load_resources()
        # Set to desired number for test run (None = all chats)
        pipeline.run(max_chats=1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌  Fatal Error: {e}")
        logger.critical(e, exc_info=True)
        sys.exit(1)
