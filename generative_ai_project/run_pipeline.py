import sys
import json
import yaml
import logging.config
import time
import re  # <--- AGGIUNTO: Necessario per il parser "cattivo"
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
except ImportError as e:
    logger.critical(f"Error importing modules: {e}")
    sys.exit(1)



def print_banner(config, models, max_chats):
    """Prints a professional academic-style banner."""
    print("\n" + "="*70)
    print(f"🔬  RANSOMWARE NEGOTIATION ANALYSIS PIPELINE  |  v1.2.2 (Robust Parser)")
    print("="*70)
    print(f"📅  Date:      {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"📊  Target:    {max_chats if max_chats else 'Full Dataset'} chats")
    print(f"🤖  Ensemble:  {', '.join(models)}")
    print(f"⚙️   Workers:   {config.get('processing', {}).get('max_workers', 4)}")
    print(f"📝  Logs:      {LOG_FILE.name}")
    print("="*70 + "\n")



class RansomwarePipeline:
    def __init__(self):
        self.base_dir = BASE_DIR
        self.config_dir = self.base_dir / "config"
        self.output_dir = self.base_dir / "data" / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Model Config
        self.model_config_path = self.config_dir / "model_config.yaml"
        with open(self.model_config_path, 'r') as f:
            self.model_config = yaml.safe_load(f)


        # Use ensemble list if present, otherwise single active model
        self.models_list = self.model_config.get('ensemble_models', [self.model_config.get('active_model')])
        self.max_workers = self.model_config.get('processing', {}).get('max_workers', 4)
        
        self.consensus_manager = ConsensusManager(self.base_dir)
        self.prompts_config = {}
        self.full_dataset = {}
        self.few_shot_cache = {}


    def load_resources(self):
        """Quietly loads resources, logging details only to file."""
        # Load Prompts
        prompts_path = self.config_dir / "prompt_templates.yaml"
        try:
            with open(prompts_path, 'r', encoding='utf-8') as f:
                self.prompts_config = yaml.safe_load(f)
            logger.info(f"Loaded templates: {list(self.prompts_config.get('tasks', {}).keys())}")
        except Exception as e:
            logger.error(f"Failed to load prompts: {e}")
            raise


        # Load Data
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
        if task_name in self.few_shot_cache: return self.few_shot_cache[task_name]
        
        example_file = self.config_dir / "few_shot_examples" / f"{task_name}.json"
        if not example_file.exists(): return ""
        
        try:
            with open(example_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            examples = data.get('examples', [])
            if not examples: return ""
            
            formatted = "\n\n" + "="*60 + "\n📚 FEW-SHOT EXAMPLES:\n" + "="*60 + "\n"
            for i, ex in enumerate(examples, 1):
                formatted += f"\n🔹 Ex {i}:\nINPUT:\n{json.dumps(ex['input'], indent=2)}\nOUTPUT:\n{json.dumps(ex['output'], indent=2)}\n" + "-"*60
            
            formatted += "\nNow analyze the actual chat below:\n" + "="*60 + "\n"
            self.few_shot_cache[task_name] = formatted
            logger.info(f"Loaded {len(examples)} shots for {task_name}")
            return formatted
        except Exception:
            return ""


    def _clean_json_output(self, text):
        """
        Versione 'Cattiva' (Robust Parser) + Protezione Input Non-Stringa.
        """
        # PROTEZIONE: Se l'input è già una lista o un dizionario, convertilo in stringa JSON
        if isinstance(text, (list, dict)):
            return json.dumps(text)
            
        if not isinstance(text, str):
            # Se è None o altro tipo strano, converti in stringa vuota o repr
            return str(text) if text is not None else ""

        text = text.strip()
        
        # 1. Tentativo veloce: Se è già un JSON valido, ottimo.
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # 2. Tentativo chirurgico con Regex
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
                
        return text


    def _process_single_chat(self, group_name, chat_id, chat_content, tasks):
        """
        Processes a single chat with ALL models defined in self.models_list.
        """
        dialogue = chat_content.get('dialogue', [])
        if not dialogue: return "SKIPPED_EMPTY"


        dialogue = clean_message_list(dialogue)
        chat_json_str = json.dumps(dialogue, indent=2)
        
        # Iterate Models
        for model_name in self.models_list:
            client = UniBSLLMClient(config_path=str(self.model_config_path), model_override=model_name)
            
            for task_name, task_cfg in tasks.items():
                task_out_dir = self.output_dir / task_name / model_name / group_name
                task_out_dir.mkdir(parents=True, exist_ok=True)
                
                out_file = task_out_dir / f"{chat_id}.{task_cfg.get('output_format', 'txt')}"


                if out_file.exists(): continue 


                try:
                    sys_msg = task_cfg['system_prompt']
                    user_template = task_cfg['user_template']
                    examples = self._load_few_shot_examples(task_name)
                    
                    final_prompt = user_template.replace("{{chat_json}}", chat_json_str)
                    
                    if examples:
                        marker = "Chat to analyze:"
                        if marker in final_prompt:
                            final_prompt = final_prompt.replace(marker, examples + "\n" + marker)
                        else:
                            final_prompt = examples + "\n\n" + final_prompt


                    messages = [{"role": "system", "content": sys_msg}, {"role": "user", "content": final_prompt}]
                    resp_text = client.generate_response(messages)
                    
                    # Clean and Validate JSON
                    content = resp_text
                    if task_cfg.get('output_format') == 'json':
                        # Qui usiamo il nuovo parser "cattivo"
                        cleaned_resp = self._clean_json_output(resp_text)
                        
                        # Validate JSON before saving
                        try:
                            json.loads(cleaned_resp)
                            content = cleaned_resp
                        except json.JSONDecodeError as e:
                            logger.warning(f"⚠️ Invalid JSON from {model_name} on {chat_id}: {e}")
                            # Skip saving corrupted file to avoid consensus errors
                            continue


                    with open(out_file, 'w', encoding='utf-8') as f:
                        f.write(content)


                except Exception as e:
                    logger.error(f"Error {task_name}/{model_name}/{chat_id}: {e}")
                    return "ERROR"


        # Run Consensus (if enabled and applicable)
        if len(self.models_list) > 1 and 'speech_act_analysis' in tasks:
            try:
                self.consensus_manager.run_consensus_pipeline(group_name, chat_id, self.models_list)
            except Exception as e:
                logger.error(f"Consensus error {chat_id}: {e}")


        return "SUCCESS"


    def run(self, max_chats=None):
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
            # FIX: Properly unpack the job tuple (group, id, content)
            future_to_chat = {
                executor.submit(self._process_single_chat, job[0], job[1], job[2], tasks): job[1] 
                for job in all_jobs
            }


            pbar = tqdm(as_completed(future_to_chat), total=len(all_jobs), 
                        unit="chat",
                        bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                        colour="green")
            
            for future in pbar:
                chat_id = future_to_chat[future]
                try:
                    status = future.result()
                    if status == "SUCCESS": success_count += 1
                    elif status == "SKIPPED_EMPTY": skip_count += 1
                    else: error_count += 1
                except Exception as e:
                    logger.error(f"Thread execution failed for {chat_id}: {e}", exc_info=True)
                    error_count += 1


                pbar.set_description(f"🔍 Processing")


        print("\n" + "="*70)
        print("✅  EXECUTION SUMMARY")
        print("="*70)
        print(f"🟢  Completed:   {success_count}")
        print(f"🟡  Skipped:     {skip_count}")
        print(f"🔴  Errors:      {error_count}")
        print(f"📂  Output Dir:  {self.output_dir}")
        print(f"📝  Full Log:    {LOG_FILE}")
        print("="*70 + "\n")



if __name__ == "__main__":
    pipeline = RansomwarePipeline()
    try:
        pipeline.load_resources()
        # Set to 30 for your test run
        pipeline.run(max_chats=5)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌  Fatal Error: {e}")
        logger.critical(e, exc_info=True)
        sys.exit(1)
