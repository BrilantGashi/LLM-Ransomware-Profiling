import sys
import os
import json
import yaml
import logging
import time
from pathlib import Path


# --- SETUP LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("RansomPipeline")


# --- SETUP PATHS ---
# Add 'src' to system path to import custom modules
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR / "src"))


# --- IMPORT CUSTOM MODULES ---
try:
    from llm.openai_client import OpenAIClient
    from utils.data_loader import download_and_load_messages_db, clean_message_list
except ImportError as e:
    logger.critical(f"Error importing modules from src/: {e}")
    sys.exit(1)


class RansomwarePipeline:
    def __init__(self):
        self.base_dir = BASE_DIR
        self.config_dir = self.base_dir / "config"
        self.output_dir = self.base_dir / "data" / "outputs"
        
        # Create output folder
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Initialize AI Client
        config_path = self.config_dir / "model_config.yaml"
        logger.info(f"Initializing AI Client with: {config_path.name}")
        self.llm_client = OpenAIClient(config_path=str(config_path))
        
        self.prompts_config = {}
        self.full_dataset = {}  # Complete dictionary of chats


    def load_resources(self):
        """Load config and complete dataset using data_loader."""
        
        # A. Load Prompt Templates
        prompts_path = self.config_dir / "prompt_templates.yaml"
        try:
            with open(prompts_path, 'r', encoding='utf-8') as f:
                self.prompts_config = yaml.safe_load(f)
            logger.info(f"Templates loaded: {list(self.prompts_config.get('tasks', {}).keys())}")
        except Exception as e:
            logger.error(f"Error loading prompt_templates.yaml: {e}")
            raise


        # B. Load Data via utils.data_loader
        # Reads path from model_config.yaml (e.g., data/raw/messages.json)
        try:
            raw_rel_path = self.llm_client.config['paths']['raw_data']
            data_path = self.base_dir / raw_rel_path
            
            logger.info(f"Loading data from: {data_path}")
            
            # Call utility function that handles local files/downloads/generation
            self.full_dataset = download_and_load_messages_db(str(data_path))
            
            if not self.full_dataset:
                raise ValueError("Loaded dataset is empty.")
                
            logger.info(f"Dataset ready. Groups found: {len(self.full_dataset)}")
            
        except KeyError:
            logger.error("Missing 'paths.raw_data' in model_config.yaml")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise


    def _clean_json_output(self, text):
        """Remove markdown ```json formatting to extract content."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"): 
                lines = lines[1:]
            if lines and lines[-1].startswith("```"): 
                lines = lines[:-1]
            return "\n".join(lines)
        return text


    def run(self, max_chats=1):
        """
        Execute the pipeline.
        :param max_chats: Maximum number of chats to analyze for testing (Default: 1).
                          Set to None to process all chats.
        """
        logger.info(f"\nStarting analysis on {max_chats if max_chats else 'ALL'} chats\n")
        
        tasks = self.prompts_config.get('tasks', {})
        processed_count = 0


        # Iterate over groups (e.g., Akira, Conti...) and their chats
        for group_name, chats in self.full_dataset.items():
            for chat_id, chat_content in chats.items():
                
                if max_chats and processed_count >= max_chats:
                    logger.info("Chat processing limit reached for testing.")
                    return


                processed_count += 1
                logger.info(f"--- Analyzing Chat #{processed_count}: {group_name}/{chat_id} ---")


                # Extract and clean messages
                dialogue = chat_content.get('dialogue', [])
                if not dialogue:
                    logger.warning(f"Chat {chat_id} is empty, skipping.")
                    continue
                
                # Clean characters using data_loader function
                dialogue = clean_message_list(dialogue)
                chat_json_str = json.dumps(dialogue, indent=2)


                # EXECUTE TASKS DEFINED IN YAML
                for task_name, task_cfg in tasks.items():
                    # Output file path specific to this chat and task
                    # Structure: data/outputs/task_name/group_name/chat_id.json
                    task_out_dir = self.output_dir / task_name / group_name
                    task_out_dir.mkdir(parents=True, exist_ok=True)
                    
                    out_file = task_out_dir / f"{chat_id}.{task_cfg.get('output_format', 'txt')}"
                    
                    if out_file.exists():
                        logger.info(f"Task '{task_name}' already executed for {chat_id}, skipping.")
                        continue


                    logger.info(f"Executing: {task_name}")


                    try:
                        # 1. Prepare Prompt
                        sys_msg = task_cfg['system_prompt']
                        usr_tmpl = task_cfg['user_template']
                        # Optional: truncate long chats to avoid token limit errors
                        # chat_json_str[:10000] 
                        final_prompt = usr_tmpl.replace("{{chat_json}}", chat_json_str)


                        # 2. Call AI
                        messages = [
                            {"role": "system", "content": sys_msg},
                            {"role": "user", "content": final_prompt}
                        ]
                        
                        resp_text = self.llm_client.generate_response(messages)


                        # 3. Save Results
                        if task_cfg.get('output_format') == 'json':
                            cleaned_resp = self._clean_json_output(resp_text)
                            with open(out_file, 'w', encoding='utf-8') as f:
                                f.write(cleaned_resp)  # Save even if not perfect JSON for debugging
                        else:
                            with open(out_file, 'w', encoding='utf-8') as f:
                                f.write(resp_text)
                        
                    except Exception as e:
                        logger.error(f"Error in task {task_name}: {e}")
                        continue


        logger.info("Pipeline execution completed.")


if __name__ == "__main__":
    pipeline = RansomwarePipeline()
    try:
        pipeline.load_resources()
        # Set max_chats=None to process the entire dataset
        pipeline.run(max_chats=5) 
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
