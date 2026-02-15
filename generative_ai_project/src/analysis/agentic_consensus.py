"""
Agentic Consensus Pipeline
--------------------------
Orchestrates the aggregation of multi-model LLM analyses into a single 
consolidated 'Gold Standard' utilizing an Agentic LLM as the adjudicator.

Features:
- Robust input handling: tolerates missing or empty model outputs.
- Graceful degradation: proceeds with partial data (1 or 2 models).
- Structured logging and error tracking.
- Metadata injection for traceability.

Author: Brilant Gashi
Project: LLM-Ransomware-Profiling
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import yaml

from ..llm.unibs_client import LLMClient

# Configure module-level logger
logger = logging.getLogger(__name__)


class AgenticConsensusManager:
    """
    Manages the lifecycle of agentic consensus generation:
    loading inputs, prompting the LLM adjudicator, and saving results.
    """

    def __init__(self, project_root: Path, consensus_model: Optional[str] = None):
        """
        Initialize the manager with project paths and LLM client.

        Args:
            project_root (Path): Root directory of the project.
            consensus_model (Optional[str]): Override model from config (e.g., via CLI).
        """
        self.project_root = project_root
        self.outputs_dir = project_root / "data" / "outputs"
        self.consensus_dir = project_root / "data" / "consensus_agentic"
        self.config_path = project_root / "config" / "model_config.yaml"
        self.prompt_path = project_root / "config" / "agentic_consensus_prompts.yaml"

        # Ensure output directory exists
        self.consensus_dir.mkdir(parents=True, exist_ok=True)

        # Load Configuration
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.app_config = yaml.safe_load(f)

        # Determine Model: CLI override > Config file > Default fallback
        # Looks into consensus -> agentic -> active_model
        config_model = self.app_config.get('consensus', {}).get('agentic', {}).get('active_model')
        self.model_name = consensus_model or config_model or 'phi4'

        # Initialize LLM Client
        try:
            self.llm_client = LLMClient(str(self.config_path), model_override=self.model_name)
            
            # Optional: Apply specific generation parameters for consensus if defined
            consensus_params = self.app_config.get('consensus', {}).get('agentic', {})
            
            # Override client parameters if specific consensus settings exist
            if 'temperature' in consensus_params:
                self.llm_client.llm_params['temperature'] = consensus_params['temperature']
            if 'max_tokens' in consensus_params:
                self.llm_client.llm_params['max_tokens'] = consensus_params['max_tokens']
                
            logger.info(f"AgenticConsensusManager initialized with model: {self.model_name}")
            
        except Exception as e:
            logger.critical(f"Failed to initialize LLM Client: {e}")
            raise

        # Load Prompt Templates
        self._load_prompts()

    def _load_prompts(self):
        """Loads consensus prompts from the YAML configuration."""
        if not self.prompt_path.exists():
            raise FileNotFoundError(f"Prompt configuration not found at {self.prompt_path}")
        
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            self.prompts = config.get('consensus_prompts', {})
            
        if not self.prompts:
            logger.warning("No 'consensus_prompts' section found in agentic_consensus_prompts.yaml")

    def load_model_outputs(self, task: str, group: str, chat_id: str, models: List[str]) -> Dict[str, Any]:
        """
        Retrieves and validates outputs from specified models.
        
        Returns:
            Dict[str, Any]: Dictionary mapping model names to their loaded JSON data.
                            Only valid, non-empty JSONs are included.
        """
        valid_data = {}

        for model in models:
            file_path = self.outputs_dir / task / model / group / f"{chat_id}.json"

            if not file_path.exists():
                logger.debug(f"File not found: {file_path}")
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                    if not content:
                        logger.warning(f"Empty file encountered: {file_path}")
                        continue

                    data = json.loads(content)

                    # Validation: Check for empty JSON structures
                    if isinstance(data, list) and not data:
                        logger.warning(f"Empty JSON list in {file_path}")
                        continue
                    if isinstance(data, dict) and not data:
                        logger.warning(f"Empty JSON object in {file_path}")
                        continue

                    valid_data[model] = data

            except json.JSONDecodeError:
                logger.error(f"Invalid JSON format in {file_path}")
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

        return valid_data

    def construct_prompt(self, task: str, inputs: Dict[str, Any]) -> Tuple[str, str]:
        """
        Constructs system and user prompts based on task templates.
        """
        template_config = self.prompts.get(task)
        if not template_config:
            # Fallback if specific task prompt is missing
            logger.warning(f"No specific prompt template for task '{task}'. Using generic fallback.")
            return (
                "You are a data aggregation assistant.",
                f"Merge these JSON inputs into one:\n{json.dumps(inputs)}"
            )

        system_prompt = template_config.get('system_prompt', '')
        user_template = template_config.get('user_template', '')

        # Serialize inputs for the prompt
        input_str = ""
        for model, data in inputs.items():
            input_str += f"\n--- MODEL: {model} ---\n{json.dumps(data, indent=2)}\n"

        # Fill placeholders
        user_message = user_template.replace('{{ num_models }}', str(len(inputs)))
        user_message = user_message.replace('{{ model_inputs }}', input_str)

        return system_prompt, user_message

    def _parse_llm_response(self, response_content: str) -> Any:
        """
        Parses LLM output, handling potential markdown code blocks.
        """
        # Strip Markdown code fencing if present
        cleaned = re.sub(r'^```json\s*', '', response_content, flags=re.MULTILINE)
        cleaned = re.sub(r'^```\s*', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
        
        return json.loads(cleaned.strip())

    def run_consensus_for_chat(self, task: str, group: str, chat_id: str, models: List[str]) -> bool:
        """
        Executes the agentic consensus pipeline for a single chat.
        
        Steps:
        1. Load valid inputs.
        2. Construct prompt.
        3. Invoke LLM.
        4. Save result with metadata.
        """
        # 1. Load Data
        inputs = self.load_model_outputs(task, group, chat_id, models)

        if not inputs:
            logger.error(f"Skipping {chat_id}: No valid model outputs found.")
            return False

        logger.info(f"Processing {chat_id} | Task: {task} | Sources: {len(inputs)}/{len(models)}")

        # 2. Prompting
        system_prompt, user_message = self.construct_prompt(task, inputs)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        # 3. Inference
        try:
            response = self.llm_client.generate_response(messages)
            result_json = self._parse_llm_response(response['content'])
        except Exception as e:
            logger.error(f"Consensus generation failed for {chat_id}: {e}")
            return False

        # 4. Save Output
        output_data = {
            "metadata": {
                "consensus_method": "agentic_adjudication",
                "aggregator_model": self.llm_client.model_name,
                "source_models_count": len(inputs),
                "source_models": list(inputs.keys()),
                "reliability_tier": "high" if len(inputs) >= 3 else "partial_data"
            },
            "consensus_result": result_json
        }

        save_dir = self.consensus_dir / task / group
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{chat_id}.json"

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved consensus to: {save_path}")
            return True
        except IOError as e:
            logger.error(f"Failed to write output file {save_path}: {e}")
            return False
