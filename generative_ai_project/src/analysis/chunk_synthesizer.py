"""
Chunk Synthesizer Module

Merges partial analysis outputs from multiple chunks into a single coherent result.
Used for tasks requiring holistic analysis (psychological profiling, tactical extraction).
"""

import json
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.llm.unibs_client import LLMClient

logger = logging.getLogger("ChunkSynthesizer")


class ChunkSynthesizer:
    """
    Synthesizes multiple chunk-level analysis outputs into unified results.
    
    Uses an LLM to intelligently merge partial outputs, resolving conflicts
    and eliminating redundancies while preserving all relevant information.
    
    Prompts are loaded from config/synthesis_prompts.yaml and the synthesis
    model is specified in model_config.yaml under synthesis.active_model.
    """
    
    def __init__(self, config_path: str, model_name: Optional[str] = None):
        """
        Initialize synthesizer with LLM client and prompt templates.
        
        Args:
            config_path: Path to model configuration YAML
            model_name: Optional model override (default: uses synthesis.active_model from config)
        """
        self.config_path = Path(config_path)
        self.base_dir = self.config_path.parent.parent
        
        # Load model configuration
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.model_config = yaml.safe_load(f)
        
        # Determine which model to use for synthesis
        if model_name is None:
            model_name = self.model_config.get('synthesis', {}).get('active_model')
            if not model_name:
                logger.warning(
                    "No synthesis.active_model found in config, "
                    "using default active_model"
                )
                model_name = self.model_config.get('active_model')
        
        # Initialize LLM client
        self.client = LLMClient(config_path=str(self.config_path), model_override=model_name)
        self.model_name = model_name
        
        # Load synthesis prompts
        self.synthesis_prompts = self._load_synthesis_prompts()
        
        logger.info(f"ChunkSynthesizer initialized with model: {self.model_name}")
    
    def _load_synthesis_prompts(self) -> Dict[str, Dict[str, str]]:
        """
        Load synthesis prompt templates from synthesis_prompts.yaml.
        
        Returns:
            Dictionary mapping task names to their synthesis prompts
        """
        prompts_path = self.base_dir / "config" / "synthesis_prompts.yaml"
        
        if not prompts_path.exists():
            logger.error(f"Synthesis prompts file not found: {prompts_path}")
            return {}
        
        try:
            with open(prompts_path, 'r', encoding='utf-8') as f:
                prompts = yaml.safe_load(f)
            
            logger.info(
                f"Loaded synthesis prompts for {len(prompts)} tasks: "
                f"{list(prompts.keys())}"
            )
            return prompts
        
        except Exception as e:
            logger.error(f"Failed to load synthesis prompts: {e}")
            return {}
    
    def synthesize_chunks(
        self, 
        task_name: str, 
        chunk_outputs: List[str]
    ) -> Optional[str]:
        """
        Synthesize multiple chunk outputs into a single unified result.
        
        Args:
            task_name: Name of the task (e.g., 'psychological_profiling')
            chunk_outputs: List of JSON strings from each chunk
        
        Returns:
            Unified JSON string, or None if synthesis fails
        """
        # Skip synthesis for single-chunk cases
        if len(chunk_outputs) <= 1:
            return chunk_outputs[0] if chunk_outputs else None
        
        # Check if this task requires synthesis
        if task_name not in self.synthesis_prompts:
            logger.warning(
                f"No synthesis prompt defined for '{task_name}', "
                f"concatenating outputs"
            )
            return self._simple_concatenate(chunk_outputs)
        
        # Get synthesis prompts
        prompts = self.synthesis_prompts[task_name]
        system_prompt = prompts.get('system', '')
        user_template = prompts.get('user_template', '')
        
        if not system_prompt or not user_template:
            logger.error(
                f"Incomplete synthesis prompts for '{task_name}', "
                f"falling back to concatenation"
            )
            return self._simple_concatenate(chunk_outputs)
        
        # Format chunk outputs for synthesis
        formatted_chunks = self._format_chunk_outputs(chunk_outputs)
        user_message = user_template.replace("{chunk_outputs}", formatted_chunks)
        
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        logger.info(
            f"Synthesizing {len(chunk_outputs)} chunks for '{task_name}' "
            f"using model '{self.model_name}' (~{len(formatted_chunks)} chars)"
        )
        
        try:
            # Call LLM for synthesis
            response = self.client.generate_response(messages)
            
            if isinstance(response, dict):
                synthesized = response.get('content', '')
            else:
                synthesized = response
            
            # Validate JSON
            try:
                json.loads(synthesized)
                logger.info(f"Synthesis successful for '{task_name}'")
                return synthesized
            except json.JSONDecodeError:
                logger.error(f"Synthesis produced invalid JSON for '{task_name}'")
                return self._simple_concatenate(chunk_outputs)
        
        except Exception as e:
            logger.error(f"Synthesis failed for '{task_name}': {e}")
            return self._simple_concatenate(chunk_outputs)
    
    def _format_chunk_outputs(self, chunk_outputs: List[str]) -> str:
        """
        Format chunk outputs for synthesis prompt.
        
        Args:
            chunk_outputs: List of JSON strings
        
        Returns:
            Formatted string for prompt
        """
        formatted_parts = []
        
        for idx, output in enumerate(chunk_outputs, 1):
            formatted_parts.append(f"=== CHUNK {idx} ===\n{output}\n")
        
        return "\n".join(formatted_parts)
    
    def _simple_concatenate(self, chunk_outputs: List[str]) -> str:
        """
        Fallback: simple array concatenation when synthesis is unavailable.
        
        Args:
            chunk_outputs: List of JSON strings
        
        Returns:
            Concatenated JSON array string
        """
        all_items = []
        
        for output in chunk_outputs:
            try:
                parsed = json.loads(output)
                if isinstance(parsed, list):
                    all_items.extend(parsed)
                elif isinstance(parsed, dict):
                    all_items.append(parsed)
            except json.JSONDecodeError:
                continue
        
        return json.dumps(all_items, indent=2, ensure_ascii=False)


def synthesize_task_outputs(
    base_dir: Path,
    task_name: str,
    model_name: str,
    config_path: str,
    synthesizer_model: Optional[str] = None
):
    """
    Batch synthesize all chunked outputs for a specific task and model.
    
    Scans the output directory for chunked results and applies synthesis
    where multiple chunks exist for the same chat.
    
    Args:
        base_dir: Base directory of the project
        task_name: Name of the task to synthesize
        model_name: Model name whose outputs to synthesize
        config_path: Path to model configuration
        synthesizer_model: Optional model to use for synthesis
    """
    synthesizer = ChunkSynthesizer(config_path, synthesizer_model)
    
    # Locate task output directory
    task_dir = base_dir / "data" / "outputs" / task_name / model_name
    
    if not task_dir.exists():
        logger.warning(f"Task directory not found: {task_dir}")
        return
    
    # Process each group
    for group_dir in task_dir.iterdir():
        if not group_dir.is_dir():
            continue
        
        # Process each chat
        for chat_file in group_dir.glob("*.json"):
            chat_id = chat_file.stem
            
            # Check for chunk markers in output
            with open(chat_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                data = json.loads(content)
                
                # Detect if this is a chunked output
                if isinstance(data, list) and len(data) > 1:
                    # Treat each element as a chunk output
                    chunk_outputs = [json.dumps(item) for item in data]
                    
                    synthesized = synthesizer.synthesize_chunks(
                        task_name, chunk_outputs
                    )
                    
                    if synthesized:
                        # Write synthesized output
                        with open(chat_file, 'w', encoding='utf-8') as f:
                            f.write(synthesized)
                        
                        logger.info(f"Synthesized {chat_id}")
            
            except json.JSONDecodeError:
                continue
