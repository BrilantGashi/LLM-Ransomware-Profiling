import argparse
import logging
import sys
from pathlib import Path

# Add src to python path to ensure imports work correctly
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))

from src.analysis.agentic_consensus import AgenticConsensusManager

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/consensus_pipeline.log", mode='a')
    ]
)

logger = logging.getLogger("ConsensusRunner")

def main():
    parser = argparse.ArgumentParser(description="Run Agentic Consensus Pipeline")
    parser.add_argument("--task", type=str, choices=['speech_act_analysis', 'psychological_profiling', 'tactical_extraction', 'all'], default='all', help="Specific task to process")
    parser.add_argument("--model", type=str, default="phi4", help="Model to use for the consensus agent (adjudicator)")
    args = parser.parse_args()

    # Define models to aggregate (should match your pipeline configuration)
    source_models = ['qwen3', 'phi4-mini', 'llama3.2'] 
    
    tasks = [args.task] if args.task != 'all' else [
        'speech_act_analysis', 
        'psychological_profiling', 
        'tactical_extraction'
    ]

    manager = AgenticConsensusManager(project_root, consensus_model=args.model)

    logger.info("=" * 60)
    logger.info(f"STARTING AGENTIC CONSENSUS PIPELINE")
    logger.info(f"Adjudicator Model: {args.model}")
    logger.info(f"Target Tasks: {tasks}")
    logger.info("=" * 60)

    for task in tasks:
        logger.info(f"--- Task: {task} ---")
        
        # Discover chats (scan the first model's directory as a baseline)
        # Note: We scan all output directories to build a comprehensive set of chat IDs
        chat_registry = set()
        base_outputs = project_root / "data" / "outputs" / task
        
        if not base_outputs.exists():
            logger.warning(f"Output directory for task {task} does not exist. Skipping.")
            continue

        for model_dir in base_outputs.iterdir():
            if model_dir.is_dir():
                for group_dir in model_dir.iterdir():
                    if group_dir.is_dir():
                        for chat_file in group_dir.glob("*.json"):
                            chat_registry.add((group_dir.name, chat_file.stem))
        
        total_chats = len(chat_registry)
        if total_chats == 0:
            logger.warning(f"No chats found for task {task}.")
            continue

        success_count = 0
        
        for idx, (group, chat_id) in enumerate(sorted(chat_registry), 1):
            try:
                if manager.run_consensus_for_chat(task, group, chat_id, source_models):
                    success_count += 1
            except Exception as e:
                logger.error(f"Unexpected error processing {chat_id}: {e}")

        logger.info(f"Task {task} completed: {success_count}/{total_chats} chats processed successfully.\n")

    logger.info("Consensus pipeline execution finished.")

if __name__ == "__main__":
    main()
