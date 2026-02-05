import json
import logging
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Tuple


logger = logging.getLogger(__name__)


class ConsensusManager:
    """
    Manages the aggregation of outputs from multiple LLM models to create a 'Gold Standard' dataset
    via Majority Voting. Handles missing messages and provides transparent tie-breaking with detailed metadata.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        # Input directory: Raw outputs per model
        self.outputs_dir = base_dir / "data" / "outputs" / "speech_act_analysis"
        
        # Output directory: Consensus data
        self.consensus_dir = base_dir / "data" / "consensus" / "speech_act_analysis"
        self.consensus_dir.mkdir(parents=True, exist_ok=True)

    def load_model_outputs(self, group: str, chat_id: str, models: List[str]) -> Dict[str, List[Dict]]:
        """
        Loads annotations from multiple models for a specific chat.
        Path: data/outputs/speech_act_analysis/MODEL/GROUP/CHAT_ID.json
        """
        model_data = {}
        for model in models:
            file_path = self.outputs_dir / model / group / f"{chat_id}.json"
            
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            model_data[model] = data
                        else:
                            logger.warning(f"Invalid format (expected list) for {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
            else:
                logger.warning(f"Missing annotation file for model {model}: {file_path}")
        
        return model_data

    def compute_consensus(self, model_data: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Applies Majority Voting to align outputs across models.
        Robust against differing message counts and provides detailed consensus metadata.
        """
        if not model_data:
            return []

        models = list(model_data.keys())
        
        # Determine the maximum chat length among all models to avoid truncation
        max_len = max(len(data) for data in model_data.values())

        consensus_chat = []

        for i in range(max_len):
            votes_primary = []
            votes_arg = []
            
            # Use the first available message content as the base for the consensus object
            base_msg = None
            
            # --- 1. VOTE COLLECTION ---
            for model in models:
                if i < len(model_data[model]):
                    msg = model_data[model][i]
                    if base_msg is None:
                        base_msg = msg.copy()
                    
                    votes_primary.append(msg.get('primary_act', 'MISSING_LABEL'))
                    votes_arg.append(msg.get('argumentative_function', 'MISSING_LABEL'))
                else:
                    # This model has fewer messages than others
                    votes_primary.append("MISSING_MSG")
                    votes_arg.append("MISSING_MSG")

            if base_msg is None:
                continue

            # --- 2. CONSENSUS CALCULATION ---
            final_primary, score_p, meta_p = self._resolve_vote(votes_primary)
            final_arg, score_a, meta_a = self._resolve_vote(votes_arg)

            # --- 3. CONSTRUCT RESULT WITH COMPREHENSIVE METADATA ---
            base_msg['primary_act'] = final_primary
            base_msg['argumentative_function'] = final_arg
            
            # Inject detailed metadata for both classification dimensions
            base_msg['consensus_meta'] = {
                'total_models': len(models),
                
                'primary_act': {
                    'label': final_primary,
                    'score': score_p,
                    'valid_votes': meta_p['valid_votes'],
                    'missing_votes': meta_p['missing_votes'],
                    'distribution': meta_p['distribution'],
                    'is_tie': meta_p['is_tie'],
                    'tied_labels': meta_p.get('tied_labels', [])
                },
                
                'argumentative_function': {
                    'label': final_arg,
                    'score': score_a,
                    'valid_votes': meta_a['valid_votes'],
                    'missing_votes': meta_a['missing_votes'],
                    'distribution': meta_a['distribution'],
                    'is_tie': meta_a['is_tie'],
                    'tied_labels': meta_a.get('tied_labels', [])
                }
            }

            consensus_chat.append(base_msg)
        
        return consensus_chat

    def _resolve_vote(self, votes: List[str]) -> Tuple[str, float, Dict[str, Any]]:
        """
        Resolves the majority vote with deterministic tie-breaking.
        Returns: (Winner Label, Confidence Score, Metadata)
        
        Tie-breaking strategy: Lexicographic ordering (alphabetical ascending)
        Missing votes ('MISSING_MSG', 'MISSING_LABEL') are excluded from scoring.
        """
        # Filter out invalid/missing votes
        valid_votes = [v for v in votes if v not in ["MISSING_MSG", "MISSING_LABEL"]]
        
        if not valid_votes:
            return "Unknown", 0.0, {
                'valid_votes': 0,
                'missing_votes': len(votes),
                'distribution': {},
                'is_tie': False,
                'tied_labels': []
            }

        counts = Counter(valid_votes)
        max_count = max(counts.values())
        
        # Deterministic tie-breaking: sort tied labels alphabetically
        tied_labels = sorted([label for label, count in counts.items() if count == max_count])
        
        is_tie = len(tied_labels) > 1
        winner = tied_labels[0]  # First in alphabetical order
        winner_count = max_count
        
        # Score: Votes for Winner / Total VALID Votes
        score = round(winner_count / len(valid_votes), 2)

        metadata = {
            'valid_votes': len(valid_votes),
            'missing_votes': len(votes) - len(valid_votes),
            'distribution': dict(counts),
            'is_tie': is_tie,
            'tied_labels': tied_labels if is_tie else []
        }
        
        return winner, score, metadata

    def run_consensus_pipeline(self, group: str, chat_id: str, models: List[str]):
        """
        Main entry point: Loads outputs, computes consensus, and saves results.
        """
        logger.info(f"Computing consensus for {chat_id}...")
        
        data_map = self.load_model_outputs(group, chat_id, models)
        
        if not data_map:
            logger.warning(f"No model outputs found for {chat_id}. Skipping.")
            return

        if len(data_map) < 2:
            logger.info(f"Note: Only {len(data_map)} model(s) found for {chat_id}. Consensus will match single source.")

        gold_data = self.compute_consensus(data_map)
        
        # Save to: data/consensus/speech_act_analysis/GROUP/CHAT_ID.json
        out_group_dir = self.consensus_dir / group
        out_group_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_group_dir / f"{chat_id}.json"
        
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(gold_data, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Consensus saved: {out_path}")
        except Exception as e:
            logger.error(f"Failed to write consensus file {out_path}: {e}")
