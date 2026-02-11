"""
Consensus Management Module for Multi-Model Validation
Implements majority voting across multiple LLM outputs for gold standard generation.

Author: Brilant Gashi
Supervisors: Prof. Federico Cerutti, Prof. Pietro Baroni
Institution: University of Brescia
"""

import json
import yaml
import logging
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Union

logger = logging.getLogger(__name__)


class ConsensusManager:
    """Multi-model consensus aggregator for ALL analysis tasks."""
    
    def __init__(
        self,
        base_dir: Path,
        task_name: str = 'speech_act_analysis',
        min_models: int = 2,
        min_confidence: float = 0.5
    ):
        """Initialize consensus manager with quality thresholds."""
        # Validation
        if min_models < 2:
            raise ValueError(f"min_models must be >= 2, got {min_models}")
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")
        
        valid_tasks = ['speech_act_analysis', 'tactical_extraction', 'psychological_profiling']
        if task_name not in valid_tasks:
            raise ValueError(f"task_name must be one of {valid_tasks}, got {task_name}")
        
        self.base_dir = base_dir
        self.task_name = task_name
        self.min_models = min_models
        self.min_confidence = min_confidence
        
        # Directory structure
        self.outputs_dir = base_dir / "data" / "outputs" / task_name
        self.consensus_dir = base_dir / "data" / "consensus" / task_name
        self.consensus_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'chats_processed': 0,
            'items_processed': 0,
            'low_confidence_items': 0,
            'tie_cases': 0,
            'missing_annotations': 0,
            'failed_chats': 0,
            'confidence_scores': [],
            'models_per_chat': []
        }
    
    def load_model_outputs(
        self,
        group: str,
        chat_id: str,
        models: List[str]
    ) -> Dict[str, Union[List[Dict], Dict]]:
        """Load annotations from multiple models for a specific chat."""
        model_data = {}
        
        for model in models:
            file_path = self.outputs_dir / model / group / f"{chat_id}.json"
            
            if not file_path.exists():
                self.stats['missing_annotations'] += 1
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Validate based on task type
                if self.task_name == 'speech_act_analysis':
                    if not isinstance(data, list) or len(data) == 0:
                        continue
                else:
                    if not isinstance(data, dict):
                        continue
                
                model_data[model] = data
                
            except (json.JSONDecodeError, Exception):
                self.stats['missing_annotations'] += 1
        
        return model_data
    
    def compute_consensus(
        self,
        model_data: Dict[str, Union[List[Dict], Dict]],
        chat_id: str
    ) -> Tuple[Union[List[Dict], Dict], Dict[str, Any]]:
        """Apply majority voting to create consensus output."""
        if not model_data:
            return ([] if self.task_name == 'speech_act_analysis' else {}), {}
        
        if self.task_name == 'speech_act_analysis':
            return self._compute_consensus_messages(model_data, chat_id)
        else:
            return self._compute_consensus_fields(model_data, chat_id)
    
    def _compute_consensus_messages(
        self,
        model_data: Dict[str, List[Dict]],
        chat_id: str
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """Compute consensus for speech act analysis (message-level)."""
        models = list(model_data.keys())
        max_len = max(len(data) for data in model_data.values())
        
        if max_len == 0:
            return [], {}
        
        consensus_chat = []
        message_confidences = []
        tie_count = 0
        low_conf_count = 0
        
        for i in range(max_len):
            votes_primary = []
            votes_arg = []
            base_msg = None
            
            for model in models:
                if i < len(model_data[model]):
                    msg = model_data[model][i]
                    if base_msg is None:
                        base_msg = msg.copy()
                    votes_primary.append(msg.get('primary_act', 'MISSING_LABEL'))
                    votes_arg.append(msg.get('argumentative_function', 'MISSING_LABEL'))
                else:
                    votes_primary.append("MISSING_MSG")
                    votes_arg.append("MISSING_MSG")
            
            if base_msg is None:
                continue
            
            final_primary, score_p, meta_p = self._resolve_vote(votes_primary)
            final_arg, score_a, meta_a = self._resolve_vote(votes_arg)
            
            avg_score = (score_p + score_a) / 2
            message_confidences.append(avg_score)
            
            if meta_p['is_tie'] or meta_a['is_tie']:
                tie_count += 1
            
            if avg_score < self.min_confidence:
                low_conf_count += 1
            
            base_msg['primary_act'] = final_primary
            base_msg['argumentative_function'] = final_arg
            base_msg['consensus_score'] = round(avg_score, 3)
            
            consensus_chat.append(base_msg)
        
        summary = {
            'total_messages': len(consensus_chat),
            'avg_confidence': round(sum(message_confidences) / len(message_confidences), 3) if message_confidences else 0.0,
            'min_confidence': round(min(message_confidences), 3) if message_confidences else 0.0,
            'max_confidence': round(max(message_confidences), 3) if message_confidences else 0.0,
            'tie_cases': tie_count,
            'low_confidence_cases': low_conf_count,
            'models_used': len(models)
        }
        
        self.stats['items_processed'] += len(consensus_chat)
        self.stats['tie_cases'] += tie_count
        self.stats['low_confidence_items'] += low_conf_count
        self.stats['confidence_scores'].extend(message_confidences)
        self.stats['models_per_chat'].append(len(models))
        
        return consensus_chat, summary
    
    def _compute_consensus_fields(
        self,
        model_data: Dict[str, Dict],
        chat_id: str
    ) -> Tuple[Dict, Dict[str, Any]]:
        """Compute consensus for tactical/psychological (field-level)."""
        models = list(model_data.keys())
        
        all_paths = set()
        for data in model_data.values():
            all_paths.update(self._get_all_paths(data))
        
        consensus_dict = {}
        field_confidences = []
        tie_count = 0
        low_conf_count = 0
        
        for path in sorted(all_paths):
            votes = []
            for model in models:
                value = self._get_nested_value(model_data[model], path)
                votes.append(value)
            
            final_value, confidence, meta = self._resolve_vote_values(votes)
            field_confidences.append(confidence)
            
            if meta['is_tie']:
                tie_count += 1
            if confidence < self.min_confidence:
                low_conf_count += 1
            
            self._set_nested_value(consensus_dict, path, final_value)
        
        summary = {
            'total_fields': len(all_paths),
            'avg_confidence': round(sum(field_confidences) / len(field_confidences), 3) if field_confidences else 0.0,
            'min_confidence': round(min(field_confidences), 3) if field_confidences else 0.0,
            'max_confidence': round(max(field_confidences), 3) if field_confidences else 0.0,
            'tie_cases': tie_count,
            'low_confidence_cases': low_conf_count,
            'models_used': len(models)
        }
        
        self.stats['items_processed'] += len(all_paths)
        self.stats['tie_cases'] += tie_count
        self.stats['low_confidence_items'] += low_conf_count
        self.stats['confidence_scores'].extend(field_confidences)
        self.stats['models_per_chat'].append(len(models))
        
        return consensus_dict, summary
    
    def _get_all_paths(self, d: Dict, parent_path: str = '') -> List[str]:
        """Extract all nested paths from dictionary."""
        paths = []
        for key, value in d.items():
            current_path = f"{parent_path}.{key}" if parent_path else key
            if isinstance(value, dict):
                paths.extend(self._get_all_paths(value, current_path))
            else:
                paths.append(current_path)
        return paths
    
    def _get_nested_value(self, d: Dict, path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = path.split('.')
        value = d
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return "MISSING_VALUE"
    
    def _set_nested_value(self, d: Dict, path: str, value: Any):
        """Set value in nested dictionary using dot notation."""
        keys = path.split('.')
        current = d
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    def _resolve_vote(self, votes: List[str]) -> Tuple[str, float, Dict[str, Any]]:
        """Resolve majority vote for string labels."""
        valid_votes = [v for v in votes if v not in ["MISSING_MSG", "MISSING_LABEL", None, ""]]
        
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
        tied_labels = sorted([label for label, count in counts.items() if count == max_count])
        
        is_tie = len(tied_labels) > 1
        winner = tied_labels[0]
        confidence = round(max_count / len(valid_votes), 3)
        
        metadata = {
            'valid_votes': len(valid_votes),
            'missing_votes': len(votes) - len(valid_votes),
            'distribution': dict(counts),
            'is_tie': is_tie,
            'tied_labels': tied_labels if is_tie else []
        }
        
        return winner, confidence, metadata
    
    def _resolve_vote_values(self, votes: List[Any]) -> Tuple[Any, float, Dict[str, Any]]:
        """Resolve majority vote for any value type."""
        valid_votes = [v for v in votes if v != "MISSING_VALUE" and v is not None]
        
        if not valid_votes:
            return None, 0.0, {'valid_votes': 0, 'missing_votes': len(votes), 'distribution': {}, 'is_tie': False}
        
        hashable_votes = []
        for v in valid_votes:
            if isinstance(v, list):
                hashable_votes.append(tuple(v))
            else:
                hashable_votes.append(v)
        
        counts = Counter(hashable_votes)
        max_count = max(counts.values())
        tied_values = sorted([val for val, count in counts.items() if count == max_count], key=str)
        
        is_tie = len(tied_values) > 1
        winner = tied_values[0]
        
        if isinstance(winner, tuple):
            winner = list(winner)
        
        confidence = round(max_count / len(valid_votes), 3)
        
        metadata = {
            'valid_votes': len(valid_votes),
            'missing_votes': len(votes) - len(valid_votes),
            'distribution': {str(k): v for k, v in counts.items()},
            'is_tie': is_tie
        }
        
        return winner, confidence, metadata
    
    def run_consensus_pipeline(self, group: str, chat_id: str, models: List[str]) -> bool:
        """Execute complete consensus workflow for a single chat."""
        data_map = self.load_model_outputs(group, chat_id, models)
        
        if not data_map:
            self.stats['failed_chats'] += 1
            return False
        
        try:
            gold_data, summary = self.compute_consensus(data_map, chat_id)
        except Exception:
            self.stats['failed_chats'] += 1
            return False
        
        is_empty = (
            (isinstance(gold_data, list) and not gold_data) or
            (isinstance(gold_data, dict) and not gold_data)
        )
        
        if is_empty:
            self.stats['failed_chats'] += 1
            return False
        
        out_group_dir = self.consensus_dir / group
        out_group_dir.mkdir(parents=True, exist_ok=True)
        
        out_path = out_group_dir / f"{chat_id}.json"
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(gold_data, f, indent=2, ensure_ascii=False)
        except Exception:
            self.stats['failed_chats'] += 1
            return False
        
        self.stats['chats_processed'] += 1
        return True
    
    def get_consensus_stats(self) -> Dict[str, Any]:
        """Generate comprehensive statistics about the consensus process."""
        conf_scores = self.stats['confidence_scores']
        
        return {
            'task_name': self.task_name,
            'chats_processed': self.stats['chats_processed'],
            'failed_chats': self.stats['failed_chats'],
            'items_processed': self.stats['items_processed'],
            'avg_confidence': round(sum(conf_scores) / len(conf_scores), 3) if conf_scores else 0.0,
            'min_confidence': round(min(conf_scores), 3) if conf_scores else 0.0,
            'max_confidence': round(max(conf_scores), 3) if conf_scores else 0.0,
            'low_confidence_items': self.stats['low_confidence_items'],
            'tie_cases': self.stats['tie_cases'],
            'missing_annotations': self.stats['missing_annotations']
        }


# =============================================================================
# BATCH PROCESSING UTILITIES
# =============================================================================

def load_models_from_config(project_root: Path) -> List[str]:
    """Load ensemble models from model_config.yaml"""
    config_path = project_root / "config" / "model_config.yaml"
    
    if not config_path.exists():
        return []
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        models = config.get('ensemble_models', [])
        if not models:
            active = config.get('active_model')
            if active:
                models = [active]
        
        return models
    except Exception:
        return []


def find_all_chats(outputs_dir: Path, models: List[str]) -> Dict[str, set]:
    """Find all chat_ids that exist across models."""
    chats_by_group = defaultdict(set)
    
    first_model_dir = outputs_dir / models[0]
    
    if not first_model_dir.exists():
        return {}
    
    for group_dir in first_model_dir.iterdir():
        if not group_dir.is_dir():
            continue
        
        group_name = group_dir.name
        for json_file in group_dir.glob("*.json"):
            chat_id = json_file.stem
            chats_by_group[group_name].add(chat_id)
    
    return chats_by_group


def run_batch_consensus(project_root: Path, tasks: List[str]):
    """Run consensus for all tasks on all available chats."""
    print("\n" + "=" * 70)
    print("🎯 BATCH CONSENSUS GENERATION")
    print("=" * 70)
    
    models = load_models_from_config(project_root)
    
    if not models:
        print("❌ No models found in config")
        return
    
    print(f"🤖 Models: {', '.join(models)}")
    print(f"📋 Tasks:  {', '.join(tasks)}\n")
    
    for task in tasks:
        print(f"📌 {task}")
        
        outputs_dir = project_root / "data" / "outputs" / task
        
        if not outputs_dir.exists():
            print(f"   ⚠️  No outputs directory\n")
            continue
        
        chats_by_group = find_all_chats(outputs_dir, models)
        
        if not chats_by_group:
            print(f"   ⚠️  No chats found\n")
            continue
        
        total_chats = sum(len(chats) for chats in chats_by_group.values())
        
        cm = ConsensusManager(
            base_dir=project_root,
            task_name=task,
            min_models=2,
            min_confidence=0.5
        )
        
        success_count = 0
        processed = 0
        
        for group_name, chat_ids in chats_by_group.items():
            for chat_id in sorted(chat_ids):
                processed += 1
                
                # Progress indicator (every 10%)
                if processed % max(1, total_chats // 10) == 0 or processed == total_chats:
                    pct = int((processed / total_chats) * 100)
                    print(f"   [{pct:3d}%] {processed}/{total_chats}", end='\r', flush=True)
                
                try:
                    if cm.run_consensus_pipeline(group_name, chat_id, models):
                        success_count += 1
                except Exception:
                    pass
        
        # Final stats
        stats = cm.get_consensus_stats()
        print(f"   ✅ {success_count}/{total_chats} chats (conf: {stats['avg_confidence']:.1%})   \n")
    
    print("=" * 70)
    print("✅ CONSENSUS COMPLETE")
    print("=" * 70 + "\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Minimal logging (only errors to console)
    logging.basicConfig(
        level=logging.ERROR,
        format='%(message)s'
    )
    
    project_root = Path(__file__).parent.parent.parent
    
    tasks = [
        'speech_act_analysis',
        'tactical_extraction',
        'psychological_profiling'
    ]
    
    run_batch_consensus(project_root, tasks)
