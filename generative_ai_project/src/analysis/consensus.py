"""
Consensus Management Module for Multi-Model Validation
Implements majority voting across multiple LLM outputs for gold standard generation.

This module aggregates predictions from ensemble models to create high-confidence
speech act classifications through democratic consensus mechanisms.

Author: Brilant Gashi
Supervisors: Prof. Federico Cerutti, Prof. Pietro Baroni
Institution: University of Brescia
"""

import json
import logging
from pathlib import Path
from collections import Counter
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class ConsensusManager:
    """
    Multi-model consensus aggregator for speech act classification.
    
    This class implements a majority voting system to combine predictions from
    multiple LLM models, creating a gold standard dataset with quality metrics.
    The system handles misaligned outputs, missing annotations, and provides
    transparent tie-breaking with comprehensive metadata.
    
    Key Features:
        - Majority voting with confidence scoring
        - Deterministic tie-breaking (alphabetical ordering)
        - Missing data handling (models with different message counts)
        - Quality thresholds for consensus validation
        - Comprehensive provenance tracking
        
    Consensus Algorithm:
        1. Collect votes from all available models
        2. Exclude missing/invalid votes from scoring
        3. Count votes per label (primary_act, argumentative_function)
        4. Select majority winner (ties broken alphabetically)
        5. Calculate confidence: winner_votes / total_valid_votes
        6. Generate metadata (distribution, ties, missing votes)
    
    Attributes:
        base_dir (Path): Project root directory
        outputs_dir (Path): Raw model outputs directory
        consensus_dir (Path): Gold standard output directory
        min_models (int): Minimum models required for valid consensus
        min_confidence (float): Minimum confidence threshold [0.0, 1.0]
        stats (Dict): Aggregated statistics tracker
        
    Example:
        >>> cm = ConsensusManager(project_root, min_models=3, min_confidence=0.6)
        >>> cm.run_consensus_pipeline('lockbit', 'chat_001', models=['phi4', 'qwen3', 'llama3'])
        ✅ Consensus saved: data/consensus/speech_act_analysis/lockbit/chat_001.json
        >>> stats = cm.get_consensus_stats()
        >>> print(f"Avg confidence: {stats['avg_confidence']:.2f}")
    """
    
    def __init__(
        self,
        base_dir: Path,
        min_models: int = 2,
        min_confidence: float = 0.5
    ):
        """
        Initialize consensus manager with quality thresholds.
        
        Args:
            base_dir: Project root directory
            min_models: Minimum models required for consensus (default: 2)
            min_confidence: Minimum confidence threshold 0.0-1.0 (default: 0.5)
            
        Raises:
            ValueError: If min_models < 2 or min_confidence not in [0, 1]
        """
        # Validation
        if min_models < 2:
            raise ValueError(f"min_models must be >= 2, got {min_models}")
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")
        
        self.base_dir = base_dir
        self.min_models = min_models
        self.min_confidence = min_confidence
        
        # Directory structure
        self.outputs_dir = base_dir / "data" / "outputs" / "speech_act_analysis"
        self.consensus_dir = base_dir / "data" / "consensus" / "speech_act_analysis"
        self.consensus_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'chats_processed': 0,
            'messages_processed': 0,
            'low_confidence_messages': 0,
            'tie_cases': 0,
            'missing_annotations': 0,
            'failed_chats': 0,
            'confidence_scores': [],
            'models_per_chat': []
        }
        
        logger.info(
            f"ConsensusManager initialized: min_models={min_models}, "
            f"min_confidence={min_confidence}"
        )
    
    def load_model_outputs(
        self,
        group: str,
        chat_id: str,
        models: List[str]
    ) -> Dict[str, List[Dict]]:
        """
        Load speech act annotations from multiple models for a specific chat.
        
        Path structure: data/outputs/speech_act_analysis/MODEL/GROUP/CHAT_ID.json
        
        Args:
            group: Ransomware group name
            chat_id: Unique chat identifier
            models: List of model names to load
            
        Returns:
            Dictionary mapping model names to their annotation lists
            
        Example:
            >>> cm = ConsensusManager(project_root)
            >>> data = cm.load_model_outputs('lockbit', 'chat_001', ['phi4', 'qwen3'])
            >>> print(data.keys())
            dict_keys(['phi4', 'qwen3'])
            >>> print(len(data['phi4']))  # Number of messages
            47
            
        Notes:
            - Invalid JSON files are skipped with warning
            - Missing files are logged but don't raise errors
            - Only returns models with valid data
        """
        model_data = {}
        
        for model in models:
            file_path = self.outputs_dir / model / group / f"{chat_id}.json"
            
            if not file_path.exists():
                logger.warning(
                    f"Missing annotation file for model '{model}': {file_path}"
                )
                self.stats['missing_annotations'] += 1
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Validate structure
                if not isinstance(data, list):
                    logger.warning(
                        f"Invalid format for {file_path}: expected list, got {type(data)}"
                    )
                    continue
                
                if len(data) == 0:
                    logger.warning(f"Empty annotation list in {file_path}")
                    continue
                
                # Validate message structure
                if not self._validate_annotation_structure(data[0], model, chat_id):
                    continue
                
                model_data[model] = data
                
            except json.JSONDecodeError as e:
                logger.error(
                    f"JSON decode error in {file_path}: {str(e)[:100]}"
                )
                self.stats['missing_annotations'] += 1
            except Exception as e:
                logger.error(
                    f"Unexpected error loading {file_path}: {e}"
                )
                self.stats['missing_annotations'] += 1
        
        return model_data
    
    def _validate_annotation_structure(
        self,
        message: Dict[str, Any],
        model: str,
        chat_id: str
    ) -> bool:
        """
        Validate that a message has required fields for consensus.
        
        Args:
            message: First message from annotation list
            model: Model name (for logging)
            chat_id: Chat ID (for logging)
            
        Returns:
            True if structure is valid, False otherwise
        """
        required_fields = ['primary_act', 'argumentative_function']
        
        for field in required_fields:
            if field not in message:
                logger.warning(
                    f"Missing required field '{field}' in {model}/{chat_id}"
                )
                return False
        
        return True
    
    def compute_consensus(
        self,
        model_data: Dict[str, List[Dict]],
        chat_id: str
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Apply majority voting to create consensus annotations.
        
        This method aligns outputs from multiple models, handles mismatched
        message counts, and generates comprehensive metadata for each decision.
        
        Args:
            model_data: Dictionary of model_name -> annotation_list
            chat_id: Chat identifier (for logging)
            
        Returns:
            Tuple of:
                - List of consensus messages with metadata
                - Summary statistics dictionary
                
        Algorithm:
            For each message position i:
                1. Collect votes from all models (if available)
                2. Exclude missing/invalid votes
                3. Count votes per label
                4. Select winner (majority + alphabetical tie-breaking)
                5. Calculate confidence score
                6. Attach comprehensive metadata
                
        Example:
            >>> consensus_msgs, summary = cm.compute_consensus(model_data, 'chat_001')
            >>> print(summary['avg_confidence'])
            0.78
            >>> print(consensus_msgs[0]['consensus_meta']['primary_act']['score'])
            0.86
            
        Notes:
            - Logs warnings for low-confidence predictions
            - Tracks tie cases in statistics
            - Handles variable message lengths across models
        """
        if not model_data:
            logger.warning(f"No model data provided for consensus: {chat_id}")
            return [], {}
        
        models = list(model_data.keys())
        
        # Check minimum models requirement
        if len(models) < self.min_models:
            logger.warning(
                f"Only {len(models)} models available for {chat_id}, "
                f"minimum required: {self.min_models}"
            )
        
        # Determine maximum chat length to avoid truncation
        max_len = max(len(data) for data in model_data.values())
        
        if max_len == 0:
            logger.warning(f"All models returned empty chat: {chat_id}")
            return [], {}
        
        consensus_chat = []
        message_confidences = []
        tie_count = 0
        low_conf_count = 0
        
        for i in range(max_len):
            votes_primary = []
            votes_arg = []
            base_msg = None
            
            # Collect votes from all models
            for model in models:
                if i < len(model_data[model]):
                    msg = model_data[model][i]
                    
                    # Use first available message as base
                    if base_msg is None:
                        base_msg = msg.copy()
                    
                    votes_primary.append(msg.get('primary_act', 'MISSING_LABEL'))
                    votes_arg.append(msg.get('argumentative_function', 'MISSING_LABEL'))
                else:
                    # Model has fewer messages
                    votes_primary.append("MISSING_MSG")
                    votes_arg.append("MISSING_MSG")
            
            if base_msg is None:
                logger.warning(
                    f"No base message found at position {i} for {chat_id}"
                )
                continue
            
            # Resolve consensus for both dimensions
            final_primary, score_p, meta_p = self._resolve_vote(votes_primary)
            final_arg, score_a, meta_a = self._resolve_vote(votes_arg)
            
            # Track statistics
            avg_score = (score_p + score_a) / 2
            message_confidences.append(avg_score)
            
            if meta_p['is_tie'] or meta_a['is_tie']:
                tie_count += 1
            
            if avg_score < self.min_confidence:
                low_conf_count += 1
                logger.warning(
                    f"Low confidence ({avg_score:.2f}) at message {i} in {chat_id}: "
                    f"primary={final_primary} (score={score_p:.2f}), "
                    f"arg={final_arg} (score={score_a:.2f})"
                )
            
            # Construct consensus message
            base_msg['primary_act'] = final_primary
            base_msg['argumentative_function'] = final_arg
            
            # Attach comprehensive metadata
            base_msg['consensus_meta'] = {
                'message_index': i,
                'total_models': len(models),
                'consensus_timestamp': datetime.now().isoformat(),
                
                'primary_act': {
                    'label': final_primary,
                    'confidence': score_p,
                    'valid_votes': meta_p['valid_votes'],
                    'missing_votes': meta_p['missing_votes'],
                    'distribution': meta_p['distribution'],
                    'is_tie': meta_p['is_tie'],
                    'tied_labels': meta_p.get('tied_labels', []),
                    'tie_resolution': 'alphabetical' if meta_p['is_tie'] else None
                },
                
                'argumentative_function': {
                    'label': final_arg,
                    'confidence': score_a,
                    'valid_votes': meta_a['valid_votes'],
                    'missing_votes': meta_a['missing_votes'],
                    'distribution': meta_a['distribution'],
                    'is_tie': meta_a['is_tie'],
                    'tied_labels': meta_a.get('tied_labels', []),
                    'tie_resolution': 'alphabetical' if meta_a['is_tie'] else None
                },
                
                'overall_confidence': round(avg_score, 3),
                'quality_flag': 'low' if avg_score < self.min_confidence else 'high'
            }
            
            consensus_chat.append(base_msg)
        
        # Generate summary statistics
        summary = {
            'total_messages': len(consensus_chat),
            'avg_confidence': round(sum(message_confidences) / len(message_confidences), 3) if message_confidences else 0.0,
            'min_confidence': round(min(message_confidences), 3) if message_confidences else 0.0,
            'max_confidence': round(max(message_confidences), 3) if message_confidences else 0.0,
            'tie_cases': tie_count,
            'low_confidence_cases': low_conf_count,
            'models_used': len(models)
        }
        
        # Update global statistics
        self.stats['messages_processed'] += len(consensus_chat)
        self.stats['tie_cases'] += tie_count
        self.stats['low_confidence_messages'] += low_conf_count
        self.stats['confidence_scores'].extend(message_confidences)
        self.stats['models_per_chat'].append(len(models))
        
        return consensus_chat, summary
    
    def _resolve_vote(
        self,
        votes: List[str]
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Resolve majority vote with deterministic tie-breaking.
        
        Tie-breaking Strategy:
            - Sort tied labels alphabetically (ascending)
            - Select first label (lexicographic ordering)
            - This ensures reproducibility across runs
        
        Args:
            votes: List of label predictions from all models
            
        Returns:
            Tuple of:
                - Winner label (str)
                - Confidence score (float in [0, 1])
                - Metadata dictionary with vote distribution
                
        Example:
            >>> votes = ['directive', 'directive', 'commissive', 'MISSING_MSG']
            >>> label, score, meta = cm._resolve_vote(votes)
            >>> print(label, score)
            directive 0.67  # 2/3 valid votes
            >>> print(meta['distribution'])
            {'directive': 2, 'commissive': 1}
            
        Notes:
            - Missing votes ('MISSING_MSG', 'MISSING_LABEL') excluded from scoring
            - Confidence = winner_votes / total_valid_votes
            - Returns 'Unknown' with 0.0 confidence if no valid votes
        """
        # Filter invalid votes
        valid_votes = [
            v for v in votes 
            if v not in ["MISSING_MSG", "MISSING_LABEL", None, ""]
        ]
        
        if not valid_votes:
            return "Unknown", 0.0, {
                'valid_votes': 0,
                'missing_votes': len(votes),
                'distribution': {},
                'is_tie': False,
                'tied_labels': []
            }
        
        # Count votes
        counts = Counter(valid_votes)
        max_count = max(counts.values())
        
        # Find all labels with maximum count (potential ties)
        tied_labels = sorted([
            label for label, count in counts.items() 
            if count == max_count
        ])
        
        is_tie = len(tied_labels) > 1
        
        # Deterministic tie-breaking: alphabetical order
        winner = tied_labels[0]
        winner_count = max_count
        
        # Calculate confidence score
        confidence = round(winner_count / len(valid_votes), 3)
        
        metadata = {
            'valid_votes': len(valid_votes),
            'missing_votes': len(votes) - len(valid_votes),
            'distribution': dict(counts),
            'is_tie': is_tie,
            'tied_labels': tied_labels if is_tie else []
        }
        
        return winner, confidence, metadata
    
    def run_consensus_pipeline(
        self,
        group: str,
        chat_id: str,
        models: List[str]
    ) -> bool:
        """
        Execute complete consensus workflow for a single chat.
        
        Workflow:
            1. Load annotations from all models
            2. Validate minimum model requirement
            3. Compute consensus with majority voting
            4. Generate comprehensive metadata
            5. Save consensus annotations and summary
            
        Args:
            group: Ransomware group name
            chat_id: Unique chat identifier
            models: List of model names to aggregate
            
        Returns:
            True if consensus successfully generated and saved, False otherwise
            
        Example:
            >>> cm = ConsensusManager(project_root)
            >>> success = cm.run_consensus_pipeline(
            ...     'lockbit', 
            ...     'chat_001', 
            ...     ['phi4', 'qwen3', 'llama3']
            ... )
            ✅ Consensus saved: data/consensus/speech_act_analysis/lockbit/chat_001.json
            >>> print(success)
            True
            
        Notes:
            - Creates output directory if it doesn't exist
            - Saves both consensus data and summary metadata
            - Updates global statistics tracker
            - Logs detailed information about consensus quality
        """
        logger.info(f"🔄 Computing consensus for {group}/{chat_id}...")
        
        # Load model outputs
        data_map = self.load_model_outputs(group, chat_id, models)
        
        if not data_map:
            logger.warning(
                f"❌ No model outputs found for {chat_id}. Skipping."
            )
            self.stats['failed_chats'] += 1
            return False
        
        if len(data_map) < self.min_models:
            logger.warning(
                f"⚠️  Only {len(data_map)} model(s) found for {chat_id}, "
                f"minimum required: {self.min_models}"
            )
        
        # Compute consensus
        try:
            gold_data, summary = self.compute_consensus(data_map, chat_id)
        except Exception as e:
            logger.error(f"❌ Consensus computation failed for {chat_id}: {e}")
            self.stats['failed_chats'] += 1
            return False
        
        if not gold_data:
            logger.warning(f"❌ Empty consensus generated for {chat_id}")
            self.stats['failed_chats'] += 1
            return False
        
        # Prepare output directory
        out_group_dir = self.consensus_dir / group
        out_group_dir.mkdir(parents=True, exist_ok=True)
        
        # Save consensus data
        out_path = out_group_dir / f"{chat_id}.json"
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(gold_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ Failed to write consensus file {out_path}: {e}")
            self.stats['failed_chats'] += 1
            return False
        
        # Save summary metadata
        summary_path = out_group_dir / f"{chat_id}_summary.json"
        summary_data = {
            'chat_id': chat_id,
            'group': group,
            'timestamp': datetime.now().isoformat(),
            'models': list(data_map.keys()),
            'statistics': summary,
            'consensus_config': {
                'min_models': self.min_models,
                'min_confidence': self.min_confidence,
                'tie_breaking': 'alphabetical'
            }
        }
        
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"⚠️  Failed to write summary file {summary_path}: {e}")
        
        # Log success with quality metrics
        logger.info(
            f"✅ Consensus saved: {out_path.name} "
            f"(confidence: {summary['avg_confidence']:.2f}, "
            f"messages: {summary['total_messages']}, "
            f"models: {summary['models_used']})"
        )
        
        self.stats['chats_processed'] += 1
        return True
    
    def get_consensus_stats(self) -> Dict[str, Any]:
        """
        Generate comprehensive statistics about the consensus process.
        
        Returns:
            Dictionary containing:
                Processing:
                    - chats_processed: Successfully processed chats
                    - failed_chats: Failed consensus attempts
                    - messages_processed: Total consensus messages
                Quality:
                    - avg_confidence: Mean confidence across all messages
                    - min_confidence: Lowest message confidence
                    - max_confidence: Highest message confidence
                    - low_confidence_messages: Count below threshold
                    - low_confidence_pct: Percentage below threshold
                Consensus:
                    - tie_cases: Number of tie situations
                    - tie_rate: Percentage of messages with ties
                    - avg_models_per_chat: Mean number of models used
                Data Coverage:
                    - missing_annotations: Missing model outputs
                    
        Example:
            >>> cm = ConsensusManager(project_root)
            >>> # ... run consensus pipeline ...
            >>> stats = cm.get_consensus_stats()
            >>> print(f"Quality: {stats['avg_confidence']:.1%}")
            Quality: 78.5%
            >>> print(f"Tie rate: {stats['tie_rate']:.1%}")
            Tie rate: 12.3%
            
        Notes:
            - Call after processing multiple chats
            - Useful for quality assessment and thesis reporting
            - Empty confidence_scores returns 0.0 for aggregates
        """
        conf_scores = self.stats['confidence_scores']
        
        return {
            # Processing metrics
            'chats_processed': self.stats['chats_processed'],
            'failed_chats': self.stats['failed_chats'],
            'messages_processed': self.stats['messages_processed'],
            
            # Quality metrics
            'avg_confidence': round(
                sum(conf_scores) / len(conf_scores), 3
            ) if conf_scores else 0.0,
            'min_confidence': round(min(conf_scores), 3) if conf_scores else 0.0,
            'max_confidence': round(max(conf_scores), 3) if conf_scores else 0.0,
            'low_confidence_messages': self.stats['low_confidence_messages'],
            'low_confidence_pct': round(
                (self.stats['low_confidence_messages'] / 
                 self.stats['messages_processed'] * 100), 2
            ) if self.stats['messages_processed'] > 0 else 0.0,
            
            # Consensus metrics
            'tie_cases': self.stats['tie_cases'],
            'tie_rate': round(
                (self.stats['tie_cases'] / 
                 self.stats['messages_processed'] * 100), 2
            ) if self.stats['messages_processed'] > 0 else 0.0,
            'avg_models_per_chat': round(
                sum(self.stats['models_per_chat']) / 
                len(self.stats['models_per_chat']), 2
            ) if self.stats['models_per_chat'] else 0.0,
            
            # Data coverage
            'missing_annotations': self.stats['missing_annotations']
        }
    
    def save_consensus_metadata(self, output_dir: Path) -> None:
        """
        Save metadata about the consensus process for reproducibility.
        
        Args:
            output_dir: Directory where metadata file will be saved
            
        Example:
            >>> cm = ConsensusManager(project_root)
            >>> # ... run consensus pipeline ...
            >>> cm.save_consensus_metadata(consensus_dir)
            💾 Saved consensus metadata: CONSENSUS_METADATA.json
            
        Notes:
            - Essential for research reproducibility
            - Documents consensus algorithm and parameters
            - Includes quality metrics for thesis reporting
        """
        metadata = {
            'metadata_timestamp': datetime.now().isoformat(),
            'consensus_version': '2.0.0',
            
            'algorithm': {
                'method': 'majority_voting',
                'tie_breaking': 'alphabetical_ascending',
                'missing_vote_handling': 'exclude_from_scoring'
            },
            
            'configuration': {
                'min_models': self.min_models,
                'min_confidence_threshold': self.min_confidence
            },
            
            'statistics': self.get_consensus_stats(),
            
            'directories': {
                'input': str(self.outputs_dir),
                'output': str(self.consensus_dir)
            },
            
            'thesis': {
                'student': 'Brilant Gashi',
                'institution': 'University of Brescia',
                'supervisors': ['Prof. Federico Cerutti', 'Prof. Pietro Baroni'],
                'year': '2025-2026'
            }
        }
        
        metadata_file = output_dir / 'CONSENSUS_METADATA.json'
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved consensus metadata: {metadata_file.name}")
        except Exception as e:
            logger.error(f"❌ Failed to save consensus metadata: {e}")


# Main execution
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "=" * 70)
    print("🎯 CONSENSUS GENERATION MODULE")
    print("=" * 70)
    
    # Configuration
    project_root = Path(__file__).parent.parent.parent
    models = ['phi4-mini', 'qwen3', 'llama3.2', 'gpt-oss']
    
    print(f"📂 Project Root: {project_root}")
    print(f"🤖 Models: {', '.join(models)}")
    print("-" * 70)
    
    # Initialize consensus manager
    cm = ConsensusManager(
        base_dir=project_root,
        min_models=3,
        min_confidence=0.6
    )
    
    # Example: Process a single chat
    success = cm.run_consensus_pipeline(
        group='lockbit',
        chat_id='example_chat_001',
        models=models
    )
    
    if success:
        print("\n✅ Consensus generation successful!")
        
        # Display statistics
        stats = cm.get_consensus_stats()
        print("\n" + "=" * 70)
        print("📊 CONSENSUS STATISTICS")
        print("=" * 70)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print("=" * 70)
        
        # Save metadata
        cm.save_consensus_metadata(cm.consensus_dir)
    else:
        print("\n❌ Consensus generation failed!")
    
    print()
