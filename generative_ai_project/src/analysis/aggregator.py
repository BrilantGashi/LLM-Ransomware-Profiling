"""
Data Aggregation Module for Ransomware Negotiation Analysis
Aggregates multi-source JSON outputs into structured Pandas DataFrames.

This module consolidates outputs from three analysis tasks:
- Tactical Extraction: Financial and technical negotiation indicators
- Psychological Profiling: Behavioral traits and communication patterns
- Speech Act Analysis: Consensus-based linguistic classifications

Author: Brilant Gashi
Supervisors: Prof. Federico Cerutti, Prof. Pietro Baroni
Institution: University of Brescia
"""

import json
import sys
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger(__name__)


class DataAggregator:
    """
    Multi-level data aggregator for ransomware negotiation analysis.
    
    This class implements a three-tier aggregation strategy:
    1. Chat-level: Merges tactical and psychological features per negotiation
    2. Message-level: Individual speech acts with temporal metadata
    3. Statistical: Temporal evolution and group attribution profiles
    
    The aggregator prioritizes consensus data when available and implements
    robust error handling to skip corrupted or malformed JSON files.
    
    Attributes:
        outputs_dir (Path): Root directory for model outputs
        consensus_dir (Path): Directory containing consensus-validated results
        tactical_dir (Path): Tactical extraction outputs directory
        profiling_dir (Path): Psychological profiling outputs directory
        speech_dir (Path): Speech act analysis (consensus preferred)
        
    Example:
        >>> agg = DataAggregator(project_root)
        >>> df_negotiations = agg.aggregate_negotiations()
        >>> df_speech = agg.aggregate_speech_acts()
        >>> stats = agg.get_aggregation_stats()
    """
    
    def __init__(self, base_dir: Path):
        """
        Initialize aggregator with project directory structure.
        
        Args:
            base_dir: Project root directory containing data/ subdirectories
        """
        self.outputs_dir = base_dir / "data" / "outputs"
        self.consensus_dir = base_dir / "data" / "consensus"
        
        # Define specific source directories
        self.tactical_dir = self.outputs_dir / "tactical_extraction"
        self.profiling_dir = self.outputs_dir / "psychological_profiling"
        
        # Prefer consensus data for speech acts (gold standard)
        self.speech_dir = self.consensus_dir / "speech_act_analysis"
        
        # Initialize statistics tracking
        self._stats = {
            'chats_processed': 0,
            'chats_with_psychological': 0,
            'corrupted_files': 0,
            'validation_failures': 0
        }
    
    def load_json_safe(self, file_path: Path) -> Dict[str, Any]:
        """
        Safely load JSON file with error handling.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON as dictionary, or empty dict on failure
            
        Notes:
            - Logs warnings for corrupted or missing files
            - Increments corrupted_files counter on failure
            - Does not raise exceptions (fail-safe design)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️  Invalid JSON in {file_path.name}: {str(e)[:50]}")
            self._stats['corrupted_files'] += 1
            return {}
        except FileNotFoundError:
            logger.warning(f"⚠️  File not found: {file_path.name}")
            self._stats['corrupted_files'] += 1
            return {}
        except Exception as e:
            logger.error(f"❌ Unexpected error loading {file_path.name}: {e}")
            self._stats['corrupted_files'] += 1
            return {}
    
    def validate_negotiation_record(self, record: Dict[str, Any]) -> bool:
        """
        Validate negotiation record against schema requirements.
        
        Performs type checking and range validation on critical fields:
        - Required fields: chat_id, group
        - Type validation: Numeric fields must be int/float
        - Range validation: Percentages in [0, 100], non-negative values
        
        Args:
            record: Dictionary containing aggregated negotiation data
            
        Returns:
            True if record passes all validations, False otherwise
            
        Notes:
            - Validation failures are logged as warnings
            - Increments validation_failures counter on failure
            - Designed for defensive data quality assurance
        """
        # Check required fields
        required_fields = ['chat_id', 'group']
        if not all(field in record for field in required_fields):
            logger.warning(
                f"Missing required fields in {record.get('chat_id', 'unknown')}"
            )
            self._stats['validation_failures'] += 1
            return False
        
        # Validate financial data types
        numeric_fields = ['initial_demand', 'final_price', 'discount_pct']
        for field in numeric_fields:
            if record.get(field) is not None:
                if not isinstance(record[field], (int, float)):
                    logger.warning(
                        f"Invalid type for {field} in {record['chat_id']}: "
                        f"{type(record[field])}"
                    )
                    self._stats['validation_failures'] += 1
                    return False
        
        # Validate ranges
        if record.get('discount_pct') is not None:
            if not (0 <= record['discount_pct'] <= 100):
                logger.warning(
                    f"Discount percentage out of range [0,100] in {record['chat_id']}: "
                    f"{record['discount_pct']}"
                )
                self._stats['validation_failures'] += 1
                return False
        
        # Validate non-negative values
        if record.get('data_volume_gb') is not None:
            if record['data_volume_gb'] < 0:
                logger.warning(
                    f"Negative data volume in {record['chat_id']}: "
                    f"{record['data_volume_gb']}"
                )
                self._stats['validation_failures'] += 1
                return False
        
        return True
    
    def aggregate_negotiations(self) -> pd.DataFrame:
        """
        Aggregate chat-level features from tactical and psychological analysis.
        
        This method performs a left join between tactical extraction outputs
        (financial indicators, technical metadata, negotiation dynamics) and
        psychological profiling outputs (behavioral traits, communication patterns).
        
        Returns:
            pd.DataFrame: Chat-level dataset with columns:
                Identifiers:
                    - chat_id: Unique conversation identifier
                    - group: Ransomware group name
                Financial:
                    - initial_demand: Opening ransom amount
                    - final_price: Agreed payment (if negotiation succeeded)
                    - discount_pct: Percentage discount from initial demand
                    - currency: Payment currency (BTC, USD, etc.)
                Technical:
                    - victim_size: Organization size category
                    - attack_type: Type of ransomware attack
                    - data_volume_gb: Exfiltrated data volume
                    - exfiltration_confirmed: Boolean flag
                Psychological:
                    - attacker_tone: Communication style (aggressive, professional, etc.)
                    - attacker_competence: Skill level assessment
                    - attacker_strategy: Primary negotiation strategy
                    - influence_tactics: Cialdini's influence tactics (comma-separated)
                    - victim_emotion: Emotional trajectory
                    - victim_strategy: Primary negotiation tactic
                    - victim_effectiveness: Tactic effectiveness rating
                Dynamics:
                    - outcome: Final status (paid, refused, abandoned, etc.)
                    - attacker_flexibility: Willingness to negotiate
        
        Raises:
            Warning (logged): If tactical_dir does not exist
            
        Example:
            >>> agg = DataAggregator(project_root)
            >>> df = agg.aggregate_negotiations()
            >>> print(df.shape)  # (156, 22)
            >>> print(df['initial_demand'].describe())
            
        Notes:
            - Missing psychological profiles result in NaN values (left join)
            - Corrupted JSON files are skipped with warning
            - Returns empty DataFrame if no valid data found
            - All records are validated before inclusion
        """
        data_records = []
        
        if not self.tactical_dir.exists():
            logger.warning(f"❌ Tactical directory not found at {self.tactical_dir}")
            return pd.DataFrame()
        
        # Iterate through ransomware group folders
        for group_dir in self.tactical_dir.iterdir():
            if not group_dir.is_dir():
                continue
            
            group_name = group_dir.name
            
            for tactical_file in group_dir.glob("*.json"):
                chat_id = tactical_file.stem
                
                # Load tactical extraction data
                tactical_data = self.load_json_safe(tactical_file)
                if not tactical_data:
                    continue
                
                # Extract nested fields safely
                meta = tactical_data.get("metadata", {})
                finance = tactical_data.get("financial_negotiation", {})
                tech = tactical_data.get("technical_indicators", {})
                dynamics = tactical_data.get("negotiation_dynamics", {})
                
                record = {
                    # Identifiers
                    "chat_id": chat_id,
                    "group": group_name,
                    
                    # Metadata
                    "victim_size": meta.get("victim_size"),
                    "attack_type": meta.get("attack_type"),
                    
                    # Financial indicators
                    "initial_demand": finance.get("initial_demand"),
                    "final_price": finance.get("final_agreed_price"),
                    "discount_pct": finance.get("discount_percentage"),
                    "currency": finance.get("currency"),
                    
                    # Technical indicators
                    "data_volume_gb": tech.get("data_volume_gb"),
                    "exfiltration_confirmed": tech.get("exfiltration_confirmed"),
                    
                    # Negotiation dynamics
                    "outcome": dynamics.get("outcome_status"),
                    "attacker_flexibility": dynamics.get("attacker_flexibility")
                }
                
                # Load psychological profiling data (left join)
                profile_file = self.profiling_dir / group_name / f"{chat_id}.json"
                if profile_file.exists():
                    profile_data = self.load_json_safe(profile_file)
                    
                    if profile_data:
                        att_prof = profile_data.get("attacker_profile", {})
                        vic_prof = profile_data.get("victim_profile", {})
                        
                        record.update({
                            # Attacker psychological profile
                            "attacker_tone": att_prof.get("communication_tone"),
                            "attacker_competence": att_prof.get("competence_level"),
                            "attacker_strategy": att_prof.get(
                                "primary_strategy_dual_concern"
                            ),
                            "influence_tactics": ", ".join(
                                att_prof.get("cialdini_influence_tactics", [])
                            ),
                            
                            # Victim psychological profile
                            "victim_emotion": vic_prof.get("emotional_trajectory"),
                            "victim_strategy": vic_prof.get("primary_negotiation_tactic"),
                            "victim_effectiveness": vic_prof.get("tactic_effectiveness")
                        })
                        
                        self._stats['chats_with_psychological'] += 1
                
                # Validate record before adding
                if self.validate_negotiation_record(record):
                    data_records.append(record)
                    self._stats['chats_processed'] += 1
        
        df = pd.DataFrame(data_records)
        logger.info(
            f"✅ Aggregated {len(df)} negotiations "
            f"({self._stats['chats_with_psychological']} with psychological data)"
        )
        
        return df
    
    def aggregate_speech_acts(self) -> pd.DataFrame:
        """
        Create message-level dataset with speech act classifications.
        
        This method generates a granular DataFrame where each row represents
        a single message from a negotiation chat. It uses consensus-validated
        speech act classifications when available.
        
        Returns:
            pd.DataFrame: Message-level dataset with columns:
                Identifiers:
                    - chat_id: Parent conversation identifier
                    - group: Ransomware group name
                    - msg_index: Sequential message number (0-indexed)
                Temporal:
                    - progress: Normalized position in chat [0.0, 1.0]
                    - progress_bin: Discretized progress (1-20, 5% bins)
                    - phase: Negotiation phase (opening, middle, closing)
                Linguistic:
                    - party: Speaker role (attacker/victim)
                    - primary_act: Primary speech act (directive, commissive, etc.)
                    - argumentative_func: Argumentative function
                Quality:
                    - text_length: Character count of message
                    - consensus_score: Inter-model agreement [0.0, 1.0]
        
        Example:
            >>> agg = DataAggregator(project_root)
            >>> df = agg.aggregate_speech_acts()
            >>> print(df.groupby('primary_act').size())
            directive      1247
            commissive      856
            assertive       744
            ...
            
        Notes:
            - Requires consensus data directory to exist
            - Returns empty DataFrame if no speech act data found
            - Progress bins enable temporal evolution analysis
            - Consensus score indicates inter-model reliability
        """
        speech_records = []
        
        if not self.speech_dir.exists():
            logger.warning(
                f"❌ Speech acts directory (consensus) not found at {self.speech_dir}"
            )
            return pd.DataFrame()
        
        for group_dir in self.speech_dir.iterdir():
            if not group_dir.is_dir():
                continue
            
            group_name = group_dir.name
            
            for speech_file in group_dir.glob("*.json"):
                chat_id = speech_file.stem
                messages_list = self.load_json_safe(speech_file)
                
                if isinstance(messages_list, list):
                    total_msgs = len(messages_list)
                    
                    for idx, msg in enumerate(messages_list):
                        # Calculate normalized temporal progress
                        progress = (idx + 1) / total_msgs if total_msgs > 0 else 0
                        
                        # Bin progress into 20 segments (5% each) for analysis
                        progress_bin = min(int(progress * 20) + 1, 20)
                        
                        speech_records.append({
                            # Identifiers
                            "chat_id": chat_id,
                            "group": group_name,
                            "msg_index": idx,
                            
                            # Temporal metadata
                            "progress": round(progress, 3),
                            "progress_bin": progress_bin,
                            "phase": msg.get("phase"),
                            
                            # Linguistic features
                            "party": msg.get("party"),
                            "primary_act": msg.get("primary_act"),
                            "argumentative_func": msg.get("argumentative_function"),
                            
                            # Quality metrics
                            "text_length": len(msg.get("text", "") or ""),
                            "consensus_score": msg.get("consensus_score", 1.0)
                        })
        
        df = pd.DataFrame(speech_records)
        logger.info(f"✅ Aggregated {len(df)} individual speech acts")
        
        return df
    
    def aggregate_temporal_evolution(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate temporal evolution matrices for time-series analysis.
        
        Creates pivot tables showing how speech act distributions change
        over the course of negotiations (binned into 20 temporal segments).
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Two pivot tables:
                1. primary_acts_pivot: Speech acts × time bins
                2. argumentative_funcs_pivot: Argumentative functions × time bins
                
            Both tables contain absolute counts (not normalized) to preserve
            information about negotiation activity levels.
        
        Example:
            >>> primary, argumentative = agg.aggregate_temporal_evolution()
            >>> print(primary.iloc[:, :5])  # First 5 time bins
                         bin_1  bin_2  bin_3  bin_4  bin_5
            directive       45     52     38     41     35
            commissive      23     28     31     29     27
            ...
            
        Notes:
            - Returns empty DataFrames if no speech act data available
            - Removes 'Unnamed' columns resulting from parsing artifacts
            - Bins enable statistical analysis of negotiation dynamics
            - Useful for identifying tactical pattern evolution
        """
        df_speech = self.aggregate_speech_acts()
        
        if df_speech.empty:
            logger.warning("⚠️  No speech act data for temporal evolution")
            return pd.DataFrame(), pd.DataFrame()
        
        # Primary speech acts over time
        temporal_primary = (
            df_speech.groupby(['progress_bin', 'primary_act'])
            .size()
            .reset_index(name='count')
        )
        temporal_primary_pivot = temporal_primary.pivot(
            index='progress_bin',
            columns='primary_act',
            values='count'
        ).fillna(0)
        
        # Argumentative functions over time
        temporal_arg = (
            df_speech.groupby(['progress_bin', 'argumentative_func'])
            .size()
            .reset_index(name='count')
        )
        temporal_arg_pivot = temporal_arg.pivot(
            index='progress_bin',
            columns='argumentative_func',
            values='count'
        ).fillna(0)
        
        # Clean artifact columns
        temporal_primary_pivot = self._clean_pivot_columns(temporal_primary_pivot)
        temporal_arg_pivot = self._clean_pivot_columns(temporal_arg_pivot)
        
        logger.info(
            f"✅ Generated temporal evolution data ({len(temporal_primary_pivot)} bins)"
        )
        
        return temporal_primary_pivot, temporal_arg_pivot
    
    def aggregate_group_profiles(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate group attribution profiles for cross-group comparison.
        
        Creates normalized proportion matrices showing linguistic patterns
        specific to each ransomware group. Normalization ensures fair
        comparison across groups with different activity levels.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Two pivot tables:
                1. group_primary_pivot: Groups × speech acts (normalized)
                2. group_arg_pivot: Groups × argumentative functions (normalized)
                
            All rows sum to 1.0 (proportion normalization).
        
        Example:
            >>> group_primary, group_arg = agg.aggregate_group_profiles()
            >>> print(group_primary.loc['lockbit'])
            directive      0.387
            commissive     0.265
            assertive      0.193
            ...
            >>> print(group_primary.sum(axis=1))  # Verify normalization
            lockbit    1.000
            conti      1.000
            ...
            
        Notes:
            - Proportions enable fair comparison across group sizes
            - Useful for group attribution and behavioral fingerprinting
            - Returns empty DataFrames if no speech act data available
            - Automatically removes parsing artifacts
        """
        df_speech = self.aggregate_speech_acts()
        
        if df_speech.empty:
            logger.warning("⚠️  No speech act data for group profiles")
            return pd.DataFrame(), pd.DataFrame()
        
        # Primary acts by group (with normalization)
        group_primary = (
            df_speech.groupby(['group', 'primary_act'])
            .size()
            .reset_index(name='count')
        )
        
        # Calculate group totals for normalization
        group_totals = group_primary.groupby('group')['count'].sum()
        
        # Normalize to proportions
        group_primary['proportion'] = group_primary.apply(
            lambda row: (row['count'] / group_totals[row['group']] 
                        if group_totals[row['group']] > 0 else 0),
            axis=1
        )
        
        group_primary_pivot = group_primary.pivot(
            index='group',
            columns='primary_act',
            values='proportion'
        ).fillna(0)
        
        # Argumentative functions by group (with normalization)
        group_arg = (
            df_speech.groupby(['group', 'argumentative_func'])
            .size()
            .reset_index(name='count')
        )
        
        group_arg_totals = group_arg.groupby('group')['count'].sum()
        
        group_arg['proportion'] = group_arg.apply(
            lambda row: (row['count'] / group_arg_totals[row['group']]
                        if group_arg_totals[row['group']] > 0 else 0),
            axis=1
        )
        
        group_arg_pivot = group_arg.pivot(
            index='group',
            columns='argumentative_func',
            values='proportion'
        ).fillna(0)
        
        # Clean artifact columns
        group_primary_pivot = self._clean_pivot_columns(group_primary_pivot)
        group_arg_pivot = self._clean_pivot_columns(group_arg_pivot)
        
        logger.info(
            f"✅ Generated group profiles for {len(group_primary_pivot)} groups"
        )
        
        return group_primary_pivot, group_arg_pivot
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """
        Generate comprehensive statistics about the aggregation process.
        
        Returns:
            Dictionary containing:
                Data Quality:
                    - total_chats: Number of negotiations processed
                    - chats_with_psychological: Chats with psychological profiles
                    - psychological_coverage_pct: Percentage with psych data
                    - corrupted_files: Number of unreadable JSON files
                    - validation_failures: Records that failed validation
                Message Statistics:
                    - total_messages: Total speech acts processed
                    - avg_messages_per_chat: Mean conversation length
                Financial Completeness:
                    - financial_completeness_pct: Non-null financial data %
                Group Coverage:
                    - groups_processed: Number of ransomware groups
                    
        Example:
            >>> agg = DataAggregator(project_root)
            >>> df_neg = agg.aggregate_negotiations()
            >>> stats = agg.get_aggregation_stats()
            >>> print(f"Coverage: {stats['psychological_coverage_pct']:.1f}%")
            Coverage: 87.3%
            
        Notes:
            - Call after running aggregation methods
            - Useful for data quality assessment
            - Supports reproducibility reporting
        """
        df_neg = self.aggregate_negotiations()
        df_speech = self.aggregate_speech_acts()
        
        if df_neg.empty or df_speech.empty:
            logger.warning("⚠️  Cannot generate stats: empty datasets")
            return {}
        
        # Calculate metrics
        total_chats = len(df_neg)
        chats_with_psych = df_neg['attacker_tone'].notna().sum()
        avg_messages = df_speech.groupby('chat_id').size().mean()
        groups = df_neg['group'].nunique()
        
        # Data completeness (financial fields)
        financial_cols = ['initial_demand', 'final_price', 'discount_pct']
        completeness = (
            df_neg[financial_cols].notna().sum().sum() / 
            (len(df_neg) * len(financial_cols))
        ) * 100
        
        return {
            # Data quality
            'total_chats': total_chats,
            'chats_with_psychological': int(chats_with_psych),
            'psychological_coverage_pct': round(
                (chats_with_psych / total_chats) * 100, 2
            ),
            'corrupted_files': self._stats['corrupted_files'],
            'validation_failures': self._stats['validation_failures'],
            
            # Message statistics
            'total_messages': len(df_speech),
            'avg_messages_per_chat': round(avg_messages, 2),
            
            # Financial completeness
            'financial_completeness_pct': round(completeness, 2),
            
            # Group coverage
            'groups_processed': groups
        }
    
    def save_aggregation_metadata(self, output_dir: Path) -> None:
        """
        Save metadata about the aggregation process for reproducibility.
        
        Generates a JSON file containing:
        - Timestamp of aggregation
        - Source directory paths
        - Aggregation statistics
        - Software versions (Python, Pandas)
        
        Args:
            output_dir: Directory where CSV files are saved
            
        Example:
            >>> agg = DataAggregator(project_root)
            >>> # ... run aggregations ...
            >>> agg.save_aggregation_metadata(processed_dir)
            💾 Saved aggregation metadata: AGGREGATION_METADATA.json
            
        Notes:
            - Essential for research reproducibility
            - Includes data provenance information
            - Useful for thesis documentation
        """
        metadata = {
            # Provenance
            'aggregation_timestamp': datetime.now().isoformat(),
            'aggregator_version': '1.0.0',
            
            # Source directories
            'source_directories': {
                'tactical': str(self.tactical_dir),
                'profiling': str(self.profiling_dir),
                'speech_consensus': str(self.speech_dir)
            },
            
            # Statistics
            'statistics': self.get_aggregation_stats(),
            
            # Software environment
            'software': {
                'pandas_version': pd.__version__,
                'python_version': sys.version.split()[0]
            },
            
            # Academic context
            'thesis': {
                'student': 'Brilant Gashi',
                'institution': 'University of Brescia',
                'supervisors': ['Prof. Federico Cerutti', 'Prof. Pietro Baroni'],
                'year': '2025-2026'
            }
        }
        
        metadata_file = output_dir / 'AGGREGATION_METADATA.json'
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved aggregation metadata: {metadata_file.name}")
        except Exception as e:
            logger.error(f"❌ Failed to save metadata: {e}")
    
    def _clean_pivot_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove artifact columns from pivot tables.
        
        Cleans:
        - Columns starting with 'Unnamed' (Pandas artifacts)
        - Fully empty columns (all NaN)
        - Whitespace-only column names
        
        Args:
            df: Pivot table DataFrame
            
        Returns:
            Cleaned DataFrame
            
        Notes:
            - Private helper method
            - Prevents downstream analysis errors
            - Improves data visualization quality
        """
        # Remove 'Unnamed' columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
        
        # Remove fully empty columns
        df = df.dropna(axis=1, how='all')
        
        # Remove whitespace-only column names
        df = df.loc[:, ~df.columns.str.strip().eq('')]
        
        return df


# Main execution
if __name__ == "__main__":
    # Setup console logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("📊 DATA AGGREGATION & PROCESSING MODULE")
    print("=" * 70)
    print(f"📂 Project Root:  {project_root}")
    print(f"📂 Output Dir:    {processed_dir}")
    print("-" * 70)
    
    # Initialize aggregator
    agg = DataAggregator(project_root)
    
    # 1. Negotiations dataset (chat-level)
    df_neg = agg.aggregate_negotiations()
    if not df_neg.empty:
        out_csv = processed_dir / "dataset_negotiations.csv"
        df_neg.to_csv(out_csv, index=False)
        print(f"💾 Saved Negotiations DB:   {out_csv.name}")
    
    # 2. Speech acts dataset (message-level)
    df_speech = agg.aggregate_speech_acts()
    if not df_speech.empty:
        out_csv = processed_dir / "dataset_speech_acts.csv"
        df_speech.to_csv(out_csv, index=False)
        print(f"💾 Saved Speech Acts DB:    {out_csv.name}")
    
    # 3. Temporal evolution datasets
    temporal_primary, temporal_arg = agg.aggregate_temporal_evolution()
    if not temporal_primary.empty:
        out_csv_p = processed_dir / "temporal_speech_acts.csv"
        out_csv_a = processed_dir / "temporal_argumentative.csv"
        temporal_primary.to_csv(out_csv_p, index=True)
        temporal_arg.to_csv(out_csv_a, index=True)
        print(f"💾 Saved Temporal Data:     temporal_*.csv")
    
    # 4. Group attribution datasets
    group_primary, group_arg = agg.aggregate_group_profiles()
    if not group_primary.empty:
        out_csv_p = processed_dir / "group_speech_acts.csv"
        out_csv_a = processed_dir / "group_argumentative.csv"
        group_primary.to_csv(out_csv_p, index=True)
        group_arg.to_csv(out_csv_a, index=True)
        print(f"💾 Saved Group Profiles:    group_*.csv")
    
    # 5. Save metadata for reproducibility
    agg.save_aggregation_metadata(processed_dir)
    
    # 6. Print statistics
    print("\n" + "=" * 70)
    print("📈 AGGREGATION STATISTICS")
    print("=" * 70)
    
    stats = agg.get_aggregation_stats()
    if stats:
        print(f"  Total Chats Processed:        {stats['total_chats']}")
        print(f"  With Psychological Data:      {stats['chats_with_psychological']} "
              f"({stats['psychological_coverage_pct']:.1f}%)")
        print(f"  Total Messages:               {stats['total_messages']}")
        print(f"  Avg Messages per Chat:        {stats['avg_messages_per_chat']}")
        print(f"  Financial Completeness:       {stats['financial_completeness_pct']:.1f}%")
        print(f"  Groups Processed:             {stats['groups_processed']}")
        print(f"  Corrupted Files:              {stats['corrupted_files']}")
        print(f"  Validation Failures:          {stats['validation_failures']}")
    
    print("=" * 70)
    print("✅  AGGREGATION COMPLETE")
    print("=" * 70 + "\n")
