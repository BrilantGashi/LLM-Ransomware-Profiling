"""
Data Aggregation Module for Ransomware Negotiation Analysis
Aggregates consensus-validated multi-source JSON outputs into structured DataFrames.

This module consolidates consensus outputs from three analysis tasks:
- Tactical Extraction: Financial and technical negotiation indicators
- Psychological Profiling: Behavioral traits and communication patterns
- Speech Act Analysis: Consensus-based linguistic classifications

Author: Brilant Gashi
Supervisors: Prof. Federico Cerutti, Prof. Pietro Baroni
Institution: University of Brescia
"""


import json
import sys
import yaml
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List


logger = logging.getLogger(__name__)


class DataAggregator:
    """
    Multi-level data aggregator for ransomware negotiation analysis.
    
    This class implements a three-tier aggregation strategy using consensus-validated data:
    1. Chat-level: Merges tactical and psychological features per negotiation
    2. Message-level: Individual speech acts with temporal metadata
    3. Statistical: Temporal evolution and group attribution profiles
    """
    
    def __init__(self, base_dir: Path):
        """Initialize aggregator with project directory structure."""
        self.base_dir = base_dir
        self.consensus_dir = base_dir / "data" / "consensus"
        
        self.tactical_dir = self.consensus_dir / "tactical_extraction"
        self.profiling_dir = self.consensus_dir / "psychological_profiling"
        self.speech_dir = self.consensus_dir / "speech_act_analysis"
        
        # Initialize statistics tracking
        self._stats = {
            'chats_processed': 0,
            'chats_with_psychological': 0,
            'corrupted_files': 0,
            'validation_failures': 0,
            'missing_consensus_data': 0
        }
        
        self._validate_consensus_availability()
    
    def _validate_consensus_availability(self):
        """
        Verify consensus data exists before aggregation.
        Raises warning if consensus directories are missing.
        """
        missing_dirs = []
        
        for task, path in [
            ('tactical_extraction', self.tactical_dir),
            ('psychological_profiling', self.profiling_dir),
            ('speech_act_analysis', self.speech_dir)
        ]:
            if not path.exists():
                missing_dirs.append(task)
                logger.warning(f"Missing consensus directory: {task}")
        
        if missing_dirs:
            logger.warning(
                f"Consensus data not found for: {', '.join(missing_dirs)}. "
                f"Run consensus.py to generate consensus data."
            )
            self._stats['missing_consensus_data'] = len(missing_dirs)
        else:
            logger.info("All consensus directories found")
    
    def _get_available_chats(self, directory: Path) -> Dict[str, List[str]]:
        """
        Get all available chat_ids by group from a consensus directory.
        
        Returns:
            Dict[group_name] -> List[chat_id]
        """
        chats_by_group = {}
        
        if not directory.exists():
            return {}
        
        for group_dir in directory.iterdir():
            if not group_dir.is_dir():
                continue
            
            group_name = group_dir.name
            chat_ids = [f.stem for f in group_dir.glob("*.json")]
            
            if chat_ids:
                chats_by_group[group_name] = chat_ids
        
        return chats_by_group
    
    def load_json_safe(self, file_path: Path) -> Dict[str, Any]:
        """Safely load JSON file with error handling."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in {file_path.name}: {str(e)[:50]}")
            self._stats['corrupted_files'] += 1
            return {}
        except FileNotFoundError:
            logger.debug(f"File not found: {file_path.name}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error loading {file_path.name}: {e}")
            self._stats['corrupted_files'] += 1
            return {}
    
    def validate_negotiation_record(self, record: Dict[str, Any]) -> bool:
        """Validate negotiation record against schema requirements."""
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
                    f"Discount percentage out of range in {record['chat_id']}: "
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
        Aggregate chat-level features from consensus tactical and psychological analysis.
        
        Returns:
            pd.DataFrame: Chat-level dataset with financial, technical, and psychological features
        """
        data_records = []
        
        if not self.tactical_dir.exists():
            logger.error(f"Tactical consensus directory not found: {self.tactical_dir}")
            logger.info("Run consensus.py to generate tactical_extraction consensus")
            return pd.DataFrame()
        
        tactical_chats = self._get_available_chats(self.tactical_dir)
        
        if not tactical_chats:
            logger.warning("No tactical consensus data found")
            return pd.DataFrame()
        
        total_chats = sum(len(chats) for chats in tactical_chats.values())
        logger.info(f"Found {total_chats} chats in tactical consensus")
        
        for group_name, chat_ids in tactical_chats.items():
            for chat_id in chat_ids:
                tactical_file = self.tactical_dir / group_name / f"{chat_id}.json"
                
                tactical_data = self.load_json_safe(tactical_file)
                if not tactical_data:
                    continue
                
                # Extract nested fields
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
                
                # Load psychological profiling data
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
                            ) if att_prof.get("cialdini_influence_tactics") else None,
                            
                            # Victim psychological profile
                            "victim_emotion": vic_prof.get("emotional_trajectory"),
                            "victim_strategy": vic_prof.get("primary_negotiation_tactic"),
                            "victim_effectiveness": vic_prof.get("tactic_effectiveness")
                        })
                        
                        self._stats['chats_with_psychological'] += 1
                
                if self.validate_negotiation_record(record):
                    data_records.append(record)
                    self._stats['chats_processed'] += 1
        
        df = pd.DataFrame(data_records)
        
        if df.empty:
            logger.warning("No valid negotiation records generated")
        else:
            logger.info(
                f"Aggregated {len(df)} negotiations from consensus data "
                f"({self._stats['chats_with_psychological']} with psychological profiles)"
            )
        
        return df
    
    def aggregate_speech_acts(self) -> pd.DataFrame:
        """
        Create message-level dataset with consensus speech act classifications.
        
        Returns:
            pd.DataFrame: Message-level dataset with speech acts and quality metrics
        """
        speech_records = []
        
        if not self.speech_dir.exists():
            logger.error(f"Speech acts consensus directory not found: {self.speech_dir}")
            logger.info("Run consensus.py to generate speech_act_analysis consensus")
            return pd.DataFrame()
        
        speech_chats = self._get_available_chats(self.speech_dir)
        
        if not speech_chats:
            logger.warning("No speech act consensus data found")
            return pd.DataFrame()
        
        total_chats = sum(len(chats) for chats in speech_chats.values())
        logger.info(f"Found {total_chats} chats in speech act consensus")
        
        for group_name, chat_ids in speech_chats.items():
            for chat_id in chat_ids:
                speech_file = self.speech_dir / group_name / f"{chat_id}.json"
                messages_list = self.load_json_safe(speech_file)
                
                if isinstance(messages_list, list):
                    total_msgs = len(messages_list)
                    
                    for idx, msg in enumerate(messages_list):
                        # Calculate normalized temporal progress
                        progress = (idx + 1) / total_msgs if total_msgs > 0 else 0
                        
                        # Bin progress into 20 segments
                        progress_bin = min(int(progress * 20) + 1, 20)
                        
                        # Determine phase
                        if progress <= 0.33:
                            phase = "opening"
                        elif progress <= 0.67:
                            phase = "middle"
                        else:
                            phase = "closing"
                        
                        speech_records.append({
                            # Identifiers
                            "chat_id": chat_id,
                            "group": group_name,
                            "msg_index": idx,
                            
                            # Temporal metadata
                            "progress": round(progress, 3),
                            "progress_bin": progress_bin,
                            "phase": msg.get("phase", phase),
                            
                            # Linguistic features
                            "party": msg.get("party"),
                            "primary_act": msg.get("primary_act"),
                            "argumentative_func": msg.get("argumentative_function"),
                            
                            # Quality metrics
                            "text_length": len(msg.get("text", "") or ""),
                            "consensus_score": msg.get("consensus_score", 1.0)
                        })
        
        df = pd.DataFrame(speech_records)
        
        if df.empty:
            logger.warning("No valid speech act records generated")
        else:
            logger.info(
                f"Aggregated {len(df)} individual speech acts from consensus data"
            )
        
        return df
    
    def aggregate_temporal_evolution(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate temporal evolution matrices for time-series analysis.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Speech acts and argumentative functions over time
        """
        df_speech = self.aggregate_speech_acts()
        
        if df_speech.empty:
            logger.warning("No speech act data for temporal evolution")
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
        
        temporal_primary_pivot = self._clean_pivot_columns(temporal_primary_pivot)
        temporal_arg_pivot = self._clean_pivot_columns(temporal_arg_pivot)
        
        logger.info(
            f"Generated temporal evolution data ({len(temporal_primary_pivot)} bins)"
        )
        
        return temporal_primary_pivot, temporal_arg_pivot
    
    def aggregate_group_profiles(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate group attribution profiles for cross-group comparison.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Normalized group profiles
        """
        df_speech = self.aggregate_speech_acts()
        
        if df_speech.empty:
            logger.warning("No speech act data for group profiles")
            return pd.DataFrame(), pd.DataFrame()
        
        # Primary acts by group with normalization
        group_primary = (
            df_speech.groupby(['group', 'primary_act'])
            .size()
            .reset_index(name='count')
        )
        
        group_totals = group_primary.groupby('group')['count'].sum()
        
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
        
        # Argumentative functions by group with normalization
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
        
        group_primary_pivot = self._clean_pivot_columns(group_primary_pivot)
        group_arg_pivot = self._clean_pivot_columns(group_arg_pivot)
        
        logger.info(
            f"Generated group profiles for {len(group_primary_pivot)} groups"
        )
        
        return group_primary_pivot, group_arg_pivot
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Generate comprehensive statistics about the aggregation process."""
        df_neg = self.aggregate_negotiations()
        df_speech = self.aggregate_speech_acts()
        
        if df_neg.empty and df_speech.empty:
            logger.warning("Cannot generate stats: empty datasets")
            return {
                'status': 'NO_DATA',
                'missing_consensus_data': self._stats['missing_consensus_data']
            }
        
        stats = {}
        
        if not df_neg.empty:
            total_chats = len(df_neg)
            chats_with_psych = df_neg['attacker_tone'].notna().sum()
            groups = df_neg['group'].nunique()
            
            # Data completeness for financial fields
            financial_cols = ['initial_demand', 'final_price', 'discount_pct']
            completeness = (
                df_neg[financial_cols].notna().sum().sum() / 
                (len(df_neg) * len(financial_cols))
            ) * 100
            
            stats.update({
                'total_chats': total_chats,
                'chats_with_psychological': int(chats_with_psych),
                'psychological_coverage_pct': round(
                    (chats_with_psych / total_chats) * 100, 2
                ) if total_chats > 0 else 0,
                'financial_completeness_pct': round(completeness, 2),
                'groups_processed': groups
            })
        
        if not df_speech.empty:
            avg_messages = df_speech.groupby('chat_id').size().mean()
            avg_consensus = df_speech['consensus_score'].mean()
            
            stats.update({
                'total_messages': len(df_speech),
                'avg_messages_per_chat': round(avg_messages, 2),
                'avg_consensus_score': round(avg_consensus, 3)
            })
        
        stats.update({
            'corrupted_files': self._stats['corrupted_files'],
            'validation_failures': self._stats['validation_failures'],
            'missing_consensus_data': self._stats['missing_consensus_data']
        })
        
        return stats
    
    def save_aggregation_metadata(self, output_dir: Path) -> None:
        """Save metadata about the aggregation process for reproducibility."""
        metadata = {
            # Provenance
            'aggregation_timestamp': datetime.now().isoformat(),
            'data_source': 'CONSENSUS_VALIDATED',
            
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
            
            logger.info(f"Saved aggregation metadata: {metadata_file.name}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _clean_pivot_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove artifact columns from pivot tables."""
        df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
        df = df.dropna(axis=1, how='all')
        df = df.loc[:, ~df.columns.str.strip().eq('')]
        return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    project_root = Path(__file__).parent.parent.parent
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("DATA AGGREGATION & PROCESSING MODULE")
    print("CONSENSUS-VALIDATED DATA")
    print("=" * 70)
    print(f"Project Root:  {project_root}")
    print(f"Output Dir:    {processed_dir}")
    print(f"Data Source:   data/consensus/")
    print("-" * 70)
    
    agg = DataAggregator(project_root)
    
    stats = agg.get_aggregation_stats()
    if stats.get('status') == 'NO_DATA':
        print("\n" + "=" * 70)
        print("NO CONSENSUS DATA FOUND")
        print("=" * 70)
        print("Run these commands first:")
        print("   1. cd src/analysis")
        print("   2. python consensus.py")
        print("   3. python aggregator.py")
        print("=" * 70 + "\n")
        sys.exit(1)
    
    # Aggregating negotiations dataset
    print("\nAggregating negotiations...")
    df_neg = agg.aggregate_negotiations()
    if not df_neg.empty:
        out_csv = processed_dir / "dataset_negotiations.csv"
        df_neg.to_csv(out_csv, index=False)
        print(f"   Saved: {out_csv.name} ({len(df_neg)} rows)")
    else:
        print("   No negotiation data generated")
    
    # Aggregating speech acts dataset
    print("\nAggregating speech acts...")
    df_speech = agg.aggregate_speech_acts()
    if not df_speech.empty:
        out_csv = processed_dir / "dataset_speech_acts.csv"
        df_speech.to_csv(out_csv, index=False)
        print(f"   Saved: {out_csv.name} ({len(df_speech)} rows)")
    else:
        print("   No speech act data generated")
    
    # Generating temporal evolution datasets
    print("\nGenerating temporal evolution...")
    temporal_primary, temporal_arg = agg.aggregate_temporal_evolution()
    if not temporal_primary.empty:
        out_csv_p = processed_dir / "temporal_speech_acts.csv"
        out_csv_a = processed_dir / "temporal_argumentative.csv"
        temporal_primary.to_csv(out_csv_p, index=True)
        temporal_arg.to_csv(out_csv_a, index=True)
        print(f"   Saved: temporal_*.csv ({len(temporal_primary)} time bins)")
    else:
        print("   No temporal data generated")
    
    # Generating group attribution datasets
    print("\nGenerating group profiles...")
    group_primary, group_arg = agg.aggregate_group_profiles()
    if not group_primary.empty:
        out_csv_p = processed_dir / "group_speech_acts.csv"
        out_csv_a = processed_dir / "group_argumentative.csv"
        group_primary.to_csv(out_csv_p, index=True)
        group_arg.to_csv(out_csv_a, index=True)
        print(f"   Saved: group_*.csv ({len(group_primary)} groups)")
    else:
        print("   No group profile data generated")
    
    # Saving metadata
    print("\nSaving metadata...")
    agg.save_aggregation_metadata(processed_dir)
    
    # Print statistics
    print("\n" + "=" * 70)
    print("AGGREGATION STATISTICS")
    print("=" * 70)
    
    final_stats = agg.get_aggregation_stats()
    if final_stats and final_stats.get('status') != 'NO_DATA':
        if 'total_chats' in final_stats:
            print(f"  Total Chats Processed:        {final_stats['total_chats']}")
            print(f"  With Psychological Data:      {final_stats['chats_with_psychological']} "
                  f"({final_stats['psychological_coverage_pct']:.1f}%)")
            print(f"  Financial Completeness:       {final_stats['financial_completeness_pct']:.1f}%")
            print(f"  Groups Processed:             {final_stats['groups_processed']}")
        
        if 'total_messages' in final_stats:
            print(f"  Total Messages:               {final_stats['total_messages']}")
            print(f"  Avg Messages per Chat:        {final_stats['avg_messages_per_chat']}")
            print(f"  Avg Consensus Score:          {final_stats['avg_consensus_score']:.3f}")
        
        print(f"  Corrupted Files:              {final_stats['corrupted_files']}")
        print(f"  Validation Failures:          {final_stats['validation_failures']}")
    
    print("=" * 70)
    print("AGGREGATION COMPLETE")
    print("=" * 70 + "\n")
