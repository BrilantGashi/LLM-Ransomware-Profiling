import json
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class DataAggregator:
    """
    Aggregates processed JSON outputs into structured Pandas DataFrames for analysis.
    Merges data from:
      - Tactical Extraction (Financial/Technical indicators)
      - Psychological Profiling (Behavioral traits)
      - Speech Act Analysis (Consensus-based linguistic labels)
    """

    def __init__(self, base_dir: Path):
        self.outputs_dir = base_dir / "data" / "outputs"
        self.consensus_dir = base_dir / "data" / "consensus"
        
        # Define specific source directories
        self.tactical_dir = self.outputs_dir / "tactical_extraction"
        self.profiling_dir = self.outputs_dir / "psychological_profiling"
        
        # Prefer CONSENSUS data for Speech Acts (Gold Standard)
        self.speech_dir = self.consensus_dir / "speech_act_analysis"

    def load_json_safe(self, file_path: Path) -> Dict[str, Any]:
        """Safely loads a JSON file, handling errors gracefully."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"⚠️  Skipping corrupted/missing file {file_path.name}: {e}")
            return {}

    def aggregate_negotiations(self) -> pd.DataFrame:
        """
        Merges 'Tactical Extraction' + 'Psychological Profiling' into a single DataFrame.
        One row per Negotiation (Chat ID).
        """
        data_records = []

        if not self.tactical_dir.exists():
            logger.warning(f"❌ Tactical directory not found at {self.tactical_dir}")
            return pd.DataFrame()

        # Iterate through Group Folders (e.g., Akira, Lockbit)
        for group_dir in self.tactical_dir.iterdir():
            if not group_dir.is_dir(): continue
            
            group_name = group_dir.name
            
            for tactical_file in group_dir.glob("*.json"):
                chat_id = tactical_file.stem
                
                # --- A. Load Tactical Data ---
                tactical_data = self.load_json_safe(tactical_file)
                if not tactical_data: continue

                # Extract nested fields safely
                meta = tactical_data.get("metadata", {})
                finance = tactical_data.get("financial_negotiation", {})
                tech = tactical_data.get("technical_indicators", {})
                dynamics = tactical_data.get("negotiation_dynamics", {})

                record = {
                    "chat_id": chat_id,
                    "group": group_name,
                    # Metadata
                    "victim_size": meta.get("victim_size"),
                    "attack_type": meta.get("attack_type"),
                    # Financials
                    "initial_demand": finance.get("initial_demand"),
                    "final_price": finance.get("final_agreed_price"),
                    "discount_pct": finance.get("discount_percentage"),
                    "currency": finance.get("currency"),
                    # Technicals
                    "data_volume_gb": tech.get("data_volume_gb"),
                    "exfiltration_confirmed": tech.get("exfiltration_confirmed"),
                    # Dynamics
                    "outcome": dynamics.get("outcome_status"),
                    "attacker_flexibility": dynamics.get("attacker_flexibility")
                }

                # --- B. Load Psychological Data (Join via Chat ID) ---
                # Checks if a corresponding profile exists for this chat
                profile_file = self.profiling_dir / group_name / f"{chat_id}.json"
                if profile_file.exists():
                    profile_data = self.load_json_safe(profile_file)
                    
                    att_prof = profile_data.get("attacker_profile", {})
                    vic_prof = profile_data.get("victim_profile", {})
                    
                    record.update({
                        "attacker_tone": att_prof.get("communication_tone"),
                        "attacker_competence": att_prof.get("competence_level"),
                        "attacker_strategy": att_prof.get("primary_strategy_dual_concern"),
                        "influence_tactics": ", ".join(att_prof.get("cialdini_influence_tactics", [])),
                        
                        "victim_emotion": vic_prof.get("emotional_trajectory"),
                        "victim_strategy": vic_prof.get("primary_negotiation_tactic"),
                        "victim_effectiveness": vic_prof.get("tactic_effectiveness")
                    })
                
                data_records.append(record)

        df = pd.DataFrame(data_records)
        logger.info(f"✅ Aggregated {len(df)} negotiations (Tactical + Psych).")
        return df

    def aggregate_speech_acts(self) -> pd.DataFrame:
        """
        Creates a granular DataFrame where each row represents a SINGLE MESSAGE.
        Uses Consensus Data if available.
        """
        speech_records = []
        
        if not self.speech_dir.exists():
            logger.warning(f"❌ Speech acts directory (Consensus) not found at {self.speech_dir}")
            return pd.DataFrame()

        for group_dir in self.speech_dir.iterdir():
            if not group_dir.is_dir(): continue
            group_name = group_dir.name
            
            for speech_file in group_dir.glob("*.json"):
                chat_id = speech_file.stem
                messages_list = self.load_json_safe(speech_file)
                
                if isinstance(messages_list, list):
                    total_msgs = len(messages_list)
                    
                    for idx, msg in enumerate(messages_list):
                        # Calculate Normalized Progress (0.0 to 1.0) for temporal analysis
                        progress = (idx + 1) / total_msgs if total_msgs > 0 else 0
                        # Bin progress into 20 segments (5% each) for plotting
                        progress_bin = min(int(progress * 20) + 1, 20)
                        
                        speech_records.append({
                            "chat_id": chat_id,
                            "group": group_name,
                            "msg_index": idx,
                            "progress": round(progress, 3),
                            "progress_bin": progress_bin,
                            "party": msg.get("party"),
                            "primary_act": msg.get("primary_act"),
                            "argumentative_func": msg.get("argumentative_function"),
                            "phase": msg.get("phase"),
                            "text_length": len(msg.get("text", "") or ""),
                            "consensus_score": msg.get("consensus_score", 1.0) # Default to 1.0 if not present
                        })
        
        df = pd.DataFrame(speech_records)
        logger.info(f"✅ Aggregated {len(df)} individual speech acts.")
        return df

    def aggregate_temporal_evolution(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates pivot tables for TEMPORAL ANALYSIS (Evolution over time bins).
        Returns: (primary_acts_pivot, argumentative_funcs_pivot)
        """
        df_speech = self.aggregate_speech_acts()
        
        if df_speech.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # 1. Primary Speech Acts over Time
        temporal_primary = df_speech.groupby(['progress_bin', 'primary_act']).size().reset_index(name='count')
        temporal_primary_pivot = temporal_primary.pivot(index='progress_bin', columns='primary_act', values='count').fillna(0)
        
        # 2. Argumentative Functions over Time
        temporal_arg = df_speech.groupby(['progress_bin', 'argumentative_func']).size().reset_index(name='count')
        temporal_arg_pivot = temporal_arg.pivot(index='progress_bin', columns='argumentative_func', values='count').fillna(0)
        
        # CLEAN DATA: Remove 'Unnamed' or empty columns resulting from bad parsing
        temporal_primary_pivot = self._clean_pivot_columns(temporal_primary_pivot)
        temporal_arg_pivot = self._clean_pivot_columns(temporal_arg_pivot)
        
        logger.info(f"✅ Generated temporal evolution data ({len(temporal_primary_pivot)} bins).")
        return temporal_primary_pivot, temporal_arg_pivot

    def aggregate_group_profiles(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates pivot tables for GROUP ATTRIBUTION (Comparison between Ransomware Groups).
        Returns normalized proportions (rows sum to 1.0).
        """
        df_speech = self.aggregate_speech_acts()
        
        if df_speech.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # 1. Primary Acts by Group (Normalized)
        group_primary = df_speech.groupby(['group', 'primary_act']).size().reset_index(name='count')
        # Normalize by row (Group Total) to compare proportions, not absolute counts
        group_totals = group_primary.groupby('group')['count'].sum()
        group_primary['proportion'] = group_primary.apply(
            lambda row: row['count'] / group_totals[row['group']] if group_totals[row['group']] > 0 else 0, axis=1
        )
        group_primary_pivot = group_primary.pivot(index='group', columns='primary_act', values='proportion').fillna(0)
        
        # 2. Argumentative Functions by Group (Normalized)
        group_arg = df_speech.groupby(['group', 'argumentative_func']).size().reset_index(name='count')
        group_arg_totals = group_arg.groupby('group')['count'].sum()
        group_arg['proportion'] = group_arg.apply(
            lambda row: row['count'] / group_arg_totals[row['group']] if group_arg_totals[row['group']] > 0 else 0, axis=1
        )
        group_arg_pivot = group_arg.pivot(index='group', columns='argumentative_func', values='proportion').fillna(0)
        
        # CLEAN DATA
        group_primary_pivot = self._clean_pivot_columns(group_primary_pivot)
        group_arg_pivot = self._clean_pivot_columns(group_arg_pivot)
        
        logger.info(f"✅ Generated group profiles for {len(group_primary_pivot)} groups.")
        return group_primary_pivot, group_arg_pivot

    def _clean_pivot_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to remove artifact columns (Unnamed, null, etc)."""
        # Remove columns starting with "Unnamed"
        df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
        # Remove columns that are fully empty/NaN
        df = df.dropna(axis=1, how='all')
        return df


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Setup simple console logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Define Paths
    project_root = Path(__file__).parent.parent.parent
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("📊 DATA AGGREGATION & PROCESSING MODULE")
    print("="*60)
    print(f"📂 Project Root:  {project_root}")
    print(f"📂 Output Dir:    {processed_dir}")
    print("-" * 60)

    agg = DataAggregator(project_root)
    
    # 1. Negotiations Dataset (Chat-Level)
    df_neg = agg.aggregate_negotiations()
    if not df_neg.empty:
        out_csv = processed_dir / "dataset_negotiations.csv"
        df_neg.to_csv(out_csv, index=False)
        print(f"💾 Saved Negotiations DB:   {out_csv.name}")
        
    # 2. Speech Acts Dataset (Message-Level)
    df_speech = agg.aggregate_speech_acts()
    if not df_speech.empty:
        out_csv = processed_dir / "dataset_speech_acts.csv"
        df_speech.to_csv(out_csv, index=False)
        print(f"💾 Saved Speech Acts DB:    {out_csv.name}")
    
    # 3. Temporal Evolution Datasets (Time-Series)
    temporal_primary, temporal_arg = agg.aggregate_temporal_evolution()
    if not temporal_primary.empty:
        out_csv_p = processed_dir / "temporal_speech_acts.csv"
        out_csv_a = processed_dir / "temporal_argumentative.csv"
        temporal_primary.to_csv(out_csv_p, index=True)
        temporal_arg.to_csv(out_csv_a, index=True)
        print(f"💾 Saved Temporal Data:     temporal_*.csv")
    
    # 4. Group Attribution Datasets (Cross-Group Comparison)
    group_primary, group_arg = agg.aggregate_group_profiles()
    if not group_primary.empty:
        out_csv_p = processed_dir / "group_speech_acts.csv"
        out_csv_a = processed_dir / "group_argumentative.csv"
        group_primary.to_csv(out_csv_p, index=True)
        group_arg.to_csv(out_csv_a, index=True)
        print(f"💾 Saved Group Profiles:    group_*.csv")

    print("="*60)
    print("✅  AGGREGATION COMPLETE")
    print("="*60 + "\n")
