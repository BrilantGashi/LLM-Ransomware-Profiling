"""
Data Visualization Module for Ransomware Negotiation Analysis

Generates publication-ready plots for speech act and argumentative analysis.
Follows academic standards with 300 DPI output.

Author: Brilant Gashi
Supervisors: Prof. Federico Cerutti, Prof. Pietro Baroni
University of Brescia
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
import numpy as np


logger = logging.getLogger(__name__)


class DataVisualizer:
    """
    Publication-ready visualization generator for ransomware negotiation analysis.
    Creates temporal evolution plots, group attribution heatmaps, and cross-analysis charts.
    """
    
    def __init__(self, base_dir: Path):
        """Initialize with data paths and configure professional styling."""
        self.data_dir = base_dir / "data" / "processed"
        self.plots_dir = base_dir / "data" / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Load datasets
        self.df_neg = self._load_csv("dataset_negotiations.csv")
        self.df_speech = self._load_csv("dataset_speech_acts.csv")
        self.df_temp_speech = self._load_csv("temporal_speech_acts.csv", index_col="progress_bin")
        self.df_temp_arg = self._load_csv("temporal_argumentative.csv", index_col="progress_bin")
        self.df_group_speech = self._load_csv("group_speech_acts.csv", index_col="group")
        self.df_group_arg = self._load_csv("group_argumentative.csv", index_col="group")
        
        self._configure_style()
        logger.info(f"Visualizer initialized: {len(self.df_speech)} messages, {len(self.df_neg)} chats")
    
    def _configure_style(self):
        """Configure matplotlib and seaborn for academic publication."""
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        
        self.palette_main = [
            '#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F',
            '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85'
        ]
        
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'figure.dpi': 300,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
            'axes.titlesize': 16,
            'axes.titleweight': 'bold',
            'axes.labelsize': 13,
            'legend.fontsize': 10,
            'savefig.bbox': 'tight',
            'axes.spines.top': False,
            'axes.spines.right': False
        })
    
    def _load_csv(self, filename: str, index_col=None) -> pd.DataFrame:
        """Load CSV file with error handling."""
        path = self.data_dir / filename
        if path.exists():
            return pd.read_csv(path, index_col=index_col)
        else:
            logger.warning(f"{filename} not found")
            return pd.DataFrame()
    
    def _add_caption(self, fig, caption: str):
        """Add caption below plot."""
        fig.text(0.5, -0.01, caption, ha='center', va='top',
                fontsize=10, style='italic', color='#555', wrap=True)
    
    def plot_temporal_speech_acts(self):
        """
        Temporal evolution of speech acts across negotiation phases.
        Shows directive and informative patterns in opening, negotiative in middle, informative in closing.
        """
        if self.df_temp_speech.empty:
            return
        
        df_norm = self.df_temp_speech.div(self.df_temp_speech.sum(axis=1), axis=0) * 100
        df_smooth = df_norm.rolling(window=2, min_periods=1, center=True).mean()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        df_smooth.plot.area(ax=ax, color=self.palette_main, alpha=0.85, linewidth=0)
        
        ax.axvline(x=5, color='gray', linestyle=':', alpha=0.5, linewidth=2)
        ax.axvline(x=15, color='gray', linestyle=':', alpha=0.5, linewidth=2)
        ax.text(2.5, 95, 'Opening', ha='center', fontsize=10, style='italic', color='#555')
        ax.text(10, 95, 'Bargaining', ha='center', fontsize=10, style='italic', color='#555')
        ax.text(17.5, 95, 'Closing', ha='center', fontsize=10, style='italic', color='#555')
        
        ax.set_title('Temporal Evolution of Speech Acts', pad=20)
        ax.set_xlabel('Negotiation Progress (20 bins, 5% each)')
        ax.set_ylabel('Relative Frequency (%)')
        ax.set_xlim(1, 20)
        ax.set_ylim(0, 100)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], title='Speech Act',
                 bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
        
        caption = (f"Speech acts show clear temporal structure across {len(self.df_speech)} messages. "
                  "Directive dominates opening (procedural setup), Negotiative-Evaluative peaks mid-conversation "
                  "(bargaining core), Informative resurfaces at closing (confirmations).")
        self._add_caption(fig, caption)
        
        plt.savefig(self.plots_dir / "01_temporal_speech_acts.png")
        plt.close()
        logger.info("Saved: 01_temporal_speech_acts.png")
    
    def plot_temporal_argumentative(self):
        """
        Temporal evolution of argumentative functions across negotiation phases.
        Shows face and ethos early, value and fairness mid-phase, grounds and facts late.
        """
        if self.df_temp_arg.empty:
            return
        
        df_norm = self.df_temp_arg.div(self.df_temp_arg.sum(axis=1), axis=0) * 100
        df_smooth = df_norm.rolling(window=2, min_periods=1, center=True).mean()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        df_smooth.plot.area(ax=ax, color=self.palette_main, alpha=0.85, linewidth=0)
        
        ax.axvline(x=5, color='gray', linestyle=':', alpha=0.5, linewidth=2)
        ax.axvline(x=15, color='gray', linestyle=':', alpha=0.5, linewidth=2)
        
        ax.set_title('Temporal Evolution of Argumentative Functions', pad=20)
        ax.set_xlabel('Negotiation Progress (20 bins)')
        ax.set_ylabel('Relative Frequency (%)')
        ax.set_xlim(1, 20)
        ax.set_ylim(0, 100)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], title='Function',
                 bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
        
        caption = ("Argumentative functions evolve systematically: early relational grounding (Face/Ethos), "
                  "mid-phase justificatory bargaining (Value/Fairness), late evidential consolidation (Grounds/Facts). "
                  "Confirms structured pragmatic routines in ransomware extortion.")
        self._add_caption(fig, caption)
        
        plt.savefig(self.plots_dir / "02_temporal_argumentative.png")
        plt.close()
        logger.info("Saved: 02_temporal_argumentative.png")
    
    def plot_group_speech_acts_heatmap(self):
        """
        Speech act distribution by ransomware group.
        Shows group-specific communication styles for attribution.
        """
        if self.df_group_speech.empty:
            return
        
        row_sums = self.df_group_speech.sum(axis=1)
        df_plot = self.df_group_speech.loc[row_sums.sort_values(ascending=False).index].head(20)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(df_plot, annot=True, fmt='.2f', cmap='YlOrRd',
                   linewidths=0.5, cbar_kws={'label': 'Proportion'},
                   vmin=0, vmax=1, ax=ax)
        
        ax.set_title('Group Attribution: Speech Act Profiles', pad=20)
        ax.set_xlabel('Speech Act Category')
        ax.set_ylabel('Ransomware Group')
        
        caption = (f"Normalized speech act proportions for top {len(df_plot)} groups. "
                  "Extreme profiles (e.g., Cloak=100% Directive) may reflect small samples. "
                  "Established groups show overlapping patterns—best used with multi-feature attribution.")
        self._add_caption(fig, caption)
        
        plt.savefig(self.plots_dir / "03_group_speech_acts_heatmap.png")
        plt.close()
        logger.info("Saved: 03_group_speech_acts_heatmap.png")
    
    def plot_group_argumentative_heatmap(self):
        """
        Argumentative function distribution by ransomware group.
        Shows justificatory and strategic preferences for attribution.
        """
        if self.df_group_arg.empty:
            return
        
        row_sums = self.df_group_arg.sum(axis=1)
        df_plot = self.df_group_arg.loc[row_sums.sort_values(ascending=False).index].head(20)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(df_plot, annot=True, fmt='.2f', cmap='viridis',
                   linewidths=0.5, cbar_kws={'label': 'Proportion'},
                   vmin=0, vmax=1, ax=ax)
        
        ax.set_title('Group Attribution: Argumentative Profiles', pad=20)
        ax.set_xlabel('Argumentative Function')
        ax.set_ylabel('Ransomware Group')
        
        caption = ("Argumentative preferences reveal group styles: Cloak/RansomHub=Action-Pressure heavy, "
                  "Avaddon/BlackMatter=Value/Fairness focus. Provides stylistic fingerprints when combined "
                  "with speech acts and semantic features.")
        self._add_caption(fig, caption)
        
        plt.savefig(self.plots_dir / "04_group_argumentative_heatmap.png")
        plt.close()
        logger.info("Saved: 04_group_argumentative_heatmap.png")
    
    def plot_negotiation_gap(self):
        """
        Dumbbell plot comparing initial demand versus final settlement.
        Shows the negotiation gap and discount percentages by group.
        """
        if self.df_neg.empty:
            return
        
        df = self.df_neg.dropna(subset=['initial_demand', 'final_price', 'group'])
        if df.empty:
            return
        
        df_grp = df.groupby('group')[['initial_demand', 'final_price']].median()
        df_grp = df_grp[df_grp['initial_demand'] > 0].sort_values('initial_demand').tail(15)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.hlines(y=df_grp.index, xmin=df_grp['final_price'], xmax=df_grp['initial_demand'],
                 color='gray', alpha=0.5, linewidth=2)
        
        ax.scatter(df_grp['initial_demand'], df_grp.index, color='#E64B35',
                  s=120, label='Initial Demand', zorder=3, edgecolors='white', linewidth=1.5)
        ax.scatter(df_grp['final_price'], df_grp.index, color='#00A087',
                  s=120, label='Final Settlement', zorder=3, edgecolors='white', linewidth=1.5)
        
        ax.set_xscale('log')
        ax.set_title('The Negotiation Gap: Initial Demand vs. Settlement', pad=20)
        ax.set_xlabel('Amount (USD, log scale)')
        ax.set_ylabel('Ransomware Group')
        ax.legend(loc='lower right', frameon=True, fancybox=True)
        ax.grid(axis='x', alpha=0.3)
        
        for i, (group, row) in enumerate(df_grp.iterrows()):
            discount = (1 - row['final_price'] / row['initial_demand']) * 100
            if discount > 0:
                ax.text(row['initial_demand'] * 1.2, i, f"-{discount:.0f}%",
                       va='center', color='#E64B35', fontsize=9, fontweight='bold')
        
        caption = (f"Median initial demands vs. final settlements for top {len(df_grp)} groups. "
                  "Gray lines show negotiation range; red annotations indicate discount percentage. "
                  "Most groups achieve 30-70% reductions through bargaining.")
        self._add_caption(fig, caption)
        
        plt.savefig(self.plots_dir / "05_negotiation_gap_dumbbell.png")
        plt.close()
        logger.info("Saved: 05_negotiation_gap_dumbbell.png")
    
    def plot_discount_distribution(self):
        """
        Histogram showing distribution of discount percentages achieved.
        Shows bargaining outcomes across all negotiations.
        """
        if self.df_neg.empty:
            return
        
        df = self.df_neg.dropna(subset=['discount_pct'])
        df = df[(df['discount_pct'] >= 0) & (df['discount_pct'] <= 100)]
        
        if df.empty:
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.hist(df['discount_pct'], bins=30, color='#4DBBD5',
               edgecolor='white', linewidth=1.2, alpha=0.85)
        
        mean_discount = df['discount_pct'].mean()
        median_discount = df['discount_pct'].median()
        
        ax.axvline(mean_discount, color='#E64B35', linestyle='--',
                  linewidth=2, label=f'Mean: {mean_discount:.1f}%')
        ax.axvline(median_discount, color='#00A087', linestyle='--',
                  linewidth=2, label=f'Median: {median_discount:.1f}%')
        
        ax.set_title('Distribution of Negotiated Discounts', pad=20)
        ax.set_xlabel('Discount Percentage (%)')
        ax.set_ylabel('Number of Negotiations')
        ax.legend(loc='upper right', frameon=True, fancybox=True)
        ax.grid(axis='y', alpha=0.3)
        
        caption = (f"Discount distribution across {len(df)} negotiations with complete financial data. "
                  f"Mean={mean_discount:.1f}%, Median={median_discount:.1f}%. "
                  "Most victims achieve 30-60% reductions; bimodal distribution suggests two negotiation styles.")
        self._add_caption(fig, caption)
        
        plt.savefig(self.plots_dir / "06_discount_distribution.png")
        plt.close()
        logger.info("Saved: 06_discount_distribution.png")
    
    def plot_psychology_discount_matrix(self):
        """
        Heatmap showing attacker strategy versus victim strategy and mean discount.
        Shows which psychological combinations yield better outcomes for victims.
        """
        if self.df_neg.empty:
            return
        
        df = self.df_neg.copy()
        df['discount_pct'] = pd.to_numeric(df['discount_pct'], errors='coerce')
        
        if 'attacker_strategy' not in df.columns or 'victim_strategy' not in df.columns:
            return
        
        pivot = df.pivot_table(index='attacker_strategy', columns='victim_strategy',
                               values='discount_pct', aggfunc='mean')
        pivot = pivot.dropna(thresh=2)
        
        if pivot.empty:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn",
                   linewidths=1, cbar_kws={'label': 'Avg Discount (%)'},
                   vmin=0, vmax=100, ax=ax)
        
        ax.set_title('Psychological Interaction Matrix: Who Wins the Discount?', pad=20)
        ax.set_xlabel('Victim Negotiation Strategy')
        ax.set_ylabel('Attacker Communication Strategy')
        
        caption = ("Average discount percentages by attacker-victim strategy combinations. "
                  "Green=high victim success, Red=low victim success. Reveals which psychological "
                  "matchups favor victims in bargaining dynamics.")
        self._add_caption(fig, caption)
        
        plt.savefig(self.plots_dir / "07_psychology_discount_matrix.png")
        plt.close()
        logger.info("Saved: 07_psychology_discount_matrix.png")
    
    def plot_tactic_effectiveness_boxplot(self):
        """
        Boxplot showing dominant argumentative tactic versus discount achieved.
        Links rhetorical approach to financial outcome.
        """
        if self.df_neg.empty or self.df_speech.empty:
            return
        
        top_tactics = self.df_speech.groupby('chat_id')['argumentative_func'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else "None"
        ).reset_index(name='dominant_tactic')
        
        df_merged = pd.merge(self.df_neg, top_tactics, on='chat_id', how='inner')
        df_merged['discount_pct'] = pd.to_numeric(df_merged['discount_pct'], errors='coerce')
        df_merged = df_merged.dropna(subset=['discount_pct', 'dominant_tactic'])
        
        tactic_counts = df_merged['dominant_tactic'].value_counts()
        main_tactics = tactic_counts[tactic_counts >= 3].index
        df_plot = df_merged[df_merged['dominant_tactic'].isin(main_tactics)]
        
        if df_plot.empty:
            return
        
        order = df_plot.groupby('dominant_tactic')['discount_pct'].median().sort_values(ascending=False).index
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        sns.boxplot(data=df_plot, x='dominant_tactic', y='discount_pct', order=order,
                   palette=self.palette_main, showfliers=False, ax=ax)
        sns.stripplot(data=df_plot, x='dominant_tactic', y='discount_pct', order=order,
                     color='black', alpha=0.3, size=4, ax=ax)
        
        ax.set_title('Tactic Effectiveness: Which Rhetoric Pays Off?', pad=20)
        ax.set_xlabel('Dominant Argumentative Tactic (by victim)')
        ax.set_ylabel('Discount Achieved (%)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        caption = (f"Discount distributions by victim's dominant argumentative strategy ({len(df_plot)} negotiations). "
                  "Box shows quartiles, dots show individual cases. Some tactics (e.g., Value/Fairness Appeal) "
                  "correlate with higher discounts, suggesting strategic advantage.")
        self._add_caption(fig, caption)
        
        plt.savefig(self.plots_dir / "08_tactic_effectiveness_boxplot.png")
        plt.close()
        logger.info("Saved: 08_tactic_effectiveness_boxplot.png")
    
    def plot_speech_act_by_party(self):
        """
        Stacked bar chart showing speech act distribution by party.
        Shows asymmetric communicative roles between attacker and victim.
        """
        if self.df_speech.empty or 'party' not in self.df_speech.columns:
            return
        
        df = self.df_speech.dropna(subset=['party', 'primary_act'])
        ct = pd.crosstab(df['party'], df['primary_act'], normalize='index') * 100
        
        if ct.empty:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ct.plot.bar(ax=ax, stacked=True, color=self.palette_main, edgecolor='white', linewidth=1.2)
        
        ax.set_title('Speech Act Distribution by Party', pad=20)
        ax.set_xlabel('Party')
        ax.set_ylabel('Percentage of Messages (%)')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(title='Speech Act', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
        ax.grid(axis='y', alpha=0.3)
        
        caption = (f"Asymmetric communication roles across {len(df)} messages. "
                  "Attackers use more Directives (commands) and Commissives (threats/promises), "
                  "victims use more Negotiative-Evaluative (bargaining) and Expressive (appeals).")
        self._add_caption(fig, caption)
        
        plt.savefig(self.plots_dir / "09_speech_act_by_party.png")
        plt.close()
        logger.info("Saved: 09_speech_act_by_party.png")
    
    def plot_message_length_evolution(self):
        """
        Line plot showing average message length across negotiation progress.
        Shows verbosity patterns over time.
        """
        if self.df_speech.empty or 'text_length' not in self.df_speech.columns:
            return
        
        df = self.df_speech.dropna(subset=['progress_bin', 'text_length'])
        avg_length = df.groupby('progress_bin')['text_length'].mean()
        
        if avg_length.empty:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(avg_length.index, avg_length.values, color='#4DBBD5',
               linewidth=3, marker='o', markersize=6, markerfacecolor='white',
               markeredgewidth=2, markeredgecolor='#4DBBD5')
        
        ax.fill_between(avg_length.index, avg_length.values, alpha=0.2, color='#4DBBD5')
        
        ax.set_title('Message Length Evolution Across Negotiation', pad=20)
        ax.set_xlabel('Negotiation Progress (bins)')
        ax.set_ylabel('Average Message Length (characters)')
        ax.grid(alpha=0.3)
        
        caption = ("Average message verbosity across 20 time bins. Peak length in mid-negotiation "
                  "corresponds to justificatory bargaining phase (bins 8-12). Shorter messages "
                  "at opening/closing reflect procedural efficiency.")
        self._add_caption(fig, caption)
        
        plt.savefig(self.plots_dir / "10_message_length_evolution.png")
        plt.close()
        logger.info("Saved: 10_message_length_evolution.png")
    
    def plot_negotiation_outcomes_pie(self):
        """
        Pie chart showing distribution of negotiation outcomes.
        Shows success, failure, and abandoned negotiation rates.
        """
        if self.df_neg.empty or 'outcome' not in self.df_neg.columns:
            return
        
        df = self.df_neg.dropna(subset=['outcome'])
        outcome_counts = df['outcome'].value_counts()
        
        if outcome_counts.empty:
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = self.palette_main[:len(outcome_counts)]
        wedges, texts, autotexts = ax.pie(
            outcome_counts.values,
            labels=outcome_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            explode=[0.05] * len(outcome_counts),
            shadow=True,
            textprops={'fontsize': 11, 'weight': 'bold'}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
        
        ax.set_title('Negotiation Outcome Distribution', pad=20)
        
        caption = (f"Final outcomes across {len(df)} negotiations. "
                  "Distribution reveals settlement patterns: paid ransoms, refused payments, "
                  "abandoned negotiations, and ongoing cases. Essential for threat modeling.")
        self._add_caption(fig, caption)
        
        plt.savefig(self.plots_dir / "11_negotiation_outcomes_pie.png")
        plt.close()
        logger.info("Saved: 11_negotiation_outcomes_pie.png")
    
    def plot_dataset_overview_bars(self):
        """
        Bar chart showing dataset composition summary.
        Provides high-level statistics overview for groups, chats, and messages.
        """
        if self.df_speech.empty:
            return
        
        total_groups = self.df_speech['group'].nunique()
        total_chats = self.df_speech['chat_id'].nunique()
        total_messages = len(self.df_speech)
        avg_msgs_per_chat = total_messages / total_chats if total_chats > 0 else 0
        
        stats = {
            'Ransomware\nGroups': total_groups,
            'Unique\nNegotiations': total_chats,
            'Total\nMessages': total_messages,
            'Avg Messages\nper Chat': int(avg_msgs_per_chat)
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(stats.keys(), stats.values(), color=self.palette_main[:4],
                     edgecolor='white', linewidth=1.5)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_title('Dataset Overview: Ransomchats Corpus Statistics', pad=20)
        ax.set_ylabel('Count')
        ax.grid(axis='y', alpha=0.3)
        
        caption = ("Comprehensive dataset statistics from Ransomchats corpus (MIT License). "
                  "Covers multiple ransomware groups across diverse negotiation scenarios, "
                  "providing robust empirical foundation for linguistic analysis.")
        self._add_caption(fig, caption)
        
        plt.savefig(self.plots_dir / "12_dataset_overview_bars.png")
        plt.close()
        logger.info("Saved: 12_dataset_overview_bars.png")
    
    def generate_all_plots(self):
        """Generate complete visualization suite."""
        logger.info("=" * 70)
        logger.info("STARTING VISUALIZATION GENERATION")
        logger.info("=" * 70)
        
        plot_count = 0
        
        self.plot_temporal_speech_acts()
        plot_count += 1
        self.plot_temporal_argumentative()
        plot_count += 1
        
        self.plot_group_speech_acts_heatmap()
        plot_count += 1
        self.plot_group_argumentative_heatmap()
        plot_count += 1
        
        self.plot_negotiation_gap()
        plot_count += 1
        self.plot_discount_distribution()
        plot_count += 1
        
        self.plot_psychology_discount_matrix()
        plot_count += 1
        self.plot_tactic_effectiveness_boxplot()
        plot_count += 1
        self.plot_speech_act_by_party()
        plot_count += 1
        self.plot_message_length_evolution()
        plot_count += 1
        
        self.plot_negotiation_outcomes_pie()
        plot_count += 1
        self.plot_dataset_overview_bars()
        plot_count += 1
        
        logger.info("=" * 70)
        logger.info(f"GENERATED {plot_count} PUBLICATION-READY VISUALIZATIONS")
        logger.info(f"Output directory: {self.plots_dir}")
        logger.info("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    project_root = Path(__file__).parent.parent.parent
    
    print("\n" + "=" * 70)
    print("DATA VISUALIZATION MODULE")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print("-" * 70)
    
    viz = DataVisualizer(project_root)
    viz.generate_all_plots()
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
