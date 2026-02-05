import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

logger = logging.getLogger(__name__)

class DataVisualizer:
    def __init__(self, base_dir: Path):
        """
        Initializes the visualizer pointing to the processed data directory.
        Sets up professional, publication-ready styling (Nature/Science inspired).
        """
        self.data_dir = base_dir / "data" / "processed"
        self.plots_dir = base_dir / "data" / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Main Datasets
        self.df_neg = self._load_csv("dataset_negotiations.csv")
        self.df_speech = self._load_csv("dataset_speech_acts.csv")
        self.df_temp_speech = self._load_csv("temporal_speech_acts.csv", index_col="progress_bin")
        self.df_temp_arg = self._load_csv("temporal_argumentative.csv", index_col="progress_bin")
        self.df_group_speech = self._load_csv("group_speech_acts.csv", index_col="group")
        self.df_group_arg = self._load_csv("group_argumentative.csv", index_col="group")

        # --- PROFESSIONAL STYLE CONFIGURATION ---
        # Using 'whitegrid' is cleaner for academic printing than dark backgrounds
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
        
        # High-contrast, colorblind-friendly palette (Npg/Lancet inspired)
        self.palette_main = [
            '#E64B35', '#4DBBD5', '#00A087', '#3C5488', 
            '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', 
            '#7E6148', '#B09C85'
        ]
        
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'figure.dpi': 300,            # High res for thesis/papers
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'axes.labelsize': 12,
            'legend.fontsize': 10,
            'legend.title_fontsize': 11,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'axes.spines.top': False,     # Minimalist spines
            'axes.spines.right': False
        })

    def _load_csv(self, filename: str, index_col=None) -> pd.DataFrame:
        path = self.data_dir / filename
        if path.exists():
            logger.info(f"Loaded dataset: {filename}")
            return pd.read_csv(path, index_col=index_col)
        else:
            logger.warning(f"Dataset {filename} not found. Run aggregator first.")
            return pd.DataFrame()

    # -------------------------------------------------------------------------
    # 1. TEMPORAL FLOW (Stream/Alluvial Proxy)
    # -------------------------------------------------------------------------
    def plot_temporal_flow(self, df_temp: pd.DataFrame, title: str, filename: str):
        """
        Stream/Flow Plot: Visualizes the evolution of discourse as a continuous flow.
        Replaces standard Stacked Area for a more organic, narrative look.
        """
        if df_temp.empty: return

        # Normalize to 100% to show relative importance over time
        df_norm = df_temp.div(df_temp.sum(axis=1), axis=0) * 100
        
        # Apply slight smoothing for "organic flow" look (optional)
        df_smooth = df_norm.rolling(window=2, min_periods=1, center=True).mean()

        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot with high transparency to emphasize overlaps/flow
        df_smooth.plot.area(ax=ax, color=self.palette_main, alpha=0.85, linewidth=0)
        
        ax.set_title(title, pad=20)
        ax.set_ylabel("Share of Conversation (%)")
        ax.set_xlabel("Conversation Progress (Normalized Time Bins)")
        ax.set_xlim(df_smooth.index.min(), df_smooth.index.max())
        ax.set_ylim(0, 100)
        
        # Minimalist Grid
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        ax.grid(axis='y', alpha=0.1)
        
        # Legend outside
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], title="Category", bbox_to_anchor=(1.01, 1), 
                  loc='upper left', frameon=False)

        out_path = self.plots_dir / filename
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Saved Flow Plot: {filename}")

    # -------------------------------------------------------------------------
    # 2. HIERARCHICAL CLUSTERMAP (Advanced Heatmap)
    # -------------------------------------------------------------------------
    def plot_tactics_clustermap(self):
        """
        Hierarchical Clustering Map: Automatically groups similar strategies and outcomes.
        Reveals hidden structural patterns in negotiation tactics.
        """
        if self.df_neg.empty: return
        
        # Use simple crosstab or normalized frequencies
        if 'victim_strategy' not in self.df_neg.columns or 'outcome' not in self.df_neg.columns:
            return

        ct = pd.crosstab(self.df_neg['victim_strategy'], self.df_neg['outcome'])
        
        # Only plot if we have enough data dimensions
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return

        # Clustermap needs to be created via seaborn directly, it manages its own figure
        g = sns.clustermap(
            ct,
            figsize=(10, 8),
            cmap="mako",            # Professional academic gradient (blue-green)
            annot=True,             # Show counts
            fmt="d", 
            linewidths=1,
            linecolor='white',
            dendrogram_ratio=(.15, .15), # Size of the clustering trees
            cbar_pos=(0.02, 0.82, 0.03, 0.15), # Move colorbar to top-left corner
            tree_kws={'linewidths': 1.5}
        )
        
        g.ax_heatmap.set_title("Clustering of Victim Strategies vs. Outcomes", pad=80, fontsize=14, fontweight='bold')
        g.ax_heatmap.set_xlabel("Negotiation Outcome")
        g.ax_heatmap.set_ylabel("Victim Strategy")
        
        out_path = self.plots_dir / "tactics_clustermap.png"
        g.savefig(out_path) # Clustermap has its own save method wrapper
        plt.close()
        logger.info(f"Saved Clustermap: tactics_clustermap.png")

    # -------------------------------------------------------------------------
    # 3. NEGOTIATION GAP (Dumbbell/Range Plot)
    # -------------------------------------------------------------------------
    def plot_negotiation_gap(self):
        """
        Range Plot (Dumbbell Chart): Shows the 'distance' between Initial Demand and Final Price.
        Much clearer than side-by-side bars for showing the 'Battleground'.
        """
        if self.df_neg.empty: return

        df = self.df_neg.dropna(subset=['initial_demand', 'final_price', 'group'])
        if df.empty: return
        
        # Aggregate median values per group
        df_grp = df.groupby('group')[['initial_demand', 'final_price']].median()
        
        # Sort by Initial Demand to order the chart
        df_grp = df_grp.sort_values('initial_demand', ascending=True)
        # Filter top 15 groups to avoid overcrowding
        df_grp = df_grp.tail(15)

        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw the connecting line (The "Wick")
        ax.hlines(y=df_grp.index, xmin=df_grp['final_price'], xmax=df_grp['initial_demand'], 
                  color='grey', alpha=0.5, linewidth=2)
        
        # Draw the points
        ax.scatter(df_grp['initial_demand'], df_grp.index, color='#E64B35', alpha=1, s=100, label='Initial Demand', zorder=3)
        ax.scatter(df_grp['final_price'], df_grp.index, color='#00A087', alpha=1, s=100, label='Final Settlement', zorder=3)
        
        # Log scale is essential for Ransomware
        ax.set_xscale('log')
        
        ax.set_title("The Negotiation Gap: Initial Demand vs. Settlement (Median)", pad=15)
        ax.set_xlabel("Amount (USD) - Log Scale")
        ax.set_ylabel("Ransomware Group")
        
        # Add legend
        ax.legend(loc='lower right', frameon=True)

        # Annotate Discount %
        for i, (group, row) in enumerate(df_grp.iterrows()):
            if row['initial_demand'] > 0:
                discount = (1 - row['final_price'] / row['initial_demand']) * 100
                # Position text slightly above the line
                if discount > 0:
                    ax.text(row['initial_demand'] * 1.15, i, f"-{discount:.0f}%", 
                            va='center', color='#E64B35', fontsize=9, fontweight='bold')

        out_path = self.plots_dir / "negotiation_gap_dumbbell.png"
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Saved Negotiation Gap Plot: negotiation_gap_dumbbell.png")

        # -------------------------------------------------------------------------
    # 4. CROSS-ANALYSIS: PSYCHOLOGY vs. DISCOUNT (Matrix)
    # -------------------------------------------------------------------------
    def plot_psych_interaction_matrix(self):
        """
        Interaction Matrix: Attacker Strategy vs. Victim Strategy -> Mean Discount.
        Shows which combination of personalities yields the best financial outcome.
        """
        if self.df_neg.empty: return
        
        # Ensure numeric discount
        df = self.df_neg.copy()
        df['discount_pct'] = pd.to_numeric(df['discount_pct'], errors='coerce')
        
        # Filter necessary columns
        if 'attacker_strategy' not in df.columns or 'victim_strategy' not in df.columns:
            return

        # Create pivot table: Rows=Attacker, Cols=Victim, Values=Mean Discount
        pivot_table = df.pivot_table(
            index='attacker_strategy', 
            columns='victim_strategy', 
            values='discount_pct', 
            aggfunc='mean'
        )
        
        # Filter sparse data (keep only strategies with enough interactions)
        pivot_table = pivot_table.dropna(thresh=1) 

        if pivot_table.empty: return

        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Heatmap with divergent color map (Red=Low Discount, Green=High Discount)
        sns.heatmap(
            pivot_table, 
            annot=True, 
            fmt=".1f", 
            cmap="RdYlGn", # Red to Green
            linewidths=1, 
            linecolor='white',
            cbar_kws={'label': 'Average Discount (%)'},
            ax=ax
        )
        
        ax.set_title("Psychological Interaction Matrix: Who wins the discount?", pad=20)
        ax.set_ylabel("Attacker Psychology (Profile)")
        ax.set_xlabel("Victim Negotiation Strategy")
        
        out_path = self.plots_dir / "psych_interaction_discount_matrix.png"
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Saved Psych Interaction Matrix.")

    # -------------------------------------------------------------------------
    # 5. CROSS-ANALYSIS: SPEECH ACTS vs. DISCOUNT (Box Plot)
    # -------------------------------------------------------------------------
    def plot_tactic_effectiveness(self):
        """
        Effectiveness Plot: Dominant Argumentative Tactic vs. Discount %.
        Links the linguistic/rhetorical approach (Speech Acts) to the financial result (Technical).
        """
        # We need to merge Speech Acts with Negotiations to link Tactics to Discounts
        if self.df_neg.empty or self.df_speech.empty: return
        
        # 1. Calculate the 'Dominant' tactic for each chat from the Speech dataset
        # (The tactic used most frequently in that specific chat)
        top_tactics = self.df_speech.groupby('chat_id')['argumentative_function'].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else "None"
        ).reset_index(name='dominant_tactic')
        
        # 2. Merge with Negotiation dataset to get the discount
        df_merged = pd.merge(self.df_neg, top_tactics, on='chat_id', how='inner')
        
        # 3. Clean and Filter
        df_merged['discount_pct'] = pd.to_numeric(df_merged['discount_pct'], errors='coerce')
        df_merged = df_merged.dropna(subset=['discount_pct', 'dominant_tactic'])
        
        # Filter out rare tactics (noise)
        tactic_counts = df_merged['dominant_tactic'].value_counts()
        main_tactics = tactic_counts[tactic_counts > 2].index # Keep tactics appearing in at least 3 chats
        df_plot = df_merged[df_merged['dominant_tactic'].isin(main_tactics)]

        if df_plot.empty: return

        # Sort by median discount for better readability
        order = df_plot.groupby('dominant_tactic')['discount_pct'].median().sort_values(ascending=False).index

        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Boxplot shows the distribution (Median, range, outliers)
        sns.boxplot(
            data=df_plot, 
            x='dominant_tactic', 
            y='discount_pct', 
            order=order,
            palette=self.palette_main,
            showfliers=False, # Hide outliers to keep scale clean
            ax=ax
        )
        
        # Add strip plot to show actual data points (transparency helps with overlap)
        sns.stripplot(
            data=df_plot, 
            x='dominant_tactic', 
            y='discount_pct', 
            order=order,
            color='black', 
            alpha=0.3, 
            size=4,
            ax=ax
        )

        ax.set_title("Which Rhetoric Pays Off? Tactic Effectiveness on Discounts", pad=20)
        ax.set_ylabel("Discount Obtained (%)")
        ax.set_xlabel("Dominant Argumentative Strategy used by Victim")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        out_path = self.plots_dir / "tactic_effectiveness_boxplot.png"
        plt.savefig(out_path)
        plt.close()
        logger.info(f"Saved Tactic Effectiveness Plot.")



    # -------------------------------------------------------------------------
    # MAIN GENERATOR
    # -------------------------------------------------------------------------
    def generate_all_plots(self):
        """Entry point to generate all enhanced visualizations."""
        logger.info("Starting advanced visualization generation...")
        
        # 1. New Temporal Flows
        self.plot_temporal_flow(
            self.df_temp_speech, 
            "Temporal Flow of Speech Acts", 
            "flow_speech_acts.png"
        )
        self.plot_temporal_flow(
            self.df_temp_arg, 
            "Temporal Flow of Argumentative Tactics", 
            "flow_argumentative.png"
        )
        
        # 2. New Clustermap
        self.plot_tactics_clustermap()
        
        # 3. New Negotiation Gap
        self.plot_negotiation_gap()

        # 4. New Interaction Matrix
        self.plot_psych_interaction_matrix()
        
        # 5. New Tactic Effectiveness
        self.plot_tactic_effectiveness()
        
        logger.info("All advanced plots generated.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    project_root = Path(__file__).parent.parent.parent
    viz = DataVisualizer(project_root)
    viz.generate_all_plots()
