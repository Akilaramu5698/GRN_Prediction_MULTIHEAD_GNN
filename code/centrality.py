"""
CENTRALITY ANALYSIS PIPELINE - WITH ACTUAL GENE NAMES (FIXED)
Loads saved results and performs comprehensive gene centrality analysis
Creates SEPARATE figures to avoid overlapping - ALL FIGURES USE TOP 20 GENES
"""

import numpy as np
import pandas as pd
import networkx as nx
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import glob
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
class CentralityConfig:
    """Configuration for centrality analysis"""
    
    # Paths
    RESULTS_DIR = "./dream5_results"
    CENTRALITY_DIR = "./centrality_analysis"
    FIGURES_DIR = "./figures"
    
    # Gene ID mapping files
    GENE_MAP_FILES = {
        'in_silico': './in_silico_gene_ids.csv',
        'e_coli': './e_coli_gene_ids.csv',
        's_cerevisiae': './s_cerevisiae_gene_ids.csv'
    }
    
    # Analysis parameters
    PREDICTION_THRESHOLD = 0.8
    TOP_K_GENES = 50
    
    CENTRALITY_MEASURES = [
        'degree', 'in_degree', 'out_degree', 'betweenness', 
        'closeness', 'pagerank', 'eigenvector', 'clustering'
    ]

# ============================================================================
# GENE NAME MAPPING
# ============================================================================
class GeneNameMapper:
    """Map gene IDs to actual gene names"""
    
    def __init__(self, config):
        self.config = config
        self.gene_maps = {}
        self.load_all_gene_maps()
    
    def load_gene_map(self, dataset_name):
        """Load gene ID to name mapping for a dataset"""
        if dataset_name not in self.config.GENE_MAP_FILES:
            print(f"   No gene map file for {dataset_name}")
            return None
        
        map_file = self.config.GENE_MAP_FILES[dataset_name]
        if not os.path.exists(map_file):
            print(f"   Gene map file not found: {map_file}")
            return None
        
        try:
            # Load the CSV file
            df = pd.read_csv(map_file)
            
            # Check columns - assuming first column is ID, second is Name
            if len(df.columns) >= 2:
                id_col = df.columns[0]
                name_col = df.columns[1]
                
                # Create mapping dictionary
                gene_map = {}
                for _, row in df.iterrows():
                    gene_id = str(row[id_col]).strip()
                    gene_name = str(row[name_col]).strip()
                    gene_map[gene_id] = gene_name
                
                print(f"   Loaded {len(gene_map)} gene names for {dataset_name}")
                return gene_map
            else:
                print(f"   Gene map file has {len(df.columns)} columns, expected at least 2")
                return None
                
        except Exception as e:
            print(f"   Error loading gene map: {e}")
            return None
    
    def load_all_gene_maps(self):
        """Load gene maps for all datasets"""
        for dataset in ['in_silico', 'e_coli', 's_cerevisiae']:
            self.gene_maps[dataset] = self.load_gene_map(dataset)
    
    def get_gene_name(self, dataset_name, gene_idx, default_prefix="G"):
        """Convert gene index to actual gene name"""
        
        # Get gene map for this dataset
        gene_map = self.gene_maps.get(dataset_name, {})
        if gene_map is None:
            return f"G{gene_idx}"
        
        # The gene ID in your results is like "G0", "G1", etc.
        gene_id = f"G{gene_idx}"
        
        # Look up the actual name
        if gene_id in gene_map:
            return gene_map[gene_id]
        else:
            # Fall back to G{idx} if not found
            return f"G{gene_idx}"

# ============================================================================
# DATA LOADER FOR SAVED RESULTS
# ============================================================================
class ResultLoader:
    """Load saved results from main pipeline"""
    
    def __init__(self, config, gene_mapper):
        self.config = config
        self.gene_mapper = gene_mapper
        os.makedirs(config.CENTRALITY_DIR, exist_ok=True)
        os.makedirs(config.FIGURES_DIR, exist_ok=True)
    
    def find_latest_results(self, dataset_name):
        """Find the most recent results file for a dataset"""
        pattern = os.path.join(self.config.RESULTS_DIR, f"{dataset_name}_results_*.pkl")
        files = glob.glob(pattern)
        
        if not files:
            print(f" No results found for {dataset_name}")
            return None
        
        latest_file = max(files, key=os.path.getctime)
        print(f" Loading: {os.path.basename(latest_file)}")
        return latest_file
    
    def load_results(self, dataset_name):
        """Load results for a specific dataset"""
        filepath = self.find_latest_results(dataset_name)
        if filepath is None:
            return None
        
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        
        print(f"Loaded results for {dataset_name}")
        return results
    
    def load_all_results(self):
        """Load all available results"""
        datasets = ['in_silico', 'e_coli', 's_cerevisiae']
        all_results = {}
        
        for dataset in datasets:
            results = self.load_results(dataset)
            if results is not None:
                all_results[dataset] = results
        
        return all_results

# ============================================================================
# CENTRALITY ANALYZER
# ============================================================================
class CentralityAnalyzer:
    """Analyze gene importance using multiple centrality measures"""
    
    def __init__(self, dataset_name, gene_names, gene_mapper, config):
        """
        Args:
            dataset_name: name of the dataset
            gene_names: list of gene indices
            gene_mapper: GeneNameMapper object
            config: CentralityConfig object
        """
        self.dataset_name = dataset_name
        self.gene_names = gene_names
        self.gene_mapper = gene_mapper
        self.n_genes = len(gene_names)
        self.config = config
        self.threshold = config.PREDICTION_THRESHOLD
        self.G = None
        self.centralities = {}
        
    def get_actual_gene_name(self, idx):
        """Get actual gene name for an index"""
        return self.gene_mapper.get_gene_name(self.dataset_name, idx)
    
    def build_network_from_predictions(self, edge_pairs, predictions):
        """Build directed graph from GNN predictions"""
        
        print(f"\nBuilding gene regulatory network...")
        print(f"  Total predictions: {len(predictions)}")
        print(f"  Threshold: ≥ {self.threshold}")
        
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(self.n_genes))
        
        edge_count = 0
        high_conf_edges = []
        
        for (u, v), prob in zip(edge_pairs, predictions):
            if prob >= self.threshold:
                u_int, v_int = int(u), int(v)
                self.G.add_edge(u_int, v_int, weight=prob, confidence=prob)
                edge_count += 1
                
                # Store with actual gene names
                u_name = self.get_actual_gene_name(u_int)
                v_name = self.get_actual_gene_name(v_int)
                high_conf_edges.append((u_name, v_name, prob))
        
        print(f"\nNetwork Statistics:")
        print(f"  • Nodes: {self.G.number_of_nodes():,}")
        print(f"  • Edges (≥{self.threshold}): {edge_count:,}")
        print(f"  • Density: {nx.density(self.G):.6f}")
        
        if edge_count > 0:
            degrees = [d for n, d in self.G.degree()]
            print(f"  • Avg degree: {np.mean(degrees):.2f}")
            print(f"  • Max degree: {max(degrees)}")
        
        return self.G, high_conf_edges
    
    def compute_all_centralities(self):
        """Compute all centrality measures"""
        print("\n Computing centrality measures...")
        
        if self.G is None or self.G.number_of_edges() == 0:
            print("   No network to analyze")
            return {}
        
        print("  Degree centrality")
        self.centralities['degree'] = nx.degree_centrality(self.G)
        
        print("   In-degree centrality)
        in_degrees = dict(self.G.in_degree())
        max_in = max(in_degrees.values()) if in_degrees else 1
        self.centralities['in_degree'] = {k: v/max_in for k, v in in_degrees.items()}
        
        print("  Out-degree centrality")
        out_degrees = dict(self.G.out_degree())
        max_out = max(out_degrees.values()) if out_degrees else 1
        self.centralities['out_degree'] = {k: v/max_out for k, v in out_degrees.items()}
        
        print("  Betweenness centrality")
        self.centralities['betweenness'] = nx.betweenness_centrality(self.G, k=100)
        
        print("  Closeness centrality")
        self.centralities['closeness'] = nx.closeness_centrality(self.G)
        
        print("  PageRank")
        self.centralities['pagerank'] = nx.pagerank(self.G)
        
        print("  Eigenvector centrality")
        try:
            self.centralities['eigenvector'] = nx.eigenvector_centrality_numpy(self.G)
        except:
            try:
                self.centralities['eigenvector'] = nx.eigenvector_centrality(self.G, max_iter=1000)
            except:
                self.centralities['eigenvector'] = {i: 0 for i in range(self.n_genes)}
        
        print("  Clustering coefficient")
        G_undirected = self.G.to_undirected()
        self.centralities['clustering'] = nx.clustering(G_undirected)
        
        print("  Load centrality")
        try:
            self.centralities['load'] = nx.load_centrality(self.G, k=100)
        except:
            self.centralities['load'] = {i: 0 for i in range(self.n_genes)}
        
        return self.centralities
    
    def create_ranking_dataframe(self):
        """Create comprehensive ranking of genes by centrality with actual names"""
        
        if not self.centralities:
            print("   No centralities computed")
            return None
        
        print("\n Creating gene rankings...")
        
        # Create dataframe with actual gene names
        data = []
        for i in range(self.n_genes):
            row = {
                'Index': i,
                'Gene_ID': f"G{i}",
                'Gene_Name': self.get_actual_gene_name(i)
            }
            for measure, values in self.centralities.items():
                row[measure] = values.get(i, 0)
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Add rankings for each measure
        for measure in self.centralities.keys():
            df[f'{measure}_rank'] = df[measure].rank(ascending=False, method='min')
        
        # Calculate composite score
        rank_cols = [f'{m}_rank' for m in self.centralities.keys()]
        df['composite_rank'] = df[rank_cols].mean(axis=1)
        
        min_rank = df['composite_rank'].min()
        max_rank = df['composite_rank'].max()
        df['composite_score'] = 1 - ((df['composite_rank'] - min_rank) / (max_rank - min_rank))
        
        # Sort by composite score
        df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)
        
        return df
    
    def identify_key_genes(self, df, top_k=20):
        """Identify key regulatory genes with actual names"""
        
        if df is None:
            return None
        
        top_genes = df.head(top_k)
        
        # Hubs (top 10% by degree)
        degree_dict = self.centralities.get('degree', {})
        degree_threshold = np.percentile(list(degree_dict.values()), 90) if degree_dict else 0
        hubs = [i for i in range(self.n_genes) 
                if degree_dict.get(i, 0) >= degree_threshold]
        
        # Bottlenecks (top 10% by betweenness)
        between_dict = self.centralities.get('betweenness', {})
        between_threshold = np.percentile(list(between_dict.values()), 90) if between_dict else 0
        bottlenecks = [i for i in range(self.n_genes) 
                      if between_dict.get(i, 0) >= between_threshold]
        
        # Master regulators (top 10% by out-degree)
        out_dict = self.centralities.get('out_degree', {})
        out_threshold = np.percentile(list(out_dict.values()), 90) if out_dict else 0
        master_regs = [i for i in range(self.n_genes) 
                      if out_dict.get(i, 0) >= out_threshold]
        
        return {
            'top_genes': top_genes,
            'hubs': [self.get_actual_gene_name(i) for i in hubs[:20]],
            'bottlenecks': [self.get_actual_gene_name(i) for i in bottlenecks[:20]],
            'master_regulators': [self.get_actual_gene_name(i) for i in master_regs[:20]]
        }

# ============================================================================
# SEPARATE VISUALIZATIONS - ALL USING TOP 20 GENES
# ============================================================================
class CentralityVisualizer:
    """Create separate, clean visualizations without overlapping"""
    
    def __init__(self, config):
        self.config = config
    
    def plot_top_genes_barchart(self, df, dataset_name):
        """FIGURE 1: Top 20 genes bar chart"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top20 = df.head(20)
        y_labels = [f"{row['Gene_Name']} ({row['Gene_ID']})" for _, row in top20.iterrows()]
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, 20))
        bars = ax.barh(range(20), top20['composite_score'].values, color=colors)
        
        ax.set_yticks(range(20))
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_xlabel('Composite Centrality Score', fontsize=12)
        ax.set_title(f'Top 20 Genes by Composite Centrality - {dataset_name.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, top20['composite_score'].values)):
            ax.text(score + 0.01, i, f'{score:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        
        # Save
        output_dir = os.path.join(self.config.FIGURES_DIR, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, '01_top_20_genes_barchart.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"   Figure 1 saved: {save_path}")
    
    def plot_centrality_heatmap(self, df, dataset_name):
        """FIGURE 2: Centrality heatmap for top 20 genes"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        measures = ['degree', 'betweenness', 'pagerank', 'closeness', 'eigenvector']
        top20 = df.head(20)
        
        # Prepare data
        heatmap_data = []
        gene_labels = []
        for _, row in top20.iterrows():
            heatmap_data.append([row.get(m, 0) for m in measures])
            gene_labels.append(row['Gene_Name'])
        
        # Create heatmap
        im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
        
        # Customize
        ax.set_yticks(range(20))
        ax.set_yticklabels(gene_labels, fontsize=9)
        ax.set_xticks(range(len(measures)))
        ax.set_xticklabels(measures, fontsize=10)
        ax.set_title(f'Centrality Profiles of Top 20 Genes - {dataset_name.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Centrality Score')
        
        plt.tight_layout()
        
        # Save
        output_dir = os.path.join(self.config.FIGURES_DIR, dataset_name)
        save_path = os.path.join(output_dir, '02_top_20_centrality_heatmap.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Figure 2 saved: {save_path}")
    
    def plot_degree_distribution(self, analyzer, dataset_name):
        """FIGURE 3: Degree distribution"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if analyzer.G.number_of_edges() > 0:
            degrees = [d for n, d in analyzer.G.degree()]
            
            # Histogram
            ax.hist(degrees, bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Degree', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'Degree Distribution - {dataset_name.replace("_", " ").title()}', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Stats box
            stats_text = f"Nodes: {analyzer.G.number_of_nodes():,}\n"
            stats_text += f"Edges: {analyzer.G.number_of_edges():,}\n"
            stats_text += f"Avg Degree: {np.mean(degrees):.2f}\n"
            stats_text += f"Max Degree: {max(degrees)}"
            
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                    ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No edges in network', ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        
        # Save
        output_dir = os.path.join(self.config.FIGURES_DIR, dataset_name)
        save_path = os.path.join(output_dir, '03_degree_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Figure 3 saved: {save_path}")
    
    def plot_degree_vs_betweenness(self, df, analyzer, dataset_name):
        """FIGURE 4: Degree vs Betweenness scatter plot with top 20 highlighted"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if analyzer.G.number_of_edges() > 0:
            # All genes
            ax.scatter(df['degree'].values, df['betweenness'].values, 
                      alpha=0.3, c='#2E86AB', s=20, label='All genes')
            
            # Highlight top 20
            top20 = df.head(20)
            ax.scatter(top20['degree'].values, top20['betweenness'].values, 
                      c='red', s=100, marker='*', label='Top 20 Genes', 
                      edgecolors='black', linewidth=1)
            
            # Add labels for top 10 (to avoid overcrowding)
            for i, row in top20.head(10).iterrows():
                ax.annotate(row['Gene_Name'], 
                           (row['degree'] + 0.005, row['betweenness'] + 0.005),
                           fontsize=8, 
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
            
            ax.set_xlabel('Degree Centrality', fontsize=12)
            ax.set_ylabel('Betweenness Centrality', fontsize=12)
            ax.set_title(f'Degree vs Betweenness (Top 20 Highlighted) - {dataset_name.replace("_", " ").title()}', 
                        fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No edges in network', ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        
        # Save
        output_dir = os.path.join(self.config.FIGURES_DIR, dataset_name)
        save_path = os.path.join(output_dir, '04_degree_vs_betweenness.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Figure 4 saved: {save_path}")
    
    def plot_top_genes_table(self, df, dataset_name):
        """FIGURE 5: Top 20 genes table"""
        
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')
        
        # Prepare table data - now using top 20
        table_data = df.head(20)[['Gene_Name', 'Gene_ID', 'degree', 'betweenness', 'pagerank', 'composite_score']].round(4)
        table_data['degree'] = (table_data['degree'] * 100).round(2)
        table_data['betweenness'] = (table_data['betweenness'] * 100).round(4)
        table_data['pagerank'] = (table_data['pagerank'] * 100).round(3)
        table_data['composite_score'] = table_data['composite_score'].round(3)
        
        # Create display labels
        table_data['Gene'] = table_data['Gene_Name'] + ' (' + table_data['Gene_ID'] + ')'
        display_data = table_data[['Gene', 'degree', 'betweenness', 'pagerank', 'composite_score']]
        
        # Create table
        table = ax.table(cellText=display_data.values,
                        colLabels=['Gene', 'Degree(%)', 'Between(%)', 'PageRank(%)', 'Score'],
                        loc='center',
                        cellLoc='center',
                        colWidths=[0.3, 0.15, 0.2, 0.15, 0.1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style header
        for i in range(5):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        ax.set_title(f'Top 20 Genes - Detailed Centrality Values - {dataset_name.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save
        output_dir = os.path.join(self.config.FIGURES_DIR, dataset_name)
        save_path = os.path.join(output_dir, '05_top_20_genes_table.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Figure 5 saved: {save_path}")
    
    def plot_network_stats(self, analyzer, key_genes, dataset_name):
        """FIGURE 6: Network statistics and key genes"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        if analyzer.G.number_of_edges() > 0:
            degrees = [d for n, d in analyzer.G.degree()]
            
            stats_text = f"""
            {'='*35}
            NETWORK STATISTICS - {dataset_name.replace('_', ' ').title()}
            {'='*35}
            
            • Nodes: {analyzer.G.number_of_nodes():,}
            • Edges: {analyzer.G.number_of_edges():,}
            • Density: {nx.density(analyzer.G):.6f}
            • Avg Degree: {np.mean(degrees):.2f}
            • Max Degree: {max(degrees)}
            
            
            {'='*35}
            KEY REGULATORS IDENTIFIED
            {'='*35}
            
            MASTER REGULATORS (High Out-degree):
            {chr(10).join(['  • ' + name for name in key_genes['master_regulators'][:8]])}
            
             HUBS (High Degree):
            {chr(10).join(['  • ' + name for name in key_genes['hubs'][:8]])}
            
             BOTTLENECKS (High Betweenness):
            {chr(10).join(['  • ' + name for name in key_genes['bottlenecks'][:8]])}
            """
        else:
            stats_text = f"No network edges found for {dataset_name}"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=11, va='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='#FFF3CD', alpha=0.9))
        
        plt.tight_layout()
        
        # Save
        output_dir = os.path.join(self.config.FIGURES_DIR, dataset_name)
        save_path = os.path.join(output_dir, '06_network_statistics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Figure 6 saved: {save_path}")

# ============================================================================
# MAIN CENTRALITY PIPELINE
# ============================================================================
class CentralityPipeline:
    """Main pipeline for centrality analysis"""
    
    def __init__(self, config):
        self.config = config
        self.gene_mapper = GeneNameMapper(config)
        self.loader = ResultLoader(config, self.gene_mapper)
        self.visualizer = CentralityVisualizer(config)
        self.all_results = {}
    
    def run_for_dataset(self, dataset_name, results):
        """Run centrality analysis for a single dataset"""
        
        print(f"\n{'='*70}")
        print(f" CENTRALITY ANALYSIS: {dataset_name.upper()}")
        print('='*70)
        
        if 'gnn' not in results:
            print("   No GNN results found")
            return None, None, None
        
        gnn_data = results['gnn']
        
        if 'all_predictions' not in gnn_data or 'all_edge_pairs' not in gnn_data:
            print("   No predictions stored in results")
            return None, None, None
        
        predictions = np.array(gnn_data['all_predictions'])
        edge_pairs = np.array(gnn_data['all_edge_pairs'])
        gene_indices = list(range(results['metadata']['n_genes']))
        
        print(f"\n Data loaded:")
        print(f"  Genes: {len(gene_indices):,}")
        print(f"  Predictions: {len(predictions):,}")
        print(f"  Edge pairs: {len(edge_pairs):,}")
        
        analyzer = CentralityAnalyzer(dataset_name, gene_indices, self.gene_mapper, self.config)
        
        G, high_conf_edges = analyzer.build_network_from_predictions(edge_pairs, predictions)
        
        centralities = analyzer.compute_all_centralities()
        
        if not centralities:
            print("  Failed to compute centralities")
            return None, None, None
        
        df = analyzer.create_ranking_dataframe()
        
        key_genes = analyzer.identify_key_genes(df, top_k=20)
        
        self._save_centrality_results(dataset_name, analyzer, df, key_genes, high_conf_edges)
        
        # Create SEPARATE visualizations (6 individual files) - ALL USING TOP 20
        print("\n Creating separate visualizations (all using Top 20 genes)...")
        self.visualizer.plot_top_genes_barchart(df, dataset_name)
        self.visualizer.plot_centrality_heatmap(df, dataset_name)
        self.visualizer.plot_degree_distribution(analyzer, dataset_name)
        self.visualizer.plot_degree_vs_betweenness(df, analyzer, dataset_name)
        self.visualizer.plot_top_genes_table(df, dataset_name)
        self.visualizer.plot_network_stats(analyzer, key_genes, dataset_name)
        
        self.all_results[dataset_name] = (df, analyzer, key_genes)
        
        self._print_summary(dataset_name, df, key_genes)
        
        return df, analyzer, key_genes
    
    def _save_centrality_results(self, dataset_name, analyzer, df, key_genes, high_conf_edges):
        """Save all centrality results to files"""
        
        # Save full top genes CSV
        csv_path = os.path.join(self.config.CENTRALITY_DIR, f'{dataset_name}_top_{self.config.TOP_K_GENES}_genes.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n Top genes saved: {csv_path}")
        
        # Save simplified version
        simple_df = df[['Gene_ID', 'Gene_Name', 'degree', 'betweenness', 'pagerank', 'composite_score']].head(50)
        simple_path = os.path.join(self.config.CENTRALITY_DIR, f'{dataset_name}_top_genes_simple.csv')
        simple_df.to_csv(simple_path, index=False)
        print(f" Simple gene list saved: {simple_path}")
        
        # Save results dictionary
        results_dict = {
            'dataset': dataset_name,
            'threshold': self.config.PREDICTION_THRESHOLD,
            'network_stats': {
                'nodes': analyzer.G.number_of_nodes(),
                'edges': analyzer.G.number_of_edges(),
                'density': nx.density(analyzer.G)
            },
            'top_genes': df.head(50).to_dict('records'),
            'key_genes': key_genes,
            'centralities': {k: dict(v) for k, v in analyzer.centralities.items()}
        }
        
        pkl_path = os.path.join(self.config.CENTRALITY_DIR, f'{dataset_name}_centrality_results.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(results_dict, f)
        print(f" Full results saved: {pkl_path}")
        
        # Save high confidence edges
        if high_conf_edges:
            edges_df = pd.DataFrame(high_conf_edges, columns=['source_name', 'target_name', 'confidence'])
            edges_path = os.path.join(self.config.CENTRALITY_DIR, f'{dataset_name}_high_confidence_edges.csv')
            edges_df.to_csv(edges_path, index=False)
            print(f" High-confidence edges saved: {edges_path}")
    
    def _print_summary(self, dataset_name, df, key_genes):
        """Print summary of findings with actual names"""
        
        print(f"\n{'='*70}")
        print(f" SUMMARY - {dataset_name.upper()}")
        print('='*70)
        
        print("\n TOP 10 REGULATORY GENES:")
        print("-" * 80)
        print(f"{'Rank':<6} {'Gene Name':<15} {'ID':<8} {'Degree':<10} {'Betweenness':<12} {'Score':<8}")
        print("-" * 80)
        
        for i, row in df.head(10).iterrows():
            print(f"{i+1:<6} {row['Gene_Name']:<15} {row['Gene_ID']:<8} "
                  f"{row['degree']:<10.3f} {row['betweenness']:<12.3f} "
                  f"{row['composite_score']:<8.3f}")
        
        print("\nKEY GENE CATEGORIES:")
        print(f"  • Master Regulators: {len(key_genes['master_regulators'])}")
        print(f"    Top 5: {', '.join(key_genes['master_regulators'][:5])}")
        print(f"  • Hubs: {len(key_genes['hubs'])}")
        print(f"    Top 5: {', '.join(key_genes['hubs'][:5])}")
        print(f"  • Bottlenecks: {len(key_genes['bottlenecks'])}")
        print(f"    Top 5: {', '.join(key_genes['bottlenecks'][:5])}")
    
    def run_all(self):
        """Run centrality analysis on all available datasets"""
        
        print("\n" + "="*80)
        print(" CENTRALITY ANALYSIS PIPELINE")
        print("="*80)
        
        all_results = self.loader.load_all_results()
        
        if not all_results:
            print("\n No results found to analyze")
            return
        
        for dataset_name, results in all_results.items():
            self.run_for_dataset(dataset_name, results)
        
        print("\n" + "="*80)
        print(" CENTRALITY ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\n Results saved in: {self.config.CENTRALITY_DIR}/")
        print(f"Figures saved in: {self.config.FIGURES_DIR}/")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    config = CentralityConfig()
    pipeline = CentralityPipeline(config)
    pipeline.run_all()
