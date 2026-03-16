"""
Complete DREAM5 Gene Regulatory Network Inference Pipeline
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                           precision_recall_curve, f1_score, accuracy_score)
from scipy.stats import rankdata, pearsonr, wilcoxon
from scipy.spatial.distance import pdist, squareform
from collections import Counter, defaultdict
import warnings
import os
import pickle
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """Centralized configuration for the entire pipeline"""
    
    # File paths
    BASE_PATH = "/media/akila/685807bf-be49-4441-859a-dc4d780b53cd/PhD/GRN.DL/e_coli"
    
    DREAM5_DATASETS = {
        'in_silico': {
            'expr': os.path.join(BASE_PATH, 'in_silico_expression_data.csv'),
            'gold': os.path.join(BASE_PATH, 'in_silico_gold_standard.csv')
        },
        'e_coli': {
            'expr': os.path.join(BASE_PATH, 'e_coli_expression_data.csv'),
            'gold': os.path.join(BASE_PATH, 'e_coli_gold_standard.csv')
        },
        's_cerevisiae': {
            'expr': os.path.join(BASE_PATH, 's_cerevisiae_expression_data.csv'),
            'gold': os.path.join(BASE_PATH, 's_cerevisiae_gold_standard.csv')
        }
    }
    
    # Model architecture
    HIDDEN_DIM = 128
    LATENT_DIM = 64
    DROPOUT = 0.3
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 5e-4
    EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    N_FOLDS = 5
    RANDOM_SEED = 42
    
    # Output directories
    OUTPUT_DIR = "./dream5_results"
    FIGURES_DIR = "./figures"
    MODELS_DIR = "./trained_models"

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
class RegulatoryFeatureExtractor:
    """Extract comprehensive features for gene pairs"""
    
    def __init__(self, expression_matrix, gene_names=None):
        self.expression = expression_matrix
        self.n_genes = expression_matrix.shape[0]
        self.n_samples = expression_matrix.shape[1]
        self.gene_names = gene_names if gene_names else [f"G{i}" for i in range(self.n_genes)]
        self.feature_names = []
        
    def compute_features_for_pairs(self, edge_pairs):
        """
        Compute 9 biologically-motivated features for each edge pair
        """
        print(f"  Computing features for {len(edge_pairs)} gene pairs...")
        
        n_edges = len(edge_pairs)
        features_list = []
        self.feature_names = [
            'pearson', 'spearman', 'mutual_info',
            'mean_diff', 'std_diff', 'max_diff', 'min_diff',
            'euclidean', 'cosine'
        ]
        
        for idx, (i, j) in enumerate(edge_pairs):
            if idx % 1000 == 0 and idx > 0:
                print(f"    Progress: {idx}/{n_edges}")
            
            gene_i = self.expression[i]
            gene_j = self.expression[j]
            
            features = []
            
            # 1. Pearson correlation
            corr = np.corrcoef(gene_i, gene_j)[0, 1]
            features.append(corr if not np.isnan(corr) else 0.0)
            
            # 2. Spearman correlation
            features.append(self._spearman(gene_i, gene_j))
            
            # 3. Mutual information
            features.append(self._mutual_info(gene_i, gene_j))
            
            # 4-7. Expression difference statistics
            diff = np.abs(gene_i - gene_j)
            features.extend([
                np.mean(diff),
                np.std(diff),
                np.max(diff),
                np.min(diff)
            ])
            
            # 8. Euclidean distance
            features.append(np.linalg.norm(gene_i - gene_j))
            
            # 9. Cosine similarity
            norm_i = np.linalg.norm(gene_i) + 1e-8
            norm_j = np.linalg.norm(gene_j) + 1e-8
            features.append(np.dot(gene_i, gene_j) / (norm_i * norm_j))
            
            features_list.append(features)
        
        X = np.array(features_list)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"  Feature matrix shape: {X.shape}")
        return X
    
    def _spearman(self, x, y):
        x_rank = rankdata(x)
        y_rank = rankdata(y)
        corr = pearsonr(x_rank, y_rank)[0]
        return corr if not np.isnan(corr) else 0.0
    
    def _mutual_info(self, x, y, n_bins=10):
        try:
            x_bins = np.digitize(x, np.percentile(x, np.linspace(0, 100, n_bins+1)[1:-1]))
            y_bins = np.digitize(y, np.percentile(y, np.linspace(0, 100, n_bins+1)[1:-1]))
            
            contingency = np.zeros((n_bins, n_bins))
            for k in range(len(x)):
                if x_bins[k] < n_bins and y_bins[k] < n_bins:
                    contingency[x_bins[k], y_bins[k]] += 1
            
            total = np.sum(contingency)
            if total == 0:
                return 0.0
                
            mi = 0
            for i in range(n_bins):
                for j in range(n_bins):
                    if contingency[i, j] > 0:
                        p_ij = contingency[i, j] / total
                        p_i = np.sum(contingency[i, :]) / total
                        p_j = np.sum(contingency[:, j]) / total
                        if p_i > 0 and p_j > 0:
                            mi += p_ij * np.log2(p_ij / (p_i * p_j) + 1e-10)
            return mi
        except:
            return 0.0

# ============================================================================
# DATA LOADER
# ============================================================================
class DataLoader:
    """Load and preprocess DREAM5 data"""
    
    def __init__(self, config):
        self.config = config
        
    def load_dataset(self, dataset_name):
        """Load expression and gold standard data"""
        dataset = self.config.DREAM5_DATASETS[dataset_name]
        
        print(f"\n Loading {dataset_name} dataset...")
        print(f"  Expression: {dataset['expr']}")
        print(f"  Gold standard: {dataset['gold']}")
        
        expr_df = pd.read_csv(dataset['expr'], index_col=0)
        print(f"  Expression shape: {expr_df.shape}")
        
        gold_df = pd.read_csv(dataset['gold'])
        print(f"  Gold shape: {gold_df.shape}")
        print(f"  Columns: {list(gold_df.columns)}")
        
        return expr_df, gold_df
    
    def preprocess(self, expr_df, gold_df):
        """Preprocess data for model input"""
        print("\n Preprocessing...")
        
        node_features = expr_df.T.values.astype(np.float32)
        node_features = np.nan_to_num(node_features, nan=0.0)
        
        gene_names = list(expr_df.columns)
        name_to_idx = {name: i for i, name in enumerate(gene_names)}
        
        print(f"  Genes: {len(gene_names)}")
        print(f"  Samples: {node_features.shape[1]}")
        
        positive_edges = []
        gene_to_edges = defaultdict(list)
        
        for _, row in gold_df.iterrows():
            regulator = str(row['from']).strip()
            target = str(row['to']).strip()
            
            if regulator in name_to_idx and target in name_to_idx:
                u, v = name_to_idx[regulator], name_to_idx[target]
                positive_edges.append([u, v])
                gene_to_edges[u].append(v)
        
        positive_edges = np.array(positive_edges)
        
        print(f"\n  Mapping results:")
        print(f"    Positive edges: {len(positive_edges)}")
        print(f"    Genes with edges: {len(gene_to_edges)}")
        
        return node_features, gene_names, positive_edges, gene_to_edges
    
    def create_balanced_dataset(self, positive_edges, n_nodes):
        """Create balanced dataset with negative sampling"""
        print("\n Creating balanced dataset...")
        
        if len(positive_edges) == 0:
            return np.array([]), np.array([])
        
        n_pos = len(positive_edges)
        pos_set = set(map(tuple, positive_edges))
        
        negative_edges = set()
        max_attempts = n_pos * 20
        attempts = 0
        
        while len(negative_edges) < n_pos and attempts < max_attempts:
            u = np.random.randint(0, n_nodes)
            v = np.random.randint(0, n_nodes)
            
            if u != v and (u, v) not in pos_set and (v, u) not in pos_set:
                negative_edges.add((u, v))
            attempts += 1
        
        while len(negative_edges) < n_pos:
            u = np.random.randint(0, n_nodes)
            v = np.random.randint(0, n_nodes)
            if u != v and (u, v) not in pos_set:
                negative_edges.add((u, v))
        
        neg_array = np.array(list(negative_edges))
        all_edges = np.vstack([positive_edges, neg_array])
        labels = np.hstack([np.ones(n_pos), np.zeros(len(negative_edges))])
        
        print(f"    Positive: {n_pos}, Negative: {len(negative_edges)}")
        print(f"    Total: {len(all_edges)}")
        
        return labels, all_edges

# ============================================================================
# MODELS
# ============================================================================
class MLPModel(nn.Module):
    """Multi-Layer Perceptron baseline"""
    
    def __init__(self, in_dim, hidden_dim=128, latent_dim=64, dropout=0.3):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features).squeeze()

class MultiHeadGNN(nn.Module):
    """Multi-head Graph Neural Network"""
    
    def __init__(self, in_dim, hidden_dim=128, latent_dim=64, dropout=0.3):
        super().__init__()
        
        self.gcn_conv1 = GCNConv(in_dim, hidden_dim)
        self.gcn_conv2 = GCNConv(hidden_dim, latent_dim)
        
        self.gat_conv1 = GATConv(in_dim, hidden_dim // 4, heads=4, concat=True)
        self.gat_conv2 = GATConv(hidden_dim, latent_dim // 4, heads=4, concat=True)
        
        self.sage_conv1 = SAGEConv(in_dim, hidden_dim)
        self.sage_conv2 = SAGEConv(hidden_dim, latent_dim)
        
        self.combine = nn.Linear(latent_dim * 3, latent_dim)
        
        self.edge_predictor = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_pairs):
        z_gcn = self.gcn_conv1(x, edge_index).relu()
        z_gcn = self.gcn_conv2(z_gcn, edge_index).relu()
        
        z_gat = self.gat_conv1(x, edge_index).relu()
        z_gat = self.gat_conv2(z_gat, edge_index).relu()
        
        z_sage = self.sage_conv1(x, edge_index).relu()
        z_sage = self.sage_conv2(z_sage, edge_index).relu()
        
        z_combined = torch.cat([z_gcn, z_gat, z_sage], dim=1)
        z = self.combine(z_combined)
        z = self.dropout(z)
        
        src = z[edge_pairs[0]]
        dst = z[edge_pairs[1]]
        edge_features = torch.cat([src, dst], dim=1)
        
        return self.edge_predictor(edge_features).squeeze()

# ============================================================================
# CROSS-VALIDATOR WITH PREDICTION STORAGE (FIXED)
# ============================================================================
class CrossValidator:
    """Robust cross-validation with prediction storage"""
    
    def __init__(self, model_class, config, model_type='mlp'):
        self.model_class = model_class
        self.config = config
        self.model_type = model_type
        self.results = []
        self.all_predictions = None  # Will be numpy array
        self.all_edge_pairs = None   # Will be numpy array
        self.all_true_labels = None  # Will be numpy array
        self.fold_predictions = []   # Temporary storage
        self.fold_edge_pairs = []    # Temporary storage
        self.fold_true_labels = []   # Temporary storage
        
    def validate(self, X, y, edge_index=None, node_features=None, edge_pairs_all=None):
        """Perform cross-validation with comprehensive storage"""
        kfold = StratifiedKFold(
            n_splits=self.config.N_FOLDS, 
            shuffle=True, 
            random_state=self.config.RANDOM_SEED
        )
        
        # Clear temporary storage
        self.fold_predictions = []
        self.fold_edge_pairs = []
        self.fold_true_labels = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            print(f"\n{'='*50}")
            print(f"Fold {fold + 1}/{self.config.N_FOLDS}")
            print('='*50)
            
            # Prepare data
            if self.model_type == 'gnn' and node_features is not None:
                X_train = node_features
                X_val = node_features
            else:
                X_train = torch.FloatTensor(X[train_idx])
                X_val = torch.FloatTensor(X[val_idx])
            
            y_train = torch.FloatTensor(y[train_idx])
            y_val = torch.FloatTensor(y[val_idx])
            
            # Get edge pairs for this fold
            if self.model_type == 'gnn' and edge_index is not None:
                train_pairs = edge_index[:, train_idx]
                val_pairs = edge_index[:, val_idx]
                
                if edge_pairs_all is not None:
                    fold_edges = edge_pairs_all[val_idx]
                else:
                    if torch.is_tensor(val_pairs):
                        fold_edges = val_pairs.T.cpu().numpy()
                    else:
                        fold_edges = val_pairs.T
            else:
                train_pairs = None
                val_pairs = None
                fold_edges = edge_pairs_all[val_idx] if edge_pairs_all is not None else None
            
            # Initialize model
            if self.model_type == 'gnn':
                model = self.model_class(
                    in_dim=node_features.shape[1],
                    hidden_dim=self.config.HIDDEN_DIM,
                    latent_dim=self.config.LATENT_DIM,
                    dropout=self.config.DROPOUT
                )
            else:
                model = self.model_class(
                    in_dim=X.shape[1],
                    hidden_dim=self.config.HIDDEN_DIM,
                    latent_dim=self.config.LATENT_DIM,
                    dropout=self.config.DROPOUT
                )
            
            # Train fold and get predictions
            history, val_preds, val_true = self._train_fold(
                model, X_train, y_train, X_val, y_val, 
                train_pairs, val_pairs
            )
            
            # Store predictions and edge pairs for this fold
            self.fold_predictions.append(val_preds)
            self.fold_true_labels.append(val_true)
            if fold_edges is not None:
                self.fold_edge_pairs.append(fold_edges)
            
            # Evaluate
            metrics = self._compute_metrics(val_true, val_preds)
            metrics['fold'] = fold
            metrics['history'] = history
            
            self.results.append(metrics)
            
            print(f"\nFold {fold + 1} Results:")
            for metric, value in metrics.items():
                if metric not in ['fold', 'history']:
                    print(f"  {metric}: {value:.4f}")
        
        # FIX: Concatenate all predictions into single numpy arrays
        if self.fold_predictions:
            self.all_predictions = np.concatenate(self.fold_predictions)
            self.all_true_labels = np.concatenate(self.fold_true_labels)
            print(f"\n Total predictions stored: {len(self.all_predictions)}")
        
        if self.fold_edge_pairs:
            try:
                self.all_edge_pairs = np.vstack(self.fold_edge_pairs)
                print(f"Total edge pairs stored: {len(self.all_edge_pairs)}")
            except:
                print(f"Could not stack edge pairs, storing as list")
                self.all_edge_pairs = self.fold_edge_pairs
        
        return self._summarize_results()
    
    def _train_fold(self, model, X_train, y_train, X_val, y_val, train_pairs, val_pairs):
        """Train a single fold with early stopping"""
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5
        )
        criterion = nn.BCELoss()
        
        best_val_aupr = 0
        patience_counter = 0
        history = {'train_loss': [], 'val_auc': [], 'val_aupr': []}
        best_val_preds = None
        best_val_true = None
        
        for epoch in range(self.config.EPOCHS):
            # Training
            model.train()
            optimizer.zero_grad()
            
            if self.model_type == 'gnn' and train_pairs is not None:
                preds = model(X_train, train_pairs, train_pairs)
            else:
                preds = model(X_train)
            
            loss = criterion(preds, y_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                if self.model_type == 'gnn' and val_pairs is not None:
                    val_preds = model(X_val, val_pairs, val_pairs).cpu().numpy()
                else:
                    val_preds = model(X_val).cpu().numpy()
                
                val_true = y_val.cpu().numpy()
                val_auc = roc_auc_score(val_true, val_preds)
                val_aupr = average_precision_score(val_true, val_preds)
            
            scheduler.step(val_aupr)
            
            history['train_loss'].append(loss.item())
            history['val_auc'].append(val_auc)
            history['val_aupr'].append(val_aupr)
            
            if val_aupr > best_val_aupr:
                best_val_aupr = val_aupr
                best_val_preds = val_preds
                best_val_true = val_true
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}, AUC={val_auc:.4f}, AUPR={val_aupr:.4f}")
        
        return history, best_val_preds, best_val_true
    
    def _compute_metrics(self, y_true, y_pred):
        return {
            'auc': roc_auc_score(y_true, y_pred),
            'aupr': average_precision_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred > 0.5),
            'accuracy': accuracy_score(y_true, y_pred > 0.5)
        }
    
    def _summarize_results(self):
        summary = {}
        metrics = ['auc', 'aupr', 'f1', 'accuracy']
        
        for metric in metrics:
            values = [r[metric] for r in self.results]
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)
            summary[f'{metric}_min'] = np.min(values)
            summary[f'{metric}_max'] = np.max(values)
        
        return summary, self.results

# ============================================================================
# MAIN PIPELINE
# ============================================================================
class DREAM5Pipeline:
    """Complete pipeline for DREAM5 benchmark with prediction storage"""
    
    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader(config)
        
        for dir_path in [config.OUTPUT_DIR, config.FIGURES_DIR, config.MODELS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
        
    def run(self, dataset_name):
        """Run complete pipeline including centrality analysis"""
        print(f"\n{'='*70}")
        print(f" PROCESSING DREAM5 DATASET: {dataset_name.upper()}")
        print('='*70)
        
        # 1. Load and preprocess data
        expr_df, gold_df = self.data_loader.load_dataset(dataset_name)
        node_features, gene_names, positive_edges, gene_to_edges = self.data_loader.preprocess(
            expr_df, gold_df
        )
        
        if len(positive_edges) == 0:
            print(f"\n No positive edges found. Skipping...")
            return None
        
        # 2. Create balanced dataset
        y_full, edge_pairs = self.data_loader.create_balanced_dataset(
            positive_edges, node_features.shape[0]
        )
        
        # 3. Feature engineering
        print("\n Engineering features...")
        feature_extractor = RegulatoryFeatureExtractor(node_features, gene_names)
        X_full = feature_extractor.compute_features_for_pairs(edge_pairs)
        print(f"  Feature matrix: {X_full.shape}")
        
        # 4. Cross-validation for MLP
        print("\n" + "="*50)
        print("MLP Model Training")
        print("="*50)
        mlp_validator = CrossValidator(MLPModel, self.config, model_type='mlp')
        mlp_summary, mlp_folds = mlp_validator.validate(X_full, y_full)
        
        # 5. Cross-validation for GNN (with prediction storage)
        print("\n" + "="*50)
        print(" Multi-Head GNN Training")
        print("="*50)
        
        edge_index = torch.tensor(
            np.concatenate([positive_edges, positive_edges[:, ::-1]], axis=0).T,
            dtype=torch.long
        )
        
        gnn_validator = CrossValidator(MultiHeadGNN, self.config, model_type='gnn')
        gnn_summary, gnn_folds = gnn_validator.validate(
            X_full, y_full, 
            edge_index=edge_index,
            node_features=torch.FloatTensor(node_features),
            edge_pairs_all=edge_pairs
        )
        
        # 6. Statistical comparison
        print("\n" + "="*50)
        print(" Statistical Analysis")
        print("="*50)
        significance = self._statistical_tests(mlp_folds, gnn_folds)
        
        # 7. Compile results - FIXED: Safe conversion to lists
        results = {
            'dataset': dataset_name,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'mlp': {
                'summary': mlp_summary, 
                'folds': mlp_folds,
                'all_predictions': mlp_validator.all_predictions.tolist() if mlp_validator.all_predictions is not None else [],
                'all_true_labels': mlp_validator.all_true_labels.tolist() if mlp_validator.all_true_labels is not None else []
            },
            'gnn': {
                'summary': gnn_summary, 
                'folds': gnn_folds,
                'all_predictions': gnn_validator.all_predictions.tolist() if gnn_validator.all_predictions is not None else [],
                'all_edge_pairs': gnn_validator.all_edge_pairs.tolist() if gnn_validator.all_edge_pairs is not None else [],
                'all_true_labels': gnn_validator.all_true_labels.tolist() if gnn_validator.all_true_labels is not None else []
            },
            'significance': significance,
            'metadata': {
                'n_genes': node_features.shape[0],
                'n_samples': node_features.shape[1],
                'n_positive_edges': len(positive_edges),
                'gene_names': gene_names,
                'feature_names': feature_extractor.feature_names
            }
        }
        
        # 8. Save results
        self._save_results(results, dataset_name)
        
        # 9. Print summary
        self._print_summary(results)
        
        return results
    
    def _statistical_tests(self, mlp_folds, gnn_folds):
        significance = {}
        
        if len(gnn_folds) == 0:
            return significance
        
        mlp_aucs = [f['auc'] for f in mlp_folds]
        gnn_aucs = [f['auc'] for f in gnn_folds]
        
        try:
            statistic, p_value = wilcoxon(mlp_aucs, gnn_aucs)
            significance['wilcoxon_pvalue'] = p_value
        except:
            significance['wilcoxon_pvalue'] = 1.0
        
        mean_diff = np.mean(gnn_aucs) - np.mean(mlp_aucs)
        pooled_std = np.sqrt((np.std(mlp_aucs)**2 + np.std(gnn_aucs)**2) / 2)
        significance['effect_size'] = mean_diff / pooled_std if pooled_std > 0 else 0
        
        significance['improvement_auc'] = (np.mean(gnn_aucs) - np.mean(mlp_aucs)) / np.mean(mlp_aucs) * 100 if np.mean(mlp_aucs) > 0 else 0
        
        print(f"\n  Wilcoxon p-value: {significance['wilcoxon_pvalue']:.4f}")
        print(f"  Effect size: {significance['effect_size']:.2f}")
        print(f"  Improvement: {significance['improvement_auc']:.1f}%")
        
        return significance
    
    def _save_results(self, results, dataset_name):
        """Save comprehensive results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full results as pickle
        filename = f"{dataset_name}_results_{timestamp}.pkl"
        filepath = os.path.join(self.config.OUTPUT_DIR, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\n Full results saved to: {filepath}")
        
        # Save summary as CSV
        summary = {
            'Metric': ['AUC', 'AUPR', 'F1', 'Accuracy'],
            'MLP_Mean': [
                results['mlp']['summary']['auc_mean'],
                results['mlp']['summary']['aupr_mean'],
                results['mlp']['summary']['f1_mean'],
                results['mlp']['summary']['accuracy_mean']
            ],
            'MLP_Std': [
                results['mlp']['summary']['auc_std'],
                results['mlp']['summary']['aupr_std'],
                results['mlp']['summary']['f1_std'],
                results['mlp']['summary']['accuracy_std']
            ],
            'GNN_Mean': [
                results['gnn']['summary']['auc_mean'],
                results['gnn']['summary']['aupr_mean'],
                results['gnn']['summary']['f1_mean'],
                results['gnn']['summary']['accuracy_mean']
            ],
            'GNN_Std': [
                results['gnn']['summary']['auc_std'],
                results['gnn']['summary']['aupr_std'],
                results['gnn']['summary']['f1_std'],
                results['gnn']['summary']['accuracy_std']
            ]
        }
        
        df = pd.DataFrame(summary)
        csv_path = os.path.join(self.config.OUTPUT_DIR, f'{dataset_name}_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"Summary saved to: {csv_path}")
        
        # Print prediction stats
        if results['gnn']['all_predictions']:
            n_pred = len(results['gnn']['all_predictions'])
            n_edges = len(results['gnn']['all_edge_pairs']) if results['gnn']['all_edge_pairs'] else 0
            print(f"Predictions stored: {n_pred}")
            print(f"Edge pairs stored: {n_edges}")
    
    def _print_summary(self, results):
        print(f"\n{'='*70}")
        print(f"FINAL RESULTS - {results['dataset'].upper()}")
        print('='*70)
        
        print("\n MLP Model:")
        for metric in ['auc', 'aupr', 'f1', 'accuracy']:
            mean = results['mlp']['summary'][f'{metric}_mean']
            std = results['mlp']['summary'][f'{metric}_std']
            print(f"  {metric.upper()}: {mean:.4f} ± {std:.4f}")
        
        print("\n GNN Model:")
        for metric in ['auc', 'aupr', 'f1', 'accuracy']:
            mean = results['gnn']['summary'][f'{metric}_mean']
            std = results['gnn']['summary'][f'{metric}_std']
            print(f"  {metric.upper()}: {mean:.4f} ± {std:.4f}")
        
        if results['significance']:
            print("\n Statistical Significance:")
            print(f"  Wilcoxon p-value: {results['significance']['wilcoxon_pvalue']:.4f}")
            print(f"  Effect size: {results['significance']['effect_size']:.2f}")
            print(f"  Improvement: {results['significance']['improvement_auc']:.1f}%")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print(" DREAM5 GENE REGULATORY NETWORK INFERENCE PIPELINE")
    print("="*80)
    
    config = Config()
    pipeline = DREAM5Pipeline(config)
    
    all_results = {}
    for dataset_name in config.DREAM5_DATASETS.keys():
        try:
            results = pipeline.run(dataset_name)
            if results is not None:
                all_results[dataset_name] = results
        except Exception as e:
            print(f"\n Error processing {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if all_results:
        combined_file = os.path.join(config.OUTPUT_DIR, 'all_datasets_results.pkl')
        with open(combined_file, 'wb') as f:
            pickle.dump(all_results, f)
        
        print(f"\n{'='*80}")
        print(" ALL DATASETS COMPLETED SUCCESSFULLY!")
        print(f" Results saved to: {config.OUTPUT_DIR}/")
        print('='*80)
        
        # Print final summary across datasets
        print("\n CROSS-DATASET SUMMARY:")
        print("-" * 60)
        for name, res in all_results.items():
            gnn_auc = res['gnn']['summary']['auc_mean']
            gnn_aupr = res['gnn']['summary']['aupr_mean']
            impr = res['significance']['improvement_auc']
            print(f"{name:12s}: GNN AUC={gnn_auc:.3f}, AUPR={gnn_aupr:.3f}, Improvement={impr:.1f}%")
    else:
        print("\n No datasets were successfully processed.")
