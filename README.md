
# DREAM5 Gene Regulatory Network Inference with Multi-Head Graph Neural Networks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.0+-green.svg)](https://pytorch-geometric.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-DREAM5-brightgreen)](https://dream5.bio.nyu.edu/)

<div align="center">
  <img src="figures/architecture_overview.png" alt="MH-GNN Architecture" width="800"/>
  <p><em>Multi-Head GNN architecture combining GCN, GAT, and GraphSAGE for GRN inference</em></p>
</div>

##  Overview

This repository implements a ** multi-head graph neural network** for inferring gene regulatory networks (GRNs) from expression data, evaluated on the **DREAM5 benchmark datasets**. Our approach combines three complementary graph convolution operations to achieve near-perfect prediction (AUC > 0.98) across all datasets.

###  Key Features

- **Multi-Head GNN Architecture**: Parallel GCN, GAT (4-head attention), and GraphSAGE layers
- **Comprehensive Evaluation**: 5-fold cross-validation with AUC, AUPR, F1, and accuracy metrics
- **Statistical Analysis**: Wilcoxon signed-rank tests and effect size calculations
- **Centrality Analysis**: Identifies master regulators, hubs, and bottlenecks
- **Publication-Ready**: Generates all figures and tables for immediate use
- **Reproducible**: Fixed random seeds (42) and complete documentation

### 📊 Performance Highlights

| Dataset | MLP AUC | GNN AUC | Improvement |
|---------|---------|---------|-------------|
| in_silico | 0.814 | **0.982** | +20.6% |
| e_coli | 0.639 | **0.981** | +53.5% |
| s_cerevisiae | 0.591 | **0.997** | +68.7% |
| **AVERAGE** | 0.681 | **0.986** | **+47.6%** |

### Prerequisites

# Python 3.8 or higher
python --version

# CUDA (optional, for GPU acceleration)
nvidia-smi

```bash
# **1. Clone**
git clone https://github.com/Akilaramu5698/GRN_Prediction_MULTIHEAD_GNN.git
cd dream5-grn-gnn

# **2. Install**
pip install -r requirements.txt

#** 3. Set your_data_path/ (Update your data path in main.py)**
├── in_silico_expression_data.csv
├── in_silico_gold_standard.csv
├── e_coli_expression_data.csv
├── e_coli_gold_standard.csv
├── s_cerevisiae_expression_data.csv
├── s_cerevisiae_gold_standard.csv
├── e_coli_gene_ids.csv          # Optional: Gene name mapping
├── in_silico_gene_ids.csv       # Optional: Gene name mapping
└── s_cerevisiae_gene_ids.csv    # Optional: Gene name mapping

class Config:
    BASE_PATH = "/path/to/your/data" 
# **4. Run pipeline**
python main_pipeline.py          # Train models
python centrality_analysis.py    # Analyze results

**Output structure**
dream5_results/     # Model predictions & metrics
centrality_analysis/ # Top genes & high-confidence edges
figures/            # 6 publication-ready figures per dataset
