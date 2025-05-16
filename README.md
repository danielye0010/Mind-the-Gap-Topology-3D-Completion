# Topology-Aware Point Cloud Completion

This repository contains the implementation of **Topo-PCN**, a topology-augmented point cloud completion framework designed to handle structural corruptions such as holes. The project also includes the implementation of a custom benchmark, **ModelNet-Topology**, which extends ModelNet with topological corruptions and annotations.

##  Project Overview

- **Baseline Model:** Point Completion Network (PCN) (Yuan et al, 3DV 2018)
- **Proposed Model:** Topo-PCN (PCN + topology features + topology loss)
- **Comparison Model:** Point-Attention (Transformer-based) (Wang et al. AAAI 2024)
- **Benchmark:** Modified ModelNet with topological corruptions
- **Input:** Incomplete point clouds with associated 3D topology vectors
- **Output:** Dense completed point clouds

## Model Details
Topo-PCN extends PCN by injecting a 3D topology vector (from persistent homology) into the latent feature, and adding a topology-aware loss based on bottleneck distance. These changes improve robustness to structural defects like holes.

**Architecture:**
- Encoder: PointNet-style shared MLP + global max pooling → 1024-d latent code
- Topology: 3D topo vector projected and concatenated to latent
- Decoder:
  - Coarse: MLP → 1024 points
  - Fine: Folding decoder → 4096 points
- Chamfer loss (coarse + fine) + ramped topology loss
- AdamW, cosine learning rate scheduling, gradient clipping, and mixed precision (AMP).

## Acknowledgement
This project is part of COM S 6720: Advanced Topics in Artificial Intelligence at Iowa State University. Developed by:
* Daniel Ye
* Shakiba Khourashahi
* Ilia Jahanshahi



