# Topology-Aware Point Cloud Completion

This repository contains the implementation of **Topo-PCN**, a topology-augmented point cloud completion framework designed to handle structural corruptions such as holes. The project also includes the implementation of a custom benchmark, **ModelNet-Topology**, which extends ModelNet with topological corruptions and annotations.

## 🔧 Project Overview

- **Baseline Model:** Point Completion Network (PCN)
- **Proposed Model:** Topo-PCN (PCN + topology features + topology loss)
- **Comparison Model:** Point-Attention (Transformer-based)
- **Benchmark:** Modified ModelNet with topological corruptions
- **Input:** Incomplete point clouds with associated 3D topology vectors
- **Output:** Dense completed point clouds

## 🔧 Results
![image](https://github.com/user-attachments/assets/12b6f7d1-b727-4f50-bafa-7e0c83b5202b)
![image](https://github.com/user-attachments/assets/0e104033-429f-4277-8e2c-7333b10b0054)



