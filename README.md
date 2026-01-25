# PADNet: Progressive-Difference-Aware Feature Reconstruction Mechanism for Anomaly Detection

[![IEEE Transactions on Multimedia](https://img.shields.io/badge/Journal-IEEE%20TMM-blue.svg)](https://ieeexplore.ieee.org/document/11175433)

This repository is the official implementation of the paper **"PADNet: Progressive-Difference-Aware Feature Reconstruction Mechanism for Anomaly Detection"**, published in *IEEE Transactions on Multimedia (TMM), 2025*.

## 📖 Introduction

Unsupervised anomaly detection often faces a trade-off: feature reconstruction-based methods are robust to noise but typically struggle with **detailed information loss** and **insufficient anomaly discriminability**.

To bridge this gap, we propose **PADNet**, a Progressive-Difference-Aware Feature Reconstruction Network. PADNet introduces a harmonic symmetric reconstruction framework designed to preserve fine-grained details while significantly enhancing the model's ability to distinguish anomalies from normal patterns.

Experimental results on **MVTec AD**, **Visa**, and **BTAD** datasets demonstrate that PADNet achieves superior performance while requiring only **25.3%** of the parameters compared to state-of-the-art baselines.

## 💡 Core Innovations

PADNet addresses the limitations of existing reconstruction-based methods through two key technical contributions:

### 1. Progressive Feature Harmonizer (PFH)
**Mitigating Detailed Information Loss**
We introduce a **Harmonic Symmetric Reconstruction Framework** integrated with the **PFH**.
* **Problem Solved:** Traditional reconstruction often smooths out critical details, leading to "identity mapping" where anomalies are also well-reconstructed, or loss of normal texture details.
* **Mechanism:** The PFH enables the progressive fusion of information flows. By harmonizing features across different levels, it effectively reduces undesired reconstruction errors and preserves the structural integrity of the input data.

### 2. Neighbor-Aided Residual Feature Representation (NRFR)
**Enhancing Anomaly Discriminability**
To make anomalies "stand out" more clearly in the residual map, we propose the **NRFR** module.
* **Problem Solved:** Simple reconstruction errors are sometimes insufficient to distinguish subtle anomalies from normal variations.
* **Mechanism:** This module utilizes a **feature memory pool** to store reference normal samples. It captures discriminative cues by interacting with neighboring reference samples, innovatively strengthening the difference-aware feature representations.

## 📊 Results Highlights

* **SOTA Performance:** Achieved state-of-the-art results across three benchmark datasets: MVTec AD, Visa, and BTAD.
* **High Efficiency:** The model is lightweight, using only **25.3%** of the parameter count of comparable SOTA methods.

## 📝 Citation

If you find this work helpful for your research, please cite our paper:

```bibtex
@ARTICLE{11175433,
  author={Yang, Fan and Jing, Peiguang and Wang, Weiming and Wang, Fu Lee and Su, Yuting},
  journal={IEEE Transactions on Multimedia}, 
  title={PADNet: Progressive-Difference-Aware Feature Reconstruction Mechanism for Anomaly Detection}, 
  year={2025},
  volume={27},
  number={},
  pages={9125-9135},
  keywords={Image reconstruction;Feature extraction;Anomaly detection;Training;Nearest neighbor methods;Harmonic analysis;Vectors;Image restoration;Artificial intelligence;Semantics;Anomaly detection;anomaly localization;feature reconstruction},
  doi={10.1109/TMM.2025.3613127}
}
