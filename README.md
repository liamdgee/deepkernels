![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/backend-PyTorch-orange.svg)

# **deepkernels** — A Unified Probabilistic Modelling Framework for Multidimensional Anomaly Detection
---

> **Architectural Insight:** This document serves as a technical overview of the production architecture of the `deepkernels` framework. As this is the public-access playground repository, this README is intended to outline the mathematical methodology and system design, rather than serve as a step-by-step training or setup guide. 
>
> **Production Note:** The containerised environment (`docker-compose.yml`) is provided for developer-level 'quick inference' testing only. Model weights are frozen, training is complete, and there are no resources available here to spin up mock model training environments. 

To see the finalised model in action, view the interactive [causal inference dashboard](https://topologicaldisparity.com), which showcases real-time multitask predictions for US loan approval rates across adjustable borrower demographic sliders.

You can also see a technical demo of a currently in-development stochastic optimal control tool for ML developers in the form of a [bot-human anomaly detection dashboard](https://kernelsforbotdetection.com).

*deepkernels* is an end-to-end probabilistic inference engine which leverages **Multitask State-Space Gaussian Processes** with dynamic **Kronecker Task Covariance (LMC)** structures provided by **Dynamic Neural Kernel Networks** and unsupervised nonparametric clustering driven by **Hierarchical Dirichlet Processes**. The **ShallowKernels** class in `src/deepkernels/model.py` is simplified proxy-logic for the proprietary **DeepKernels Bayesian Inference Engine**  which is optimised for large-scale inference via **PyKeOps CUDA-JIT compilation**.

### 🌌 Project Status: PRODUCTION

| Metric | Status / Value |
| :--- | :--- |
| **Speedrun** | `40 epochs / 195 seconds` |
| **GPU Utilisation** | `92-99%` |
| **Validation Root Mean Squared Error (RMSE)** | `0.2889` |
| **Final Marginal Log Likelihood (MLL)** | `1.0615` |
---

### 🧠 Latent Kernel Health During Training ($K_{zz}$)
At $t=0$, the inducer covariance matrix was initialized with the following spectral properties:

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Max Eigenval** | `0.078` | Global Variance Cap |
| **Min Eigenval** | `0.0052` | Numerical Stability Floor |
| **Num NaNs** | `0` | NaN values in kernel matrix |
| **Diag Average** | `0.0243` | Prior Signal Power |
| **Condition #** | `[~15~]` | Matrix Well-Conditioning |

---

### 📟 Hardware During Training / Inference

| Component | Training | Inference |
| :--- | :--- | :--- |
| **GPU** | 🟢 1/2 NVIDIA A100-80GB (40GB slice)| 🟢 1/20 NVIDIA A100-80GB (4GB Slice) |
| **CUDA** | 🟢 12.1 | 🟢 12.4 |
| **batch_dim/seq_len** | 🟢 512/32 | 🟢 6/1 (Generative Model Capability) |

---

## 🚀 System Architecture

* **Encoder:** Deep Convolutional Neural Network for projection of data into latent space. *(Logic: `src/deepkernels/models/vae.py`)*
* **Clustering:** Bayesian nonparametric layer for global and local categorical clustering amortised through a variational autoencoder. *(Logic: `PROPRIETARY`)*
* **Decoder:** Latent convolutional decoder operates in a bottlenecked dimension of projected random fourier features and regularises the clustering module against custom loss terms -- acts as the decoding bridge between latent space and data space.
*(Logic: `src/deepkernels/models/vae.py`)*
* **Kernel Network:** A neural hypernet child of the dirichlet clustering module—random fourier features are classified into base kernel primitives. *(Logic: `PROPRIETARY`)*
* **KeOps Symbolic Tensor Operations Kernel:** The input for the GP `covar_module`. Designed to maximise computational efficiency when calculating complex, learnable kernel combinatorics. `PyKeOps` allows the CUDA GPU to compile complex JIT covariance kernel structure in C++. *(Logic: `PRORIETARY`)*
* **LMC Gaussian Process:** Highly customised multitask gaussian process with custom probabilistic `mean_module`, custom GPU-accelerated `covar_module` and `dynamic variational strategy` that is fed simplex probabilities across tasks each forward pass. *(Logic: `src/deepkernels/model/gp.py`)*
* **Inference Engine:** CUDA-optimised KeOps cache for linear $O(n)$ or quasi-linear $O(n \log n)$ scaling.

---

## 🧪 Multi-stage Curriculum Learning
The framework utilizes a distinct 6-stage ensemble reinforcement learning class with warmups and Cyclical Refinement Loops utilizing 3 distinct optimizers.

* **Stage 1:** VAE reconstructs latent space with dirichlet process gradients frozen and all KL penalties suspended.
* **Stage 2:** VAE continues to optimise recon loss with hierarchical dirichlet process module gradients flowing.
* **Stage 3:** GP warmup with only upstream bayesian gradients active.
* **Stage 4:** GP trains with gradient flow to Neural Kernel Network -> KeOps JIT-CUDA Kernel.
* **Stage 5:** Cyclical E-step M-step Maximum Likelihood Estimation.
* **Stage 6:** Full training with all gradients unfrozen.

---

## 🧬 Mathematical Methodology

The core objective of the `deepkernels` framework is to jointly optimize a deep generative projection (VAE) and a probabilistic non-parametric surrogate (Multitask GP) by maximizing the Evidence Lower Bound (ELBO). 

Because the model jointly estimates hierarchical latent structures and multi-target continuous functions, the global objective function is decomposed into a data-fit (likelihood) component and a highly regularised 6-part Kullback-Leibler (KL) divergence penalty.

The total objective to be maximized is formulated as:

$$\mathcal{L}_{ELBO} = \mathcal{L}_{Likelihood} - \sum_{i=1}^{6} \mathcal{D}_{KL}^{(i)}$$

### 1. The Likelihood Component (Data Fit / Uncertainty)
The likelihood term bridges the unsupervised dimensionality reduction with the supervised predictive task. It consists of two expectations:

$$\mathcal{L}_{Likelihood} = \mathbb{E}_{q_{\phi}(Z|X)}[\log p_{\theta}(X|Z)] + \mathbb{E}_{q(F|Z)}[\log p(Y|F)]$$

* **VAE Reconstruction Loss:** ![Equation](https://latex.codecogs.com/svg.latex?\mathbb{E}_{q_{\phi}(Z|X)}[\log&space;p_{\theta}(X|Z)]) 
ensures the latent projection retains the topological structure of the high-dimensional time-variant input $X$.

* **Multitask GP Gaussian Likelihood:** $\mathbb{E}_{q(F|Z)}[\log p(Y|F)]$ evaluates the probability of the target variables $Y$ given the latent GP function $F$. We utilize a Rank-1 Intrinsic Coregionalization Model (ICM) to handle multi-output covariance, computed via our CUDA-optimised KeOps engine to maintain $O(n)$ or $O(n \log n)$ scaling.

### 2. The 6-Term KL Divergence Penalty
To prevent posterior collapse and enforce the Bayesian priors across both the Neural Kernel Network and the hierarchical Dirichlet process, we evaluate six distinct KL divergence terms. During the multi-stage curriculum, these gradients are cyclically frozen and unfrozen to stabilize the loss landscape.

$$\sum_{i=1}^{6} \mathcal{D}_{KL}^{(i)} = \mathcal{D}_{Global} + \mathcal{D}_{Local} + \mathcal{D}_{\alpha} + \mathcal{D}_{\ell} + \mathcal{D}_{Recon} + \mathcal{D}_{\mathcal{IW}}$$

* **Global KL:** $\mathcal{D}_{KL}(q(\pi) \parallel p(\pi))$ regularises the global stick-breaking process.
* **Local KL:** $\mathcal{D}_{KL}(q(c) \parallel p(c))$ penalises the deviation of local latent clusters.
* **Alpha KL:** $\mathcal{D}_{KL}(q(\alpha) \parallel p(\alpha))$ enforces the prior on the Dirichlet concentration parameter, controlling cluster 'atom' sparsity.
* **Lengthscale KL:** $\mathcal{D}_{KL}(q(\ell) \parallel p(\ell))$ acts on the intermediate GP kernel hyperparameters, preventing the Neural Kernel Network from overfitting the covariance surface. The GP never directly sees this covariance structure, but dictates kernel nonstationarity across heterogenous clusters.
* **Recon (Latent) KL:** ![Equation](https://latex.codecogs.com/svg.latex?\mathcal{D}_{KL}(q_{\phi}(Z|X)&space;\parallel&space;p(Z))) regularises the VAE encoder's variational distribution against a low-rank multivariate normal prior.
* **Inverse Wishart KL:** ![Equation](https://latex.codecogs.com/svg.latex?\mathcal{D}_{KL}(q(\Sigma_{T})&space;\parallel&space;\mathcal{IW}(\Psi,&space;\nu))) enforces the Inverse Wishart prior on the Multitask GP's task-covariance matrix (Often referred to as the B matrix in Linear Model of Coregionalisation Theory). This acts as a regulariser on positive-semi-definiteness in the output covariance matrix and dynamic mixing weights.

### 3. Stage 5: Cyclical E-step M-step Maximum Likelihood Estimation

Jointly optimizing the VAE parameters ($\phi$) and the GP/Dirichlet hyperparameters ($\theta$) from a random initialization often results in degenerate local optima. For instance, the VAE might collapse the latent space $Z$ to trivially satisfy the GP lengthscale penalty, or the GP variance might explode to absorb a poor VAE reconstruction.

To resolve this, Stage 5 implements a cyclical, alternating optimization scheme analogous to Stochastic Expectation-Maximization (EM). We partition our 3 optimizers to strictly route gradients to mutually exclusive parameter groups, iteratively holding one fixed while updating the other.

#### The E-Step (Expectation / Latent Refinement)
During the E-step phase of the cycle, the Multitask GP hyperparameters and Dirichlet priors ($\theta$) are strictly frozen. The objective reduces to finding the optimal latent projection $Z = q_{\phi}(X)$ that maximizes the expected log-likelihood under the current structural assumptions of the Neural Kernel Network.

The active optimizer (Optimizer 1) updates the VAE parameters $\phi$ according to:

$$\phi^{(t+1)} = \arg\max_{\phi} \left( \mathbb{E}_{q_{\phi}(Z|X)}[\log p_{\theta^{(t)}}(X|Z)] - \mathcal{D}_{Recon} + \mathbb{E}_{q(F|Z)}[\log p(Y|F)] \right)$$

*Methodology:* By freezing the GP, the VAE is forced to adjust its latent topological mapping so that the continuous target $Y$ can be smoothly interpolated by the existing KeOps covariance matrix.

#### The M-Step (Maximization / GP Hyperparameter Update)
In the alternating M-step, the VAE encoder/decoder weights ($\phi$) are frozen, effectively locking the latent coordinates $Z$. The active optimisers (Optimisers 2 and 3) now update the Multitask GP, the inducing points, and the hierarchical Dirichlet/Inverse Wishart modules ($\theta$).

The update rule focuses on refining the kernel surface and adapting the 5 prior penalties to the new latent positions:

$$\theta^{(t+1)} = \arg\max_{\theta} \left( \mathbb{E}_{q_{\phi^{(t+1)}}(Z|X)}[\log p_{\theta}(Y|Z)] - \sum_{i \in \{G, L, \alpha, \ell, \mathcal{IW}\}} \mathcal{D}_{KL}^{(i)} \right)$$

*Methodology:* With the latent inputs $Z$ held constant, the Neural Kernel Network is optimized to maximize the marginal likelihood of the targets $Y$. The KeOps JIT-compiler evaluates the dense matrix inversions highly efficiently during this step, adjusting the task-covariances and lengthscales without disrupting the learned dimensionality reduction.

---


## 📄 License

This project is licensed under the APACHE-2.0 License - see the LICENSE file for details.

## 📧 Contact
Liam Douglas Giles - [liamdgiles@outlook.com](mailto:liamdgiles@outlook.com) / [https://linkedin.com/in/liamdouglasgiles](https://linkedin.com/in/liamdouglasgiles)