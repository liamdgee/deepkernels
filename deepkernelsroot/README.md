deepkernels

An End-to-End Modular Framework to provide nonparametric dirichlet process clustering and Sparse Gaussian Process Equivalent Uncertainty Quantification in strictly linear time -- O(N) from raw input data.

deepkernels is a production-grade machine learning library that establishes a framework for efficient and scalable nonparametric Bayes. By combining a pretrained Vision Transformer (ViT_b_16) as the backbone feature extractor, feedihg into a dirichlet clustering module that outputs amortised logits in weight space, we are able to produce a mean-field variational approximation through a linear decoder for posterior sampling. Sampling is mathematically robust given the decoder operates and optimises in Reproducing Kernel Hilbert Space, allowing for a quasi-variational posterior approximation. Given the lightweight architecture, we achieve strictly linear computational efficiency, allowing for a scalable probabilistic model that reliably predicts uncertainty as opposed to a typical neural net MAP estimate. 


🚀 Key Features
Linear Scalability: Bipasses traditional bottleneck of Gaussian Processes by Fourier Projection in weight space

Modular Architecture: Dedicated modules for data preprocessing, models, kernels, losses, inference, optimisers, training, utils and orchestration.

Deep Kernel Learning: Integrates Vision Transformers and pytorch-based nbeural networks with GPyTorch Layers rooted in classical and Bayesian statistical methods.

MLOps Ready: Fully containerised docker buid for reproducibility

Data Harmonisation: Library endogenous tools for cleaning, harmonising, feature selection and inducing point selection for numerical data

📂 Directory Structure

The project follows a strict src layout to ensure robust packaging and import isolation.

deepkernels/
├── config.yaml #-global hyperparameters-#
├── Dockerfile #-multistage-#
├── pyproject.toml #-dependency management-#
├── requirements.txt
├── .dockerignore
├── .gitignore
├── src/
│   └── deepkernels/       #-main-#
│       ├── infer/         #-quick inference-#
│       ├── models/        #-core architecture-#
│       ├── kernels/       #-Custom kernels-#
│       ├── preprocess/    #-clean data, harmonise and preprocess-#
│       ├── losses/        #-custom loss functions
│       ├── train/         #-customer trainers e.g. stochastic gradient langevin dyanmics
│       ├── optimisers/    #-custom monte carlo optimisers-#
│       └── utils/         #-useful model tools-#
└── tests/                 #-SWE and ML standard tests

🛠️ Installation
We recommend installing the package in editable mode. This allows you to modify the source code in src/ without reinstalling.

# Clone the repository
git clone https://github.com/yourusername/deepkernels.git
cd deepkernels

# Install dependencies (requires Python 3.10+)
pip install -e ".[dev]"

🛠️ Production (Docker)
Build an optimised container using a multistage build.

# Build the image
docker build -t deepkernels:latest .

# Run the inference pipeline
docker run --rm -v $(pwd)/data:/app/data deepkernels:latest

⚡ Usage

Quick Inference -- use default configuration outline in config.yaml

# Run the module directly
python -m deepkernels.infer.predict --input data/sample.csv


Custom Training Loops -- build custom workflows for model optimisation

from deepkernels.models import DeepKernelGP
from deepkernels.preprocess import Harmoniser
from deepkernels.kernels import LinearKernelON

# 1. Harmonise Data
clean_data = DataCleaner().fit(X)
harmonise_data = SchemaHarmoniser.fit(X)
select_features = FeatureTransformer.fit(X)
prune_features = LassoFeatures.fit(X)
inducing_points = InducingPointSelect.fit(X)
process_for_torch = TorchPreprocessor(X)

# 2. Initialize O(N) Model
model = Model()

# 3. Train
LangevinTrainer(model)


🧪 Testing
Pytest for unit testing and logic validation

# Run all tests
pytest tests/

# Run specific ML logic checks (e.g., convergence)
python -m pytest tests/ml_logic/test_dirichlet_convergence.py

🔧 Configuration
global settingas are managed in config.yaml

model:
  backbone: "vit"
  dropout: 0.1

gp:
  kernel: "custom linear operator (fourier approx)"
  gp: GPyTorch ExactGP
  approximation: Harmonic (RFF) Projection
  n_fourier_features: 2048

🤝 Contributing
Fork the repository.

Create a feature branch (git checkout -b feature/new-kernel).

Commit your changes.

Push to the branch.

Open a Pull Request.


📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

📧 Contact
Liam Douglas Giles - [liamdgiles@outlook.com / https://linkedin.com/in/liamdouglasgiles]


