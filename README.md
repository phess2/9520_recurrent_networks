Recurrent Networks for Directed Graphical Models (DGM)

Motivation
We study the computational differences between modern sequence models that separate linear aggregation and nonlinearity (transformers, modern state space models) and classical RNNs (LSTM/GRU/Elman) that interleave temporal propagation with nonlinearities. Our goal is to quantify inductive biases and capacity in settings where underlying data dependencies arise from directed graphical models (DGMs) with controllable lag structures.

Project Goals
- Build multivariate DGM generators with discrete and continuous latent states, distinct observation models, and configurable lag sets to interpolate between HMM-like and fully connected temporal dependencies.
- Implement comparable sequence models in JAX using Modula primitives and adapters, including a Transformer adapter, LSTM, Elman RNN, and a Linear Recurrent Unit (LRU) as a state space baseline.
- Provide common training utilities, GPU-friendly data loaders, and logging with Weights & Biases (wandb), including checkpointing and evaluation cadence controls.
- Track accuracy, negative log-likelihood (NLL), and mutual information with true latent states; provide capacity proxies such as parameter counts and receptive field estimates.

Environment
- Python 3.12
- JAX with CUDA 12.x wheels (via pip). Ensure a recent NVIDIA driver is available.
- Modula installed from GitHub.
See environment.yml for full details.

Repo Structure
- src/
  - data/: DGM dataset generator and iterators
  - models/: Base model, Transformer adapter, LSTM/Elman RNN, LRU SSM
  - train/: Training loop, evaluation, checkpointing
  - configs/: Simple dataclass configs for datasets/models/training
  - utils/: Metrics, logging, capacity estimators
- scripts/: CLI helpers to launch training and evaluation
- tests/: Pytest smoke tests for datasets, models, training

References
- Modula library (JAX-based modular deep learning): https://github.com/modula-systems/modula.git

License
MIT