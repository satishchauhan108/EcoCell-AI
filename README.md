 
# EcoCell-AI: AI for 5G Energy Consumption Modelling

An AI-based machine learning project for predicting energy consumption across 5G base stations using telecom traffic, operational data, temporal feature engineering, neural networks, and model explainability.

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Dataset Description](#dataset-description)
- [Technology Stack](#technology-stack)
- [Methodology](#methodology)
- [Feature Engineering](#feature-engineering)
- [Model Development](#model-development)
- [Model Evaluation](#model-evaluation)
- [Ensemble Learning](#ensemble-learning)
- [Model Explainability with SHAP](#model-explainability-with-shap)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [Running the Project](#running-the-project)
- [Results](#results)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Disclaimer](#disclaimer)

## Overview

EcoCell-AI explores the application of machine learning and deep learning to energy consumption modelling in 5G cellular networks.

The project uses historical energy measurements, network load, energy-saving modes, temporal information, and base-station configuration data to predict energy consumption.

The implementation combines data preprocessing, feature engineering, signal processing, tabular neural networks, ensemble learning, and SHAP-based interpretability in an offline machine learning pipeline.

The repository includes a primary modelling notebook, a separate model-explainability notebook, three CSV datasets, and generated prediction files.

## Problem Statement

Energy consumption in cellular networks can vary with traffic load, time, operational states, and base-station configuration.

Understanding these patterns can help researchers and telecom operators investigate energy demand and explore opportunities for more energy-efficient network management.

The objective of EcoCell-AI is to develop a predictive model that estimates base-station energy consumption from available network and operational features.

## Objectives

- Predict energy consumption using historical and operational telecom data.
- Combine energy, cell-level, and base-station configuration datasets.
- Extract useful temporal and historical information through feature engineering.
- Investigate signal-processing techniques for representing network load.
- Train and evaluate tabular neural-network models.
- Combine model predictions using ensemble learning.
- Investigate model predictions using SHAP-based explainability.

## Key Features

- **Multi-source data integration:** Combines energy measurements, cell-level traffic information, and base-station configuration data.
- **Temporal feature engineering:** Extracts hour, day, weekday, and periodic time representations.
- **Historical feature generation:** Creates lagged features for energy, load, and selected energy-saving modes.
- **Signal processing:** Generates smoothed load features and derivatives using Savitzky-Golay and Butterworth filtering techniques.
- **Deep learning:** Implements tabular neural networks using FastAI and Keras.
- **Categorical embeddings:** Learns dense representations for categorical features in the Keras model.
- **Ensemble learning:** Combines predictions from two neural-network implementations.
- **Model evaluation:** Uses mean absolute error (MAE) and mean absolute percentage error (MAPE).
- **Explainable AI:** Includes a separate notebook for SHAP-based model interpretation.

## System Architecture

EcoCell-AI follows an offline machine learning workflow.

```text
             Raw Datasets
                   |
        +----------+----------+
        |          |          |
     ECdata.csv CLdata.csv BSinfo.csv
        |          |          |
        +----------+----------+
                   |
          Data Integration
                   |
          Data Preprocessing
                   |
          Feature Engineering
                   |
       +-----------+-----------+
       |                       |
 Temporal and Lag Features   Signal Processing
       |                       |
       +-----------+-----------+
                   |
          Model Input Features
                   |
       +-----------+-----------+
       |                       |
  FastAI Tabular NN       Keras Tabular NN
       |                       |
       +-----------+-----------+
                   |
          Weighted Ensemble
                   |
          Energy Predictions
                   |
       Evaluation and Analysis
                   |
          SHAP Explainability
```

The diagram represents the logical modelling workflow. SHAP analysis is implemented separately in its own notebook.

## Dataset Description

The repository contains three primary datasets.

### 1. ECdata.csv

Contains energy-consumption observations.

| Column | Description |
|---|---|
| `Time` | Timestamp associated with the observation |
| `BS` | Base-station identifier |
| `Energy` | Target variable for regression |

### 2. CLdata.csv

Contains cell-level network information.

| Column | Description |
|---|---|
| `Time` | Timestamp |
| `BS` | Base-station identifier |
| `CellName` | Cell identifier |
| `load` | Network load |
| `ESMode1`–`ESMode6` | Energy-saving mode features |

### 3. BSinfo.csv

Contains base-station and cell configuration information.

| Column | Description |
|---|---|
| `BS` | Base-station identifier |
| `CellName` | Cell identifier |
| `RUType` | Radio-unit type |
| `Mode` | Operating mode |
| `Frequency` | Frequency information |
| `Bandwidth` | Bandwidth configuration |
| `Antennas` | Antenna configuration |
| `TXpower` | Transmission-power information |

### Dataset Statistics

The uploaded dataset contains approximately:

| Dataset | Rows |
|---|---:|
| `ECdata.csv` | 92,629 |
| `CLdata.csv` | 125,575 |
| `BSinfo.csv` | 1,217 |

The energy dataset contains 923 unique base-station identifiers and 168 unique timestamps.

These counts describe the uploaded CSV files and may differ if the datasets are updated.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading, merging, filtering, and manipulation |
| NumPy | Numerical operations and array processing |
| Scikit-learn | Preprocessing, transformations, and grouped validation |
| FastAI | Tabular neural-network modelling |
| Keras / TensorFlow | Neural-network implementation and categorical embeddings |
| SciPy | Signal processing and filtering |
| SHAP | Model explainability and feature-contribution analysis |
| Jupyter Notebook | Interactive experimentation and analysis |
| Google Colab / Google Drive | Notebook execution and model storage used in the workflow |

The repository does not currently contain a complete FastAPI service, frontend application, or production inference API.

## Methodology

### 1. Data Integration

The project integrates the three datasets using their shared identifiers.

The cell-level dataset is first merged with base-station information using `BS` and `CellName`.

The merged features are then filtered to `Cell0` in the feature-construction pipeline.

Finally, the feature data is joined with energy observations using `BS` and `Time`.

This associates the target energy value with the corresponding network and configuration features.

### 2. Data Preprocessing

The preprocessing workflow prepares structured data for neural-network training.

The implementation includes:

- Timestamp conversion and temporal extraction.
- Sorting observations by base station and time.
- Categorical feature handling.
- Continuous-feature normalization.
- Feature selection.
- Preparation of training, validation, and test data.

### 3. Temporal Feature Engineering

The project derives time-related features such as:

- Hour of day.
- Day of month.
- Weekday number.
- Periodic spline representations of the hour.

Periodic encoding represents the cyclic nature of time more naturally than treating the hour as an ordinary numerical variable.

### 4. Lag Feature Generation

Lag features provide historical information to the model.

The implementation generates lagged values for selected variables, including:

- Network load.
- Historical energy consumption.
- Selected energy-saving mode indicators.

Lags from one to three time steps are created within each base station.

Chronological sorting and base-station grouping are important because they prevent a lag feature from accidentally referring to a different base station.

### 5. Signal Processing

The project creates additional representations of network load using signal-processing techniques.

**Savitzky-Golay filtering**

The filter smooths a signal by fitting local polynomials. The implementation uses a window length of 5 and polynomial order 3.

Derived features include smoothed load and first-, second-, and third-order load differences.

**Butterworth filtering**

Butterworth-based filtering is used to generate an additional filtered representation of load.

These transformations provide the model with information about local trends and changes in network utilization.

For forecasting applications, filtering must be validated carefully to ensure future observations do not influence features used for earlier predictions.

## Model Development

EcoCell-AI uses two tabular neural-network implementations.

### 1. FastAI Tabular Neural Network

The FastAI model uses the tabular learning API.

Its documented hidden-layer configuration is:

```python
[256, 512, 1024, 512, 256]
```

The model uses:

- `Categorify` for categorical features.
- `Normalize` for continuous features.
- A single output for energy regression.
- MAE-based optimization.

The model learns nonlinear relationships between the engineered input features and the energy target.

### 2. Keras Neural Network

The Keras implementation explicitly separates categorical and continuous inputs.

Categorical features are represented using embedding layers, with an embedding dimension of 16 in the implementation.

The embedding outputs are flattened and combined with the numerical features.

The combined representation is passed through dense layers:

```text
Categorical Inputs
        |
Embedding Layers
        |
     Flatten
        |
        +---------- Continuous Features
        |                    |
        +---------+----------+
                  |
            Concatenation
                  |
             Dense (256)
                  |
             Dense (512)
                  |
            Dense (1024)
                  |
             Dense (512)
                  |
             Dense (256)
                  |
          Linear Output (1)
                  |
        Predicted Energy
```

The Keras model uses the Adam optimizer with an initial learning rate of 0.002.

Its training configuration includes validation monitoring, EarlyStopping, and ReduceLROnPlateau.

EarlyStopping helps stop training when validation performance no longer improves, while ReduceLROnPlateau reduces the learning rate when progress stalls.

## Model Evaluation

The project evaluates predictions using regression metrics.

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted values.

\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

A lower MAE indicates smaller average absolute prediction errors.

### Mean Absolute Percentage Error (MAPE)

MAPE measures the average absolute error relative to the actual target values.

\[
MAPE = \frac{100}{n}
\sum_{i=1}^{n}
\left|\frac{y_i-\hat{y}_i}{y_i}\right|
\]

MAPE can be difficult to interpret when actual values are zero or very close to zero.

The implementation reports MAPE as a ratio in its notebook outputs; this README expresses it as a percentage for readability.

### Validation Strategy

The workflow uses a chronological train/test split and GroupKFold validation grouped by base station.

These approaches evaluate different forms of generalization:

- Chronological splitting evaluates performance on a later time period.
- GroupKFold evaluates generalization across base-station groups.

A more rigorous forecasting evaluation could combine temporal ordering and group separation in a time-aware validation design.

## Ensemble Learning

The project combines predictions from the FastAI and Keras models using a weighted average.

The implemented equal-weight ensemble is:

\[
\hat{y}_{ensemble}
=
0.5\hat{y}_{Keras}
+
0.5\hat{y}_{FastAI}
\]

For example, if the two models predict 60 and 64 units, the ensemble prediction is 62 units.

The repository also explores a harmonic-mean ensemble.

Ensembling can improve predictive performance when the individual models make different errors. However, the improvement must be assessed using an appropriate validation strategy.

## Model Explainability with SHAP

The repository includes a separate notebook named `shap-analysis-of-models.ipynb`.

SHAP stands for SHapley Additive exPlanations. It uses Shapley-value concepts from cooperative game theory to estimate feature contributions to model predictions.

The analysis helps investigate:

- Which features are important to the model.
- How feature values influence predictions.
- Why individual predictions differ from a baseline.
- Which network and temporal features may be associated with higher or lower predicted energy.

SHAP explains model behavior; it does not establish causal relationships between features and energy consumption.

## Project Structure

```text
EcoCell-AI/
├── AI_5G.ipynb
├── shap-analysis-of-models.ipynb
├── ECdata.csv
├── CLdata.csv
├── BSinfo.csv
├── submission_fastai.csv
├── submission_ensemble.csv
├── submission_ensemble_hm.csv
└── README.md
```

### File Descriptions

| File | Purpose |
|---|---|
| `AI_5G.ipynb` | Main data preparation, feature engineering, model training, validation, and prediction workflow |
| `shap-analysis-of-models.ipynb` | Separate model-explainability analysis |
| `ECdata.csv` | Energy-consumption dataset |
| `CLdata.csv` | Cell-level load and energy-saving mode dataset |
| `BSinfo.csv` | Base-station and cell configuration dataset |
| `submission_fastai.csv` | Predictions produced by the FastAI workflow |
| `submission_ensemble.csv` | Predictions from the weighted ensemble |
| `submission_ensemble_hm.csv` | Predictions from the harmonic-mean ensemble |
| `README.md` | Project documentation |

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version supported by the selected library versions.
- Jupyter Notebook or Google Colab.
- The three CSV datasets included in the repository.
- Sufficient memory and compute resources for neural-network training.

The original notebook may rely on environment-specific paths and library versions, so adjustments may be necessary when running it outside its original environment.

### 1. Clone the Repository

Replace the URL below with the actual repository URL.

```bash
git clone <YOUR_ECOCELL_AI_REPOSITORY_URL>
cd EcoCell-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

Install the core packages used by the modelling workflow:

```bash
pip install numpy pandas scipy scikit-learn jupyter
pip install fastai tensorflow shap
```

Library compatibility can vary across Python versions. For reproducible results, use versions compatible with the original notebook environment.

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `AI_5G.ipynb` to explore the main modelling pipeline.

After running the training workflow, open `shap-analysis-of-models.ipynb` to examine the explainability analysis.

**Note:** The repository does not include a complete, pinned `requirements.txt`, so the installation commands above provide a starting point rather than a guarantee of exact reproduction.

## Running the Project

1. Place the three CSV files in the expected data directory.
2. Open `AI_5G.ipynb`.
3. Update any environment-specific file paths.
4. Execute the data-loading and preprocessing cells.
5. Run the feature-engineering pipeline.
6. Train and evaluate the FastAI and Keras models.
7. Generate out-of-fold predictions and ensemble outputs.
8. Review the generated submission CSV files.
9. Run `shap-analysis-of-models.ipynb` for model interpretation.

The workflow is notebook-based. A separate inference service is not included in the repository.

## Results

The uploaded notebook reports the following out-of-fold evaluation results.

| Model / Approach | OOF MAE | OOF MAPE |
|---|---:|---:|
| Keras neural network | ~1.48 | ~0.062 |
| Weighted ensemble | ~1.044 | ~0.0426 |
| Harmonic-mean ensemble | ~1.058 | ~0.0425 |

MAPE values above are shown as ratios in the original evaluation and are approximately 6.2%, 4.26%, and 4.25%, respectively.

The weighted ensemble achieves the lowest MAE among these reported results, while the harmonic-mean ensemble has a marginally lower MAPE.

These are notebook-reported out-of-fold results. They should not be interpreted as independently verified production performance or as proof of generalization to every future timestamp and unseen base station.

## Limitations

- **Offline pipeline:** The project is primarily implemented in notebooks rather than as a production application.
- **No inference API:** A complete FastAPI service is not present in the repository.
- **Validation design:** Grouped validation and chronological splitting test different generalization scenarios and could be combined more rigorously.
- **Preprocessing leakage risk:** The Keras preprocessing implementation fits some transformations using concatenated training and test features. A stronger approach is to fit transformations on training data only.
- **Forecasting leakage risk:** Filtering and historical feature construction should be audited to ensure that future observations cannot influence past predictions.
- **Cell coverage:** The feature-construction pipeline filters the cell information to `Cell0`.
- **Baseline comparisons:** Additional comparisons against LightGBM, XGBoost, and CatBoost would help establish whether neural networks are the best approach.
- **Reproducibility:** A pinned dependency file and standardized execution configuration would make the results easier to reproduce.
- **Operational validation:** The current metrics do not establish live network performance or actual energy savings.

## Future Improvements

### Short-Term Improvements

- Add a pinned `requirements.txt`.
- Document dataset provenance and column definitions.
- Improve preprocessing validation and leakage checks.
- Compare the models using consistent validation folds.
- Add repeatable evaluation scripts and result summaries.

### Medium-Term Improvements

- Benchmark against XGBoost, LightGBM, and CatBoost.
- Optimize ensemble weights using out-of-fold predictions.
- Evaluate generalization to unseen base stations and future timestamps.
- Compare alternative forecasting models.
- Improve SHAP-based visualizations and reporting.

### Production-Level Improvements

- Create a reusable preprocessing and inference pipeline.
- Expose model inference through a validated API.
- Containerize the inference service.
- Add structured logging, monitoring, and model-version management.
- Monitor changes in input data and prediction quality.
- Evaluate predicted energy against real operational measurements before integrating with network-management systems.

These are proposed improvements, not features currently implemented in the repository.

## Disclaimer

EcoCell-AI is an experimental machine-learning project focused on 5G base-station energy-consumption modelling.

Predictions and evaluation results depend on the available datasets, preprocessing, validation design, and model configuration. The current implementation does not directly control base stations, switch energy-saving modes, or guarantee real-world energy savings.

Use the reported metrics as experimental model-evaluation results rather than as a guarantee of operational performance.


