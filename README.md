# Predictive Maintenance Platform

An end-to-end **Data Science, Data Engineering, and MLOps** project for predicting equipment failure risk from industrial telemetry and vibration data.

The project combines:

* **NASA IMS bearing run-to-failure experiments**
* Vibration signal processing and feature engineering
* LightGBM predictive modelling
* Run-held-out model evaluation
* Alert-threshold optimization
* PySpark data processing
* MLflow experiment tracking
* FastAPI model serving
* Streamlit monitoring
* Dockerized deployment

The objective is not only to build an accurate machine-learning model, but to demonstrate how a predictive-maintenance solution can progress from **raw equipment data → condition monitoring → failure-risk prediction → actionable maintenance alerts**.

---

## 1. Business Problem

Unexpected equipment failures can lead to:

* Unplanned downtime
* Production losses
* Higher maintenance costs
* Reduced equipment availability
* Quality and throughput issues

Traditional preventive maintenance relies heavily on fixed maintenance intervals.

Predictive maintenance instead asks:

> **Can equipment condition data be used to identify developing failures before the equipment actually fails?**

This project builds a machine-learning pipeline that converts equipment telemetry and vibration signals into a **failure-risk score** that can help maintenance teams prioritize inspections.

```text
Machine / Sensor Data
        │
        ▼
Data Processing
        │
        ▼
Feature Engineering
        │
        ▼
Machine Learning
        │
        ▼
Failure Risk
        │
        ▼
Maintenance Alert
```

---

# 2. NASA IMS Bearing Experiment

The main public-data experiment uses the **NASA IMS Bearings dataset**, which contains vibration measurements collected from physical bearing run-to-failure experiments.

Rather than randomly mixing measurements from the same degradation trajectory between training and testing, the experiment uses separate runs for model development and evaluation.

```text
NASA IMS vibration recordings
            │
            ▼
    Signal Processing
            │
            ▼
 Time + Frequency Features
            │
            ▼
         LightGBM
            │
            ▼
     Failure Risk Score
            │
            ▼
    Maintenance Alert
```

### Experimental Split

The three experimental runs are separated as:

| Dataset | Purpose                          |
| ------- | -------------------------------- |
| Set 1   | Model training                   |
| Set 2   | Validation / threshold selection |
| Set 3   | Final held-out evaluation        |

This design reduces leakage between measurements belonging to the same degradation trajectory and provides a stronger test of model generalization.

---

# 3. Vibration Feature Engineering

Each vibration recording contains high-frequency waveform measurements sampled at approximately **20 kHz**.

Instead of directly providing the raw waveform to LightGBM, each recording is converted into condition-monitoring indicators.

### Time-Domain Features

Features include:

* RMS
* Standard deviation
* Peak amplitude
* Peak-to-peak amplitude
* Skewness
* Kurtosis
* Crest factor
* Impulse factor

For example, RMS represents the overall vibration magnitude:

$$
RMS = \sqrt{\frac{1}{N}\sum_{i=1}^{N}x_i^2}
$$

Increasing vibration magnitude or increasingly impulsive signals can indicate deterioration in bearing condition.
### Frequency-Domain Features

FFT-based features are also extracted:

* Spectral centroid
* Dominant frequency
* Low-frequency band power
* Mid-frequency band power
* High-frequency band power

These features allow the model to detect changes not only in vibration magnitude but also in the **frequency distribution of vibration energy**.

### Feature Interpretation

The engineered features capture complementary characteristics of bearing vibration:

| Feature                 | What It Measures                                           | Relevance to Bearing Condition                                                           |
| ----------------------- | ---------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **RMS**                 | Overall vibration magnitude                                | Increasing vibration levels can indicate developing mechanical degradation               |
| **Kurtosis**            | Impulsiveness of the vibration signal                      | Localized bearing defects can produce sharp impacts, increasing kurtosis                 |
| **Crest Factor**        | Peak amplitude relative to RMS                             | Highlights unusually large vibration peaks relative to the overall vibration level       |
| **Peak-to-Peak**        | Difference between maximum and minimum vibration amplitude | Captures the overall amplitude range of the vibration signal                             |
| **Spectral Centroid**   | Frequency-weighted center of the vibration spectrum        | Indicates shifts in where vibration energy is concentrated                               |
| **Dominant Frequency**  | Frequency with the strongest spectral component            | Captures changes in the strongest vibration frequency                                    |
| **Spectral Band Power** | Signal energy within selected frequency ranges             | Captures how vibration energy is distributed across low, mid, and high frequency regions |

Together, the features allow the model to capture not only **how strongly the bearing is vibrating**, but also **how the shape and frequency characteristics of the vibration change as the bearing approaches failure**.

The resulting feature pipeline is:

```text
20 kHz Vibration Waveform
            │
            ├── Time Domain
            │     ├── RMS
            │     ├── Kurtosis
            │     ├── Crest Factor
            │     └── Peak-to-Peak
            │
            └── Frequency Domain
                  ├── Spectral Centroid
                  ├── Dominant Frequency
                  └── Spectral Band Power
                            │
                            ▼
                         LightGBM
                            │
                            ▼
                      Failure Risk
```

To maintain a common model schema across the different IMS experimental configurations, feature values are aggregated across available vibration axes.

---

# 4. Failure Labelling

NASA documents the failed bearings at the end of each experimental run.

The experiment therefore treats the documented run end as the observed failure point for the failed bearing.

Other bearings are retained as **right-censored controls**.

Importantly, information that could directly reveal proximity to failure is excluded from model training.

The following are retained only as metadata:

```text
hours_to_run_end
bearing_id
run_id
```

This prevents the model from simply learning how much time remains before the experiment ends.

The model must instead identify failure-related patterns from the vibration features themselves.

---

# 5. Model

The main model is **LightGBM**, a gradient-boosted decision-tree algorithm.

LightGBM is well suited to the engineered condition-monitoring features because it can model nonlinear interactions between variables such as:

```text
RMS
+
Kurtosis
+
Crest Factor
+
Spectral Energy
+
Dominant Frequency
        │
        ▼
Failure Risk
```

The output is a continuous risk score rather than only a binary prediction.

For example:

| Observation | Failure Risk | Action               |
| ----------- | -----------: | -------------------- |
| Bearing A   |         0.01 | Continue monitoring  |
| Bearing B   |         0.03 | Continue monitoring  |
| Bearing C   |         0.08 | Raise alert          |
| Bearing D   |         0.72 | High-risk inspection |

---

# 6. Alert Threshold Selection

Using a default classification threshold of **0.50** is not necessarily appropriate for predictive maintenance.

Equipment failures are rare, and the cost of missing a developing failure can be substantially greater than the cost of an unnecessary inspection.

The alert threshold is therefore selected using the **validation dataset only**.

```text
Training Set
     │
     ▼
Train LightGBM

Validation Set
     │
     ▼
Threshold Sweep
     │
     ▼
Select Operating Threshold
     │
     ▼
Held-Out Test Set
```

The selected threshold for the current experiment is:

```text
0.05
```

The held-out test set is evaluated only after threshold selection.

---

# 7. NASA IMS Results

The final model is evaluated on the held-out **Set 3** run.

| Metric             |     Score |
| ------------------ | --------: |
| ROC-AUC            | **0.969** |
| PR-AUC             | **0.793** |
| Precision          | **0.704** |
| Recall             | **0.690** |
| F1 Score           | **0.697** |
| Selected Threshold |  **0.05** |

### Confusion Matrix

```text
                       Predicted
                   Healthy   Failure

Actual Healthy      25,109       42
Actual Failure          45      100
```

The failure class is highly imbalanced, making **PR-AUC, precision, and recall** more informative than accuracy alone.

### Interpretation

**Recall ≈ 69%**

Approximately 69% of positive failure-window observations were successfully identified.

**Precision ≈ 70%**

Approximately 70% of positive alerts corresponded to the target failure condition.

**ROC-AUC ≈ 0.97**

The model demonstrates strong ability to rank failure-related observations above healthy/control observations across classification thresholds.

**PR-AUC ≈ 0.79**

The model maintains useful precision-recall performance despite the strong class imbalance.

---

# 8. Early Warning Performance

Classification metrics describe how well the model distinguishes failure-related observations from healthy controls, but predictive maintenance also depends on **how early a developing failure can be detected**.

For the documented failed bearing in the held-out Set 3 run, an alert is triggered whenever the model's predicted failure-risk score exceeds the **validation-selected threshold of 0.05**.

The first alert occurred approximately **21.8 hours before the documented end of the run**:

```text id="8uapx9"
First Alert
Risk ≥ 0.05
    │
    │
    │  ~21.8 hours warning
    │
    ▼
Documented Run End
 / Bearing Failure
```

This means the model identified a failure-related vibration pattern almost **22 hours before the documented failure point**, providing potential lead time for inspection or maintenance intervention.

The model also continued to identify the developing failure across the failure window, alerting on approximately **69% of the positive failure-window observations**.

### False Alerts

Early detection is only useful if it does not generate excessive alerts on healthy equipment.

Across the healthy-control observations in the held-out run:

| Metric                           |         Result |
| -------------------------------- | -------------: |
| First warning before run end     | **21.8 hours** |
| Failure-window alert coverage    |      **69.0%** |
| Healthy-control false alerts     |         **42** |
| Healthy-control false-alert rate |      **0.22%** |

The combination of **early warning and low healthy-control false-alert rate** demonstrates how the model output can be translated from a classification score into an operational maintenance signal.

However, the 21.8-hour warning time comes from a single documented failed bearing in the held-out IMS run and should therefore be interpreted as a **case-specific demonstration rather than a general expected warning time for industrial equipment**.


# 9. End-to-End Architecture

The broader project extends beyond the NASA modelling experiment.

```text
Machine / Sensor Telemetry
           │
           ▼
      Raw / Bronze
           │
           ▼
   Data Processing
           │
           ▼
  Feature Engineering
           │
           ▼
      LightGBM
       │       │
       ▼       ▼
    MLflow   Batch Scoring
               │
               ▼
          Risk Ranking
           │       │
           ▼       ▼
       FastAPI  Streamlit
```

An optional **PySpark** pipeline demonstrates how the data-processing layer could be scaled for larger telemetry datasets.

A production implementation could evolve toward:

```text
Machines / PLC / Sensors
          │
          ▼
Kafka / Azure Event Hubs
          │
          ▼
Spark Structured Streaming
          │
          ▼
       Delta Lake
   Bronze → Silver → Gold
          │
          ▼
    Feature Pipeline
          │
          ▼
    Model Serving
          │
          ▼
Maintenance Dashboard
```

---

# 10. Technology Stack

### Data Science

* Python
* Pandas
* NumPy
* SciPy
* scikit-learn
* LightGBM
* FFT / vibration signal processing

### Data Engineering

* PySpark
* Parquet
* Scalable transformation pipelines

### MLOps / Deployment

* MLflow
* FastAPI
* Streamlit
* Docker
* pytest
* GitHub Actions

---

# 11. Repository Structure

```text
predictive-maintenance-project/
│
├── api/
│   └── app.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── artifacts/
│   ├── ims_test_metrics.json
│   ├── ims_threshold_sweep.csv
│   └── ims_feature_importance.csv
│
├── models/
│
├── src/
│   ├── features.py
│   ├── train.py
│   ├── batch_score.py
│   ├── spark_pipeline.py
│   ├── ims_download.py
│   ├── ims_features.py
│   └── ims_train.py
│
├── tests/
│
├── config.yaml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# 12. Running the NASA IMS Experiment

### Create environment

```bash
python -m venv .venv
```

Activate the environment.

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Download and prepare NASA IMS data

```bash
python -m src.ims_download
```

The raw dataset is excluded from Git because the compressed archive is large and expands to several gigabytes.

### Extract vibration features

```bash
python -m src.ims_features
```

For a quick preprocessing check:

```bash
python -m src.ims_features \
    --max-files 20 \
    --output data/processed/ims_features_smoke.parquet
```

### Train and evaluate

```bash
python -m src.ims_train
```

The pipeline:

1. Trains LightGBM using Set 1.
2. Sweeps alert thresholds using Set 2.
3. Selects the operating threshold.
4. Evaluates the final configuration once on Set 3.
5. Saves metrics, feature importance, and threshold-analysis artifacts.

---

# 13. Original Synthetic Telemetry Experiment

The repository also contains a reproducible synthetic industrial telemetry pipeline.

The generated data includes:

* Temperature
* Vibration
* Pressure
* RPM
* Power consumption
* Machine load
* Machine age
* Hours since maintenance
* Equipment failures

Rolling features are generated over multiple time windows before predicting whether equipment will fail within the following **24 hours**.

The synthetic experiment primarily demonstrates the **end-to-end engineering and MLOps architecture**, while the NASA IMS experiment provides a public run-to-failure vibration benchmark.

---

# 14. Limitations

The NASA IMS dataset represents controlled run-to-failure bearing experiments.

Therefore, the results should **not** be interpreted as evidence that the same performance would be achieved on a production manufacturing line.

Real deployment would introduce additional challenges including:

* Different equipment types
* Variable operating conditions
* Sensor drift
* Maintenance interventions
* Different failure modes
* Changes in load and production recipes
* Limited labelled failures
* Data quality issues

The NASA experiment should therefore be interpreted as a **run-to-failure predictive-maintenance benchmark**.

---

# 15. Potential Improvements

Future extensions could include:

* SHAP-based failure explanations
* Remaining Useful Life (RUL) regression
* Temporal deep-learning models
* CNN modelling directly on vibration spectra
* Equipment-specific threshold calibration
* Cost-sensitive threshold optimization
* Drift monitoring
* Kafka / Azure Event Hubs ingestion
* Delta Lake Bronze/Silver/Gold architecture
* Airflow or Prefect orchestration
* Model registry and champion/challenger deployment
* Automated retraining pipelines

---

# 16. Key Takeaway

This project demonstrates the complete transition from:

```text
Raw Machine Signals
        ↓
Signal Processing
        ↓
Feature Engineering
        ↓
Predictive Modelling
        ↓
Model Validation
        ↓
Failure Risk
        ↓
Maintenance Decision
```

The NASA IMS experiment achieved a **0.969 ROC-AUC and 0.793 PR-AUC** on a held-out experimental run, with approximately **70% precision, 69% recall, and 21.8 hours of warning before the documented run end**.

More importantly, the project demonstrates how machine-learning outputs can be integrated into an end-to-end **data engineering, MLOps, and operational decision-support architecture**.

---

## Disclaimer

This repository is a portfolio and research project. The NASA IMS experiment is a controlled run-to-failure benchmark and should not be used for real equipment-maintenance decisions without validation on representative operational data.
