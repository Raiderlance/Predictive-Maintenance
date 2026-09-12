# Predictive Maintenance Platform

An end-to-end **Data Science + Data Engineering + MLOps** portfolio project for predictive maintenance using synthetic industrial telemetry and physical NASA IMS bearing vibration data.

The project goes beyond a modelling notebook. It demonstrates the complete workflow from machine data to maintenance decision support:

```text
Machine / Sensor Data
        ↓
Data Ingestion
        ↓
Data Preparation
        ↓
Feature Engineering
        ↓
Machine Learning
        ↓
Model Selection & Validation
        ↓
Operational Evaluation
        ↓
Explainability
        ↓
Model Serving
        ↓
Operational Dashboard
        ↓
Maintenance Decision
```

The repository contains two complementary workflows:

1. **Manufacturing Deployment Pipeline** — synthetic industrial telemetry demonstrates point-in-time feature engineering, temporal validation, LightGBM, hyperparameter tuning, operational alert metrics, MLflow, PySpark, batch scoring, FastAPI, Streamlit, SHAP, Docker, and automated testing.

2. **NASA IMS Bearing Experiment** — physical run-to-failure vibration measurements are transformed into time-domain, frequency-domain, and cross-bearing relative features for failure-window detection and fault-localization analysis.

### Synthetic Fleet Dashboard

![Synthetic predictive-maintenance dashboard showing fleet KPIs and machine risk ranking](docs/images/synthetic-dashboard.png)

*The main dashboard uses the reproducible synthetic telemetry pipeline and ranks current machine risk for maintenance review.*

---

# 1. Business Problem

Unexpected equipment downtime can reduce throughput, increase maintenance costs, affect product quality, and disrupt production schedules.

Traditional preventive maintenance often relies on fixed maintenance intervals. However, equipment does not necessarily degrade at the same rate.

Predictive maintenance instead uses equipment-condition data to help answer:

> **Which machines are showing signs associated with elevated failure risk and should be inspected first?**

The primary modelling objective in this project is to identify equipment conditions associated with failure within the:

## **Next 24 Hours**

Example operational output:

| Machine | Risk Score | Risk Band | Recommended Action        |
| ------- | ---------: | --------- | ------------------------- |
| M017    |       0.82 | HIGH      | Inspect immediately       |
| M031    |       0.46 | MEDIUM    | Schedule diagnostic check |
| M008    |       0.09 | LOW       | Continue monitoring       |

The LightGBM output is treated as a **model-estimated failure-risk score** rather than assumed to be a perfectly calibrated real-world probability of failure.

---

# 2. Project Overview

```text
                    PREDICTIVE MAINTENANCE PLATFORM

          ┌────────────────────┴────────────────────┐
          │                                         │
          ▼                                         ▼

SYNTHETIC MANUFACTURING                    NASA IMS BEARING
       PIPELINE                                EXPERIMENT
          │                                         │
          ▼                                         ▼
Industrial Telemetry                    Physical Vibration Data
          │                                         │
          ▼                                         ▼
Data Preparation                       Signal Processing
          │                                         │
Rolling Features                       Time-Domain Features
          │                             Frequency Features
          │                             Cross-Bearing Features
          │                                         │
          ▼                                         ▼
      LightGBM                                  LightGBM
          │                                         │
Hyperparameter Search                    Run-Separated Evaluation
          │                                         │
Temporal Validation                      SHAP + Fault Analysis
          │                                         │
Operational Metrics                              Dashboard
          │
          ▼
MLflow + SHAP
          │
          ▼
Batch / API / Dashboard
```

The **synthetic pipeline** demonstrates how predictive-maintenance modelling can be operationalized.

The **NASA IMS experiment** demonstrates signal processing and predictive-maintenance modelling using physical vibration measurements from a controlled bearing run-to-failure experiment.

---

# 3. End-to-End Manufacturing Architecture

```text
Factory Equipment
temperature / vibration / pressure / RPM / power / load
                         │
                         ▼
              PLC / Control System
                         │
                         ▼
           SCADA / Historian / IoT Gateway
                         │
                         ▼
                Telemetry Ingestion
             CSV locally / streaming in production
                         │
                         ▼
                  Bronze Data Layer
              Immutable raw observations
                         │
                         ▼
              Validation + Silver Layer
        Schema / timestamps / ordering / deduplication
                         │
                         ▼
                 Feature Engineering
          Rolling statistics + condition features
                         │
                         ▼
              Temporal Train / Val / Test
                         │
                         ▼
                    LightGBM
                         │
                Model Risk Score
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     Batch Scoring     FastAPI        MLflow
     Risk Ranking    Model Serving   Experiments
          │              │
          └───────┬──────┘
                  ▼
          Streamlit Dashboard
       Risk + Trends + SHAP
                  │
                  ▼
          Maintenance Decision
       Inspect / Diagnose / Monitor
                  │
                  ▼
           Outcome Feedback
     Fault / Repair / Downtime / Cost
```

The local implementation uses CSV and Parquet files so the project can run without proprietary manufacturing infrastructure.

In a production environment, the same component boundaries could connect to:

* PLCs
* SCADA systems
* industrial historians
* IoT gateways
* Kafka / Azure Event Hubs
* Delta Lake
* MES
* CMMS
* maintenance-planning systems

The project demonstrates understanding of how shop-floor sensor data can progress from equipment acquisition through data preparation, analytics, modelling, and operational decision support.

---

# 4. Manufacturing Data Pipeline

## 4.1 Equipment Telemetry

Each observation represents one machine at one event time.

The synthetic telemetry includes:

* temperature
* vibration
* pressure
* RPM
* power consumption
* machine load
* machine age
* hours since maintenance
* observed equipment failure

`src/generate_data.py` creates reproducible development data.

In a production environment:

```text
Physical Sensor
      ↓
PLC
      ↓
SCADA / Historian / IoT Gateway
      ↓
Message Broker / Data Platform
      ↓
Bronze Raw Data
```

Raw observations would first be persisted in an append-only **Bronze layer**.

Keeping the original events supports:

* late-arriving data handling
* data-quality investigation
* auditability
* feature recomputation
* historical replay

---

# 5. Data Preparation and Silver Processing

Before modelling, raw telemetry must be converted into a reliable analytical dataset.

The pipeline performs:

```text
Raw Sensor Data
       ↓
Schema Validation
       ↓
Timestamp Parsing
       ↓
Machine / Time Ordering
       ↓
Duplicate Removal
       ↓
Missing / Invalid Data Handling
       ↓
Time-Window Aggregation
       ↓
Silver Dataset
```

The optional PySpark implementation demonstrates scalable processing of machine telemetry.

Run:

```bash
spark-submit src/spark_pipeline.py
```

The job performs:

* timestamp parsing
* machine/time ordering
* duplicate removal
* time-window aggregation
* Silver-layer Parquet output

A production implementation could evolve toward:

```text
Kafka / Azure Event Hubs
          ↓
Spark Structured Streaming
          ↓
Delta Lake
Bronze → Silver → Gold
          ↓
Feature Store
          ↓
Model Serving
```

This separation between raw, cleaned, and model-ready data makes the pipeline easier to audit, reproduce, and scale.

---

# 6. Synthetic Feature Engineering

For each sensor, the pipeline creates:

* 6-hour rolling mean
* 24-hour rolling mean
* 24-hour rolling standard deviation
* one-period change

Additional contextual features include:

* machine age
* hours since maintenance
* hour of day
* day of week

Every feature at time **t** uses information available at or before **t**.

The target is:

```text
failure_within_horizon = 1
```

when failure occurs within the following 24 hours.

Conceptually:

```text
Historical Sensor Data
          ↓
Point-in-Time Feature Engineering
          ↓
Current Equipment Condition
          ↓
Failure Within Next 24 Hours?
```

Point-in-time feature construction is important because using information unavailable at prediction time would create leakage and unrealistic model performance.

---

# 7. Temporal Validation

Predictive-maintenance data is time-dependent.

A random train/test split can allow future machine behaviour to influence model development.

The synthetic experiment therefore uses:

```text
Past ------------------------------------------→ Future

|            Train            | Validation | Test |
```

A **24-hour purge** is applied before relevant partition boundaries.

This prevents observations whose labels look 24 hours into the future from referring to failures occurring inside the next partition.

Each partition has a distinct purpose:

```text
TRAIN
  ↓
Fit Candidate Models

VALIDATION
  ↓
Compare Models
  ↓
Select Hyperparameters
  ↓
Select Alert Threshold

TEST
  ↓
Final Evaluation
```

The test period does not participate in hyperparameter or threshold selection.

This more closely approximates future deployment than randomly mixing historical observations.

---

# 8. Model Training

`src/train.py` trains a LightGBM binary classifier.

Training produces:

* versioned model bundle
* ordered feature contract
* selected threshold
* prediction horizon
* evaluation metrics
* confusion matrix
* feature importance
* threshold sweep
* MLflow experiment records
* SHAP explanations

The saved model bundle can subsequently be reused for inference.

Training and inference are therefore separated.

## Offline Training

```text
Historical Data
      ↓
Feature Engineering
      ↓
Train LightGBM
      ↓
Validation
      ↓
Select Threshold
      ↓
Save Approved Model
```

## Batch / Online Inference

```text
New Production Data
      ↓
Same Feature Engineering
      ↓
Load Approved Model
      ↓
Risk Score
      ↓
Apply Threshold
      ↓
Alert / Risk Band
      ↓
Dashboard / API
```

New production data does **not** automatically retrain the model.

Retraining should occur as a separate controlled lifecycle process when sufficient new labelled data becomes available or monitoring indicates model degradation.

---

# 9. Hyperparameter Tuning

The repository includes a reproducible randomized LightGBM hyperparameter search.

Run:

```bash
python -m src.tune_hyperparameters --dataset both --n-iter 12
```

The experiment compares **12 candidate configurations**, including the original baseline.

The selection procedure intentionally separates:

```text
Model Selection
      ↓
Validation PR-AUC

Then

Threshold Selection
      ↓
Validation F1

Then

Final Evaluation
      ↓
Held-Out Test
```

This prevents the test set from being used to select hyperparameters or the operating threshold.

The search varies:

* number of trees
* learning rate
* number of leaves
* maximum tree depth
* minimum child samples
* row subsampling
* feature subsampling
* L1 regularization
* L2 regularization

For the synthetic experiment, the selected candidate used:

```text
n_estimators       = 300
learning_rate      = 0.02
num_leaves         = 15
max_depth          = 5
min_child_samples  = 20
subsample          = 1.0
colsample_bytree   = 0.7
reg_alpha          = 1.0
reg_lambda         = 1.0
```

Complete search results are written to:

```text
artifacts/tuning/
```

---

# 10. Hyperparameter Tuning Results

The fixed baseline and selected tuned candidate produced:

| Synthetic Test Metric           |    Fixed Baseline |  Tuned Winner |
| ------------------------------- | ----------------: | ------------: |
| ROC-AUC                         |             0.799 |     **0.811** |
| PR-AUC                          |             0.108 |     **0.117** |
| Precision                       |             0.090 |     **0.155** |
| Recall                          |         **0.679** |         0.445 |
| F1                              |             0.160 |     **0.230** |
| Validation-selected threshold   |              0.10 |          0.75 |
| Failure-event detection rate    | **80.0% (20/25)** | 48.0% (12/25) |
| Alert episodes per machine-week |              3.34 |      **0.91** |

The tuned model improved ranking performance, precision, and F1 while substantially reducing alert workload.

However, this came with a major operational trade-off:

```text
Fixed Model
80% failure-event detection
3.34 alert episodes / machine-week

              vs.

Tuned Model
48% failure-event detection
0.91 alert episodes / machine-week
```

Therefore:

> **The model with the better statistical F1 score is not automatically the better maintenance model.**

The preferred operating configuration depends on the relative cost of:

* missed equipment failures
* unnecessary inspections
* downtime
* maintenance labour
* technician capacity

For safety-critical or expensive equipment, the higher-recall baseline could be preferable despite generating more alerts.

For lower-criticality equipment with limited maintenance capacity, the lower-alert tuned configuration could be more practical.

This experiment demonstrates why predictive-maintenance model selection should combine **statistical performance with operational impact**.

---

# 11. NASA Hyperparameter Search

The same tuning framework was also applied to the NASA IMS experiment.

For NASA IMS, the original LightGBM configuration remained the strongest of the 12 evaluated candidates.

Validation PR-AUC remained:

## **0.819**

No tuned candidate provided a measured improvement.

The weaker alternatives were therefore not selected simply because a tuning experiment had been performed.

This negative result is intentionally retained.

It demonstrates an important modelling principle:

> **Hyperparameter tuning does not guarantee a better model.**

Feature quality, validation design, and the underlying information in the data can matter more than additional parameter optimization.

NASA Set 3 results remain exploratory for the later cross-bearing model because Set 3 previously influenced localization feature design.

---

# 12. Statistical Evaluation Metrics

Machine failures are relatively rare.

Therefore:

## **Accuracy is not the primary metric.**

The project reports:

| Metric        | Interpretation                                                                          |
| ------------- | --------------------------------------------------------------------------------------- |
| **ROC-AUC**   | Ability to rank positive observations above negative observations across thresholds     |
| **PR-AUC**    | Precision-recall performance, particularly informative for imbalanced failure detection |
| **Recall**    | Proportion of positive pre-failure observations successfully detected                   |
| **Precision** | Proportion of alerts corresponding to positive observations                             |
| **F1**        | Harmonic balance between precision and recall                                           |

PR-AUC is particularly useful because the failure class is uncommon.

However, timestamp-level classification metrics alone do not fully represent the maintenance problem.

---

# 13. Operational Alert Metrics

In equipment monitoring, many predictions are generated for the same machine.

For example:

```text
10:00 → ALERT
11:00 → ALERT
12:00 → ALERT
13:00 → ALERT
```

These should not necessarily be interpreted as four independent maintenance incidents.

The project therefore supplements standard ML metrics with **event- and episode-level operational metrics**.

## Event Detection Rate

Measures the percentage of eligible failure events that receive at least one alert during the preceding 24-hour window.

```text
Failure Event
      ↑
      │ 24-hour prediction window
      │
At least one alert?
      ↓
Detected / Missed
```

This answers:

> **Did the system provide any warning before the actual failure event?**

---

## Lead Time

Measures the time from the first qualifying alert to the observed failure.

```text
First Alert ---------------------- Failure
              Lead Time
```

This answers:

> **How much time did maintenance teams have to respond?**

---

## Alert Episodes

Consecutive alerting timestamps are grouped into one maintenance incident.

```text
10:00 ALERT
11:00 ALERT
12:00 ALERT
13:00 ALERT

        ↓

1 Alert Episode
```

This better approximates maintenance workload than counting every timestamp independently.

---

## False Alert Episodes

An alert episode is considered false when no failure follows any alert within the prediction horizon.

This measures unnecessary maintenance investigations more realistically than timestamp-level false-positive counts.

---

## Alerts per Machine-Week

```text
Total Alert Episodes
────────────────────────
Observed Machine-Weeks
```

This measures the alert workload generated by the system.

It answers:

> **How frequently would maintenance personnel need to respond to alerts?**

---

## Alert Persistence

Measures the duration of consecutive alerts.

The project reports:

* mean alert duration
* maximum alert duration

Persistent alerts may indicate sustained abnormal behaviour, whereas isolated alerts may represent transient events or noise.

---

## Failure-Window Coverage

Measures the proportion of timestamps under alert within each 24-hour pre-failure window.

For example:

```text
24-hour failure window
24 hourly predictions

18 predictions = ALERT

Coverage = 18 / 24
         = 75%
```

This indicates whether the model provides a sustained warning rather than a single isolated threshold crossing.

---

## Boundary Handling

Failure events occurring less than 24 hours after the test partition begins are excluded from event-level rates because their complete prediction window is not observable within that partition.

This avoids unfairly evaluating events for which part of the required historical evaluation window is unavailable.

Consecutive alerts are grouped using the dataset's median sampling interval.

Together, the statistical and operational metrics answer different questions:

```text
STATISTICAL METRICS
        ↓
How well does the model classify/rank observations?

OPERATIONAL METRICS
        ↓
Did we detect the failure?
How early?
How often do we alert?
How much workload is generated?
Are alerts sustained?
```

---

# 14. Synthetic Benchmark

Using the default synthetic generator:

* 50 machines
* 120 days
* 144,000 hourly observations

the fixed baseline produced:

| Metric                          |             Score |
| ------------------------------- | ----------------: |
| ROC-AUC                         |         **0.799** |
| PR-AUC                          |         **0.108** |
| Selected threshold              |          **0.10** |
| Precision                       |         **0.090** |
| Recall                          |         **0.679** |
| F1                              |         **0.160** |
| Failure-event detection rate    | **80.0% (20/25)** |
| Alert episodes per machine-week |          **3.34** |

The class is intentionally imbalanced.

The operating threshold is selected using validation data only and then evaluated on the held-out test period.

The synthetic experiment primarily demonstrates:

* architecture
* data preparation
* temporal validation
* model selection
* operational evaluation
* serving
* explainability
* MLOps

rather than expected real-world predictive performance.

---

# 15. Risk-to-Action Layer

The model generates a **failure-risk score**.

The operating threshold is selected using validation data.

The output contains:

* `failure_risk` — model-estimated risk score
* `alert` — whether risk exceeds the selected threshold
* `risk_band` — LOW, MEDIUM, or HIGH
* SHAP contributions — features pushing the model score higher or lower

For the baseline synthetic benchmark:

```text
risk < 0.10
    → LOW

0.10 ≤ risk < 0.60
    → MEDIUM

risk ≥ 0.60
    → HIGH
```

The HIGH cutoff is a demonstration value.

Production thresholds should consider:

* missed-failure cost
* unnecessary inspection cost
* equipment criticality
* downtime cost
* technician capacity
* spare-parts availability
* production constraints

The hyperparameter experiment demonstrates why threshold selection cannot be considered independently from operational requirements.

---

# 16. SHAP Explainability

Maintenance engineers need more than a risk score.

They also need to understand:

> **Why did the model assign this equipment a high risk score?**

The project uses **TreeSHAP** for global and local explanations.

Generated artifacts include:

```text
artifacts/shap_global_importance.csv
artifacts/shap_local_explanations.csv
artifacts/shap_beeswarm.png
artifacts/shap_bar.png

artifacts/ims_shap_global_importance.csv
artifacts/ims_shap_local_explanations.csv
artifacts/ims_shap_beeswarm.png
artifacts/ims_shap_bar.png
```

## Global SHAP

Global SHAP answers:

> **Which features does the model generally rely on?**

## Local SHAP

Local SHAP explains an individual prediction.

Example:

```text
Machine M017
Risk Score = 0.82
        │
        ├── vibration ↑       → increases score
        ├── temperature ↑     → increases score
        ├── maintenance age   → increases score
        └── pressure          → decreases score
```

The API exposes explanations through:

```text
POST /predict/explain
```

SHAP values are reported in the model's raw-score space.

SHAP explains **model behaviour**, not physical causality.

A positive SHAP contribution indicates that a feature pushed the model toward a higher score. It does not prove that the feature physically caused equipment degradation.

---

# 17. Batch Scoring

`src/batch_score.py` selects the latest feature row for each machine and generates a ranked risk table.

```text
Latest Observation Per Machine
             ↓
          LightGBM
             ↓
         Risk Score
             ↓
      Alert + Risk Band
             ↓
Sort by Highest Risk
             ↓
Maintenance Priority List
```

A production scheduler could run this periodically and publish results to:

* dashboard
* MES
* CMMS
* alerting platform
* maintenance-planning system

---

# 18. FastAPI Model Serving

The trained model can be consumed programmatically through **FastAPI**.

Start the API:

```bash
uvicorn api.app:app --reload
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

Conceptually:

```text
Manufacturing Application
          ↓
       FastAPI
          ↓
Saved LightGBM Model
          ↓
Risk + Alert + Explanation
```

Potential consumers include:

* CMMS
* MES
* maintenance mobile applications
* alerting services
* manufacturing dashboards

The saved feature contract is enforced so malformed or incomplete requests are rejected rather than silently scored.

---

# 19. Manufacturing Dashboard

Launch:

```bash
streamlit run dashboard/app.py
```

The dashboard provides a human decision-support layer.

A reliability engineer or maintenance planner can:

1. Review fleet-level machine counts and active alerts.
2. Identify high-risk assets.
3. Rank machines by risk.
4. Select an individual machine.
5. Inspect recent sensor trends.
6. Review local SHAP explanations.
7. Compare local drivers with global model behaviour.
8. Decide whether to inspect, diagnose, or continue monitoring.

The intended workflow is:

```text
Fleet Overview
      ↓
Risk Ranking
      ↓
Select Machine
      ↓
Telemetry Trends
      ↓
SHAP Explanation
      ↓
Engineer Assessment
      ↓
Maintenance Action
```

The dashboard is **decision support**, not an autonomous shutdown system.

---

# 20. MLflow Experiment Tracking

Training experiments are tracked using **MLflow**.

The pipeline records:

* model parameters
* ROC-AUC
* PR-AUC
* precision
* recall
* F1
* selected threshold
* failure horizon
* feature count
* model artifacts

Start MLflow:

```bash
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

This provides traceability between model configuration, evaluation results, and saved artifacts.

---

# 21. NASA IMS Bearing Experiment

The repository also contains a separate experiment using the **NASA IMS Bearings dataset**, which contains physical vibration measurements from bearing run-to-failure experiments.

### NASA IMS Dashboard

![NASA IMS dashboard showing the Set 3 bearing risk overview](docs/images/ims-dashboard-overview.png)

*The separate IMS dashboard uses waveform features extracted from the physical NASA bearing test rig. Set 3 is labelled as an exploratory re-evaluation because it influenced the later localization feature design.*

The objective is:

> **Can vibration characteristics identify that a bearing is approaching a defined failure window before documented run end?**

The workflow is:

```text
NASA IMS Archive
        ↓
ims_download.py
        ↓
20 kHz Vibration Waveforms
        ↓
ims_features.py
        ↓
Time-Domain Features
        +
Frequency-Domain Features
        +
Cross-Bearing Relative Features
        ↓
Processed Feature Table
        ↓
ims_train.py
        ↓
LightGBM
        ↓
Run-Separated Evaluation
        ↓
SHAP + Fault Analysis
        ↓
Dedicated IMS Dashboard
```

The raw archive is approximately 1 GB and expands to several GB, so raw recordings are excluded from Git.

---

# 22. NASA Experimental Split

Complete experimental runs are separated rather than randomly mixing highly correlated vibration snapshots.

| Dataset            | Role                             | Documented Failed Bearing(s) |
| ------------------ | -------------------------------- | ---------------------------- |
| Set 1 (`1st_test`) | Training                         | Bearings 3 and 4             |
| Set 2 (`2nd_test`) | Validation / threshold selection | Bearing 1                    |
| Set 3 (`4th_test`) | Original held-out test           | **Bearing 3**                |

Conceptually:

```text
Set 1
1st_test
   ↓
Training

Set 2
2nd_test
   ↓
Validation / Threshold Selection

Set 3
4th_test
   ↓
Original Held-Out Evaluation
```

Holding out complete runs provides a stronger evaluation than randomly mixing observations from the same degradation trajectory.

> **Dataset naming:** NASA's third experimental set is stored in a directory named `4th_test`. The project therefore uses `4th_test` internally for Set 3; this does not represent a fourth experimental set.

For the held-out experiment, Bearing 3 is the documented failed bearing.

Bearings 1, 2, and 4 do not have documented failures by experiment end and are therefore treated as **right-censored controls**, rather than assumed to be permanently healthy.

---

# 23. NASA Vibration Feature Engineering

Each vibration waveform is sampled at:

## **20 kHz**

Rather than feeding raw waveforms directly into LightGBM, each recording is transformed into interpretable condition-monitoring features.

```text
20 kHz Vibration Waveform
            │
       ┌────┴────┐
       ▼         ▼
 Time Domain   Frequency Domain
       │         │
       ▼         ▼
 RMS            FFT
 STD            Spectral Centroid
 Peak           Dominant Frequency
 Peak-to-Peak   Spectral Band Power
 Skewness
 Kurtosis
 Crest Factor
 Impulse Factor
       │         │
       └────┬────┘
            ▼
      Feature Vector
```

## Time-Domain Features

| Feature                | What It Measures                         | Relevance                                |
| ---------------------- | ---------------------------------------- | ---------------------------------------- |
| **RMS**                | Overall vibration magnitude              | Detects changes in vibration energy      |
| **Standard Deviation** | Signal variability                       | Measures vibration instability           |
| **Peak**               | Maximum absolute amplitude               | Detects large vibration events           |
| **Peak-to-Peak**       | Maximum minus minimum amplitude          | Captures vibration range                 |
| **Skewness**           | Distribution asymmetry                   | Detects changes in signal distribution   |
| **Kurtosis**           | Signal impulsiveness                     | Sensitive to sharp impact-like behaviour |
| **Crest Factor**       | Peak relative to RMS                     | Highlights unusually large peaks         |
| **Impulse Factor**     | Peak relative to mean absolute amplitude | Measures impulsive vibration severity    |

## Frequency-Domain Features

| Feature                 | What It Measures                              |
| ----------------------- | --------------------------------------------- |
| **Spectral Centroid**   | Frequency-weighted centre of vibration energy |
| **Dominant Frequency**  | Strongest spectral component                  |
| **0–1 kHz Band Power**  | Relative low-frequency energy                 |
| **1–5 kHz Band Power**  | Relative mid-frequency energy                 |
| **5–10 kHz Band Power** | Relative high-frequency energy                |

For each bearing, the pipeline calculates the **mean and maximum** of the 13 base features across available vibration channels.

This creates:

## **26 base numerical features**

and provides a common model schema across different IMS sensor configurations.

---

# 24. NASA Failure Labelling and Leakage Prevention

The NASA model is a **binary failure-window classifier**.

For bearings documented as failed at run end:

```text
More than 24 hours before run end
              ↓
failure_within_horizon = 0


Within final 24 hours
              ↓
failure_within_horizon = 1
```

Other bearings are treated as right-censored controls because experiment end does not imply that those bearings exhausted their useful life.

Metadata excluded from model inputs includes:

```text
run_id
timestamp
snapshot_index
bearing_id
failed_at_run_end
hours_to_run_end
failure_within_horizon
```

In particular, `hours_to_run_end` is retained only for evaluation.

Using it as a feature would directly reveal proximity to the labelled failure window and introduce target leakage.

The model therefore learns from **vibration-derived condition features**, rather than bearing identity or explicit knowledge of future failure timing.

---

# 25. Original NASA Held-Out Evaluation

The original experiment was defined before inspecting Set 3:

```text
Set 1 → Model Training
Set 2 → Threshold Selection
Set 3 → Untouched Final Test
```

The original model achieved:

| Metric                        |     Score |
| ----------------------------- | --------: |
| ROC-AUC                       | **0.969** |
| PR-AUC                        | **0.793** |
| Precision                     | **0.704** |
| Recall                        | **0.690** |
| F1                            | **0.697** |
| Validation-selected threshold |  **0.05** |

This represents the original held-out evaluation.

---

# 26. What the Held-Out Run Revealed

After evaluating the original model, exploratory analysis of Set 3 showed an important pattern.

Vibration increased not only on Bearing 3, the documented failed bearing, but also on Bearings 1, 2, and 4.

The final-24-hour changes were:

| Bearing       | RMS Increase | Peak Increase | Original Alerts | First Alert Before Run End |
| ------------- | -----------: | ------------: | --------------: | -------------------------: |
| Bearing 1     |        1.46× |         1.47× |               2 |                     1.17 h |
| Bearing 2     |        1.77× |         1.55× |              10 |                     3.33 h |
| **Bearing 3** |    **3.51×** |     **4.21×** |         **100** |                **21.83 h** |
| Bearing 4     |        2.34× |         2.00× |              30 |                     9.17 h |

Bearing 3 shows both the **largest vibration increase** and the **earliest alerting**.

However, the other bearings also show elevated vibration near run end.

Because all four bearings are mounted on the same shaft and test rig, a severe fault developing at Bearing 3 can plausibly transmit vibration through the shaft, bearing housings, and supporting structure.

Sensors mounted near the other bearings can therefore detect the broader mechanical response even when those bearings are not documented as the originating failed component.

The alert timing supports this interpretation:

* every Bearing 1 alert coincided with a Bearing 3 alert;
* all 10 Bearing 2 alerts coincided with Bearing 3 alerts;
* all 30 Bearing 4 alerts coincided with Bearing 3 alerts;
* Bearing 3 began alerting substantially earlier than the other bearings.

Therefore, calling all 42 non-failed-bearing alerts simple model mistakes would be incomplete.

Under the **component-level labels**, they are false positives because Bearings 1, 2, and 4 do not have documented failures by run end.

Physically, however, some may represent detection of a **system-level mechanical response propagating through the shared rig**.

The useful interpretation is:

* **Bearing 3 is the likely fault origin**, because its degradation is strongest and appears earliest.
* **Bearings 1, 2, and 4 show secondary vibration responses** later in the run.
* The original classifier detects abnormal bearing-local vibration but does not explicitly localize the originating component.

This distinction between **fault detection** and **fault localization** motivated the next feature-engineering step.

---

# 27. Cross-Bearing Relative Feature Engineering

To improve localization, the pipeline adds **point-in-time cross-bearing relative features**.

For selected vibration indicators:

* RMS
* standard deviation
* peak
* kurtosis
* crest factor
* spectral centroid
* spectral-band power

each bearing is compared with the median value across all four bearings at the **same timestamp**.

Two transformations are created:

```text
relative_difference
    = bearing value - median across bearings

relative_ratio
    = bearing value / median across bearings
```

Conceptually:

```text
                    Same Timestamp

Bearing 1 ──┐
Bearing 2 ──┤
Bearing 3 ──┼────→ Cross-Bearing Median
Bearing 4 ──┘               │
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
      Bearing Value - Median    Bearing Value / Median
                │                         │
                └────────────┬────────────┘
                             ▼
                  Relative Severity Features
                             ▼
                         LightGBM
```

Example:

```text
Bearing 1 RMS = 2.0
Bearing 2 RMS = 2.1
Bearing 3 RMS = 5.2
Bearing 4 RMS = 2.0

Cross-bearing median ≈ 2.05

Bearing 3 relative ratio
≈ 5.2 / 2.05
≈ 2.54
```

The absolute RMS value tells the model that Bearing 3 has strong vibration.

The relative feature additionally tells the model that Bearing 3 is vibrating much more strongly than its peers at the same moment.

These features are intended to help distinguish:

```text
Shared Rig-Level Vibration

          from

Bearing-Specific Abnormality
```

Importantly, the transformation is **point-in-time safe**.

It uses only simultaneous sensor measurements and does not use:

* future observations
* failure labels
* bearing identity
* run-end information

---

# 28. Evaluation Protocol and Test-Set Feedback

There is an important evaluation caveat.

The original Set 3 experiment was genuinely held out:

```text
Set 1 → Training
Set 2 → Validation
Set 3 → Untouched Test
```

The original Set 3 analysis then revealed the shared-rig vibration pattern.

That observation influenced the design of the cross-bearing relative features.

The same broad physical pattern was subsequently observed in Set 1: non-failed bearings moved with the rig while the documented failed Bearings 3 and 4 showed stronger local behaviour.

Set 1 supports the physical plausibility of the feature, but because it is training data it is not independent evidence of generalization.

The updated model was then:

```text
Set 1 → Training
Set 2 → Threshold Selection
Set 3 → Exploratory Re-Evaluation
```

At this point, Set 3 was no longer a pristine unseen test set.

This is:

## **Test-Set Feedback Bias**

rather than direct future-information leakage.

The feature itself uses only contemporaneous measurements and treats bearings symmetrically.

No future observations, run-end timing, bearing identity, or failure labels are used to calculate the cross-bearing features.

However, because observations from Set 3 influenced the decision to create those features, the updated Set 3 metrics are no longer an unbiased estimate of generalization performance.

---

# 29. Cross-Bearing Exploratory Results

The updated model was trained on Set 1, tuned on Set 2, and re-evaluated on Set 3.

| Metric                        | Original Model | Cross-Bearing Model |
| ----------------------------- | -------------: | ------------------: |
| ROC-AUC                       |          0.969 |           **0.986** |
| PR-AUC                        |          0.793 |           **0.889** |
| Precision                     |          0.704 |           **0.802** |
| Recall                        |          0.690 |           **0.897** |
| F1                            |          0.697 |           **0.847** |
| Validation-selected threshold |           0.05 |            **0.10** |
| Non-failed-bearing alerts     |             42 |              **32** |
| Non-failed-bearing alert rate |         ~0.22% |          **~0.17%** |

Within Bearing 3's final 24-hour failure window:

| Outcome            |      Count |
| ------------------ | ---------: |
| Positive snapshots |    **145** |
| Correctly alerted  |    **130** |
| Missed             |     **15** |
| Recall             | **89.66%** |

The cross-bearing model detected more of Bearing 3's pre-failure observations while reducing alerts on the other bearings.

However:

> **The 0.986 ROC-AUC and 0.889 PR-AUC should not be interpreted as performance on a never-seen test set.**

They are exploratory results on a reused evaluation run.

---

# 30. Early Warning Analysis

Classification metrics alone do not capture the operational usefulness of predictive maintenance.

A key question is:

> **How early did the model begin identifying the developing failure?**

For Bearing 3:

```text
First Alert Within
24-Hour Failure Window
Risk ≥ Threshold
        │
        │
        │ ~21.8 hours
        │
        ▼
Documented Run End
```

The first alert within the defined target window occurred approximately:

## **21.8 hours before documented run end**

For the cross-bearing model, **130 of 145** Bearing 3 observations within that window were alerted.

Importantly:

## **21.8 hours is an observed warning lead time, not an RUL prediction.**

The model does not predict:

> “Bearing 3 has exactly 21.8 hours of useful life remaining.”

Instead, each vibration snapshot is classified according to whether its condition resembles the defined pre-failure state.

During evaluation, the first threshold-crossing alert within the 24-hour target window occurred approximately 21.8 hours before documented run end.

---

# 31. NASA IMS Dashboard

Launch:

```bash
streamlit run dashboard/ims_app.py --server.port=8502
```

Open:

```text
http://localhost:8502
```

The dashboard provides:

### Fleet Overview

* latest bearing risk
* alert state
* bearing status

### Bearing Drill-Down

* complete risk trajectory
* final 24-hour window
* vibration-feature trends
* alert timing

### Fault Localization

Comparison of:

* model risk
* absolute vibration severity
* peer-relative vibration severity

### Explainability

* local SHAP drivers
* global feature drivers

### Model Evaluation

* ROC-AUC
* PR-AUC
* precision
* recall
* F1
* confusion matrix
* non-failed-bearing alert rate
* warning timing

The dashboard connects:

```text
Vibration Signal
      ↓
Model Risk
      ↓
Relative Bearing Severity
      ↓
SHAP Explanation
      ↓
Engineering Interpretation
```

---

# 32. Maintenance Feedback Loop

A production predictive-maintenance system should not stop after generating an alert.

Maintenance outcomes should be written back to the data platform:

```text
Model Alert
     ↓
Engineer Inspection
     ↓
Maintenance Action
     ↓
Outcome
     │
     ├── Confirmed fault
     ├── No fault found
     ├── Component replaced
     ├── Downtime
     ├── Parts cost
     └── Labour cost
     ↓
Historical Maintenance Dataset
     ↓
Threshold Calibration
Drift Monitoring
Retraining
Champion / Challenger
```

This feedback enables:

* maintenance-cost-based threshold selection
* false-alert analysis
* data drift monitoring
* feature drift monitoring
* model performance monitoring
* supervised retraining
* champion/challenger comparison
* maintenance-policy optimization

This feedback loop separates a one-time ML model from a continuously improving maintenance system.

---

# 33. Quick Start

## Create Environment

```bash
python -m venv .venv
```

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

---

## Run the Synthetic Pipeline

```bash
python -m src.generate_data
python -m src.train
python -m src.batch_score
```

Launch dashboard:

```bash
streamlit run dashboard/app.py
```

Launch API:

```bash
uvicorn api.app:app --reload
```

Launch MLflow:

```bash
mlflow ui
```

---

## Run Hyperparameter Search

Synthetic + NASA:

```bash
python -m src.tune_hyperparameters --dataset both --n-iter 12
```

Tuning outputs are stored in:

```text
artifacts/tuning/
```

The tuning experiment does not automatically overwrite the deployed models.

---

## Run NASA IMS Pipeline

Download and extract:

```bash
python -m src.ims_download
```

Extract vibration features:

```bash
python -m src.ims_features
```

Train and evaluate:

```bash
python -m src.ims_train
```

Quick preprocessing check:

```bash
python -m src.ims_features \
    --max-files 20 \
    --output data/processed/ims_features_smoke.parquet
```

Launch NASA dashboard:

```bash
streamlit run dashboard/ims_app.py --server.port=8502
```


---

# 34. Repository Structure

```text
predictive-maintenance-project/
│
├── api/
│   └── app.py
│
├── artifacts/
│   ├── tuning/
│   ├── test_metrics.json
│   ├── feature_importance.csv
│   ├── threshold_sweep.csv
│   ├── shap_global_importance.csv
│   ├── shap_local_explanations.csv
│   ├── shap_beeswarm.png
│   ├── shap_bar.png
│   ├── ims_test_metrics.json
│   ├── ims_feature_importance.csv
│   ├── ims_shap_global_importance.csv
│   ├── ims_shap_local_explanations.csv
│   ├── ims_shap_beeswarm.png
│   └── ims_shap_bar.png
│
├── dashboard/
│   ├── app.py
│   └── ims_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   └── images/
│       ├── synthetic-dashboard.png
│       └── ims-dashboard-overview.png
│
├── models/
│
├── scripts/
│   └── capture_dashboards.py
│
├── src/
│   ├── generate_data.py
│   ├── features.py
│   ├── train.py
│   ├── tune_hyperparameters.py
│   ├── threshold_analysis.py
│   ├── batch_score.py
│   ├── spark_pipeline.py
│   ├── ims_download.py
│   ├── ims_features.py
│   └── ims_train.py
│
├── tests/
│   └── test_features.py
│
├── config.yaml
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

# 35. Technology Stack

| Layer                      | Technologies                                                      |
| -------------------------- | ----------------------------------------------------------------- |
| **Data Science**           | Python, Pandas, NumPy, LightGBM, scikit-learn                     |
| **Signal Processing**      | FFT, time-domain and frequency-domain vibration features          |
| **Feature Engineering**    | Rolling telemetry + cross-bearing relative features               |
| **Model Selection**        | Randomized hyperparameter search + validation PR-AUC              |
| **Operational Evaluation** | Event detection, lead time, alert episodes, workload, persistence |
| **Explainability**         | SHAP / TreeSHAP                                                   |
| **Data Engineering**       | PySpark, Parquet                                                  |
| **Experiment Tracking**    | MLflow                                                            |
| **Model Serving**          | FastAPI                                                           |
| **Visualization**          | Streamlit, Plotly                                                 |
| **Deployment**             | Docker, Docker Compose                                            |
| **Testing**                | pytest                                                            |

---

# 36. Production Extensions

A fuller production implementation could evolve toward:

```text
Machine / PLC / Sensor
          ↓
SCADA / IoT Gateway
          ↓
Kafka / Azure Event Hubs
          ↓
Spark Structured Streaming
          ↓
Delta Lake
Bronze → Silver → Gold
          ↓
Feature Store
          ↓
Model Registry
          ↓
Approved Model
          ↓
Online + Batch Model Serving
          ↓
MES / CMMS / Dashboard
          ↓
Maintenance Outcomes
          ↓
Monitoring + Retraining
```

Potential extensions include:

1. Streaming ingestion with Kafka or Azure Event Hubs.
2. Delta Lake Bronze/Silver/Gold architecture.
3. Airflow or Prefect orchestration.
4. Centralized feature-store support.
5. Maintenance-cost-based threshold optimization.
6. Data, feature, prediction, and model drift monitoring.
7. Maintenance-outcome capture for supervised retraining.
8. Model registry and champion/challenger deployment.
9. CI/CD for automated testing and deployment.
10. Independent validation of the cross-bearing features.
11. Remaining Useful Life modelling using NASA C-MAPSS or another suitable dataset.

---

# 37. Next Validation Experiment

The most important next modelling extension is **independent validation**, rather than simply adding another algorithm.

The cross-bearing features were designed after examining Set 3.

The next experiment should therefore freeze:

* feature definitions
* preprocessing
* model family
* hyperparameter search procedure
* threshold-selection procedure
* evaluation metrics

and apply them to new run-to-failure data.

Potential datasets include:

### FEMTO / PRONOSTIA

Provides accelerated bearing degradation trajectories suitable for evaluating whether relative vibration features generalize across independent experiments.

### XJTU-SY

Provides multiple bearing run-to-failure trajectories under different operating conditions.

### NASA C-MAPSS

Provides multiple turbofan degradation trajectories and is particularly suitable for a separate **Remaining Useful Life** experiment.

The goal is to progress from:

```text
Promising Exploratory Feature Engineering
                 ↓
Frozen Modelling Procedure
                 ↓
Independent Validation
                 ↓
Evidence of Generalization
```

---

# 38. Limitations

The NASA IMS experiment is a **controlled accelerated run-to-failure benchmark**, not validation under ordinary factory operations.

Important limitations include:

* the number of complete experimental runs is small;
* the number of documented failed bearings is limited;
* experimental conditions differ from production environments;
* the 24-hour horizon is a modelling choice;
* bearings without documented failure are right-censored;
* non-failed-bearing vibration can contain the mechanical response of a fault elsewhere in the shared rig;
* cross-bearing features require simultaneous measurements from peer components;
* the original Set 3 evaluation is held out, but the later cross-bearing results reuse Set 3 after it influenced feature design;
* improved cross-bearing metrics therefore represent exploratory re-evaluation rather than an unbiased final test;
* the ~21.8-hour warning result comes from the documented failed Bearing 3 in Set 3;
* hyperparameter tuning results depend on the defined search space and validation period;
* a statistically stronger model may not provide the preferred operational trade-off;
* LightGBM risk scores are not assumed to be calibrated real-world failure probabilities;
* SHAP explains model behaviour rather than physical causality;
* the synthetic benchmark does not represent expected production performance.

A real deployment would additionally require:

* equipment-specific validation
* prospective validation
* sensor-quality monitoring
* cybersecurity controls
* access control
* model governance
* change management
* integration with maintenance processes
* appropriate engineering and safety review

---

# 39. Key Takeaway

This project demonstrates more than training a classifier.

It connects:

```text
Sensor / PLC Data
      ↓
Data Ingestion
      ↓
Data Preparation
      ↓
Signal Processing
      ↓
Feature Engineering
      ↓
Machine Learning
      ↓
Hyperparameter Selection
      ↓
Temporal / Run-Separated Validation
      ↓
Statistical Metrics
      ↓
Operational Metrics
      ↓
Explainability
      ↓
Model Serving
      ↓
Maintenance Decision
      ↓
Outcome Feedback
```

The **synthetic manufacturing pipeline** demonstrates how shop-floor data can move through data preparation, feature engineering, model development, deployment, and maintenance decision support.

The **hyperparameter experiment** demonstrates that improving statistical metrics does not automatically improve operational outcomes.

The **operational metrics** connect ML performance with failure coverage, warning time, alert workload, and maintenance usability.

The **NASA IMS experiment** demonstrates how physical vibration measurements can be converted into interpretable condition-monitoring features.

The **held-out analysis** demonstrates how unexpected predictions can reveal useful information about the physical system.

The **cross-bearing experiment** demonstrates a realistic data-science iteration:

```text
Initial Model
     ↓
Held-Out Evaluation
     ↓
Unexpected Pattern
     ↓
Physical Interpretation
     ↓
Feature Engineering
     ↓
Exploratory Re-Evaluation
     ↓
Need for Independent Validation
```

The central question is therefore not only:

> **Can the model predict failure risk?**

but also:

> **Can the data be prepared correctly, the model validated honestly, the alerts evaluated operationally, and the output translated into a useful maintenance decision?**

---

# Disclaimer

This project is intended for **educational and portfolio purposes**.

It combines reproducible synthetic manufacturing telemetry with physical vibration measurements from the NASA IMS accelerated bearing-life experiments.

The original NASA Set 3 result represents a run-separated held-out evaluation. Later cross-bearing feature engineering was informed by analysis of that test run, so the improved cross-bearing results are explicitly reported as **exploratory re-evaluation rather than independent test performance**.

The hyperparameter experiments demonstrate model-selection methodology and operational trade-offs rather than evidence that a particular configuration will generalize to production equipment.

Neither the synthetic benchmark nor the NASA experiment represents validation under ordinary factory operating conditions.

Model outputs should not be used for real maintenance or safety decisions without prospective equipment-specific validation, appropriate engineering review, and integration with established operational procedures.
