# Predictive Maintenance Platform

An end-to-end **Data Science + Data Engineering + MLOps** portfolio project for predictive maintenance using both synthetic industrial telemetry and physical NASA IMS bearing vibration data.

The project goes beyond a modelling notebook. It demonstrates the complete workflow from sensor data to maintenance decision support:

```text
Machine / Sensor Data
        ↓
Data Processing
        ↓
Feature Engineering
        ↓
Machine Learning
        ↓
Model Evaluation
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

1. **Manufacturing Deployment Pipeline** — synthetic industrial telemetry demonstrates temporal feature engineering, LightGBM, MLflow, PySpark, batch scoring, FastAPI, Streamlit, SHAP, Docker, and testing.

2. **NASA IMS Bearing Experiment** — physical run-to-failure vibration measurements are transformed into time-domain, frequency-domain, and cross-bearing relative features for failure-window detection and fault-localization analysis.

---

*The main dashboard uses the reproducible synthetic telemetry pipeline and ranks current machine risk for maintenance review.*

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

The model score is used as a **relative failure-risk indicator** rather than assumed to be a perfectly calibrated real-world failure probability.

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
Rolling Features                       Signal Processing
          │                                         │
          │                              Time-Domain Features
          │                              Frequency Features
          │                              Cross-Bearing Features
          │                                         │
          ▼                                         ▼
      LightGBM                                  LightGBM
          │                                         │
          ▼                                         ▼
Temporal Validation                     Run-Separated Evaluation
          │                                         │
          ▼                                         ▼
MLflow + SHAP                           SHAP + Fault Analysis
          │                                         │
          ▼                                         ▼
Batch / API / Dashboard                  IMS Dashboard
```

The **synthetic pipeline** demonstrates how predictive-maintenance modelling can be operationalized.

The **NASA IMS experiment** demonstrates signal processing and predictive-maintenance modelling using physical vibration measurements from a controlled bearing run-to-failure experiment.

---

# 3. Key Results

## 3.1 Original NASA IMS Held-Out Evaluation

The initial experiment was defined before inspecting Set 3:

```text
Set 1 → Training
Set 2 → Validation / Threshold Selection
Set 3 → Untouched Test
```

The original model achieved:

| Metric             | Original Set 3 Result |
| ------------------ | --------------------: |
| ROC-AUC            |             **0.969** |
| PR-AUC             |             **0.793** |
| Precision          |             **0.704** |
| Recall             |             **0.690** |
| F1                 |             **0.697** |
| Selected threshold |              **0.05** |

This represents the original held-out evaluation.

---

## 3.2 Cross-Bearing Exploratory Re-Evaluation

Analysis of the original Set 3 results revealed that vibration also increased on bearings without documented failure.

This observation motivated additional **cross-bearing relative features** designed to distinguish shared rig-level vibration from bearing-specific abnormality.

After introducing those features:

| Metric                          | Base Features | + Cross-Bearing Features |
| ------------------------------- | ------------: | -----------------------: |
| ROC-AUC                         |         0.969 |                **0.986** |
| PR-AUC                          |         0.793 |                **0.889** |
| Precision                       |         0.704 |                **0.802** |
| Recall                          |         0.690 |                **0.897** |
| F1                              |         0.697 |                **0.847** |
| Non-failed-bearing alerts       |            42 |                   **32** |
| Non-failed-bearing alert rate   |        ~0.22% |               **~0.17%** |
| First in-window Bearing 3 alert |       ~21.8 h |              **~21.8 h** |

However, these improved metrics are reported as:

## **Exploratory re-evaluation**

rather than performance on a pristine unseen test set.

Set 3 had already influenced feature design, so the updated Set 3 result contains **test-set feedback bias**.

The feature calculation itself remains point-in-time safe and does not use future observations or failure labels, but a new independent dataset would be required for an unbiased estimate of the improved model's generalization performance.

---

## 3.3 Synthetic Manufacturing Benchmark

The default synthetic benchmark contains:

* 50 machines
* 120 days
* hourly telemetry
* approximately 144,000 observations

A reproducible verification run produced:

| Metric             |     Score |
| ------------------ | --------: |
| ROC-AUC            | **0.799** |
| PR-AUC             | **0.108** |
| Selected threshold |  **0.10** |
| Precision          | **0.090** |
| Recall             | **0.679** |
| F1                 | **0.160** |

The synthetic experiment primarily demonstrates **system architecture, temporal validation, model serving, monitoring, and MLOps**, rather than expected real-world predictive performance.

---

# 4. End-to-End Manufacturing Architecture

```text
Factory Equipment
temperature / vibration / pressure / RPM / power / load
                         │
                         ▼
                Telemetry Ingestion
             CSV locally / IoT in production
                         │
                         ▼
                  Bronze Data Layer
              Immutable raw observations
                         │
                         ▼
              Validation + Silver Layer
        Schema / timestamps / deduplication
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
               Failure-Risk Scoring
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

In a production environment, the same component boundaries could connect to PLC/SCADA systems, historians, IoT gateways, Kafka or Azure Event Hubs, Delta Lake, MES, CMMS, and maintenance-planning systems.

---

# 5. Manufacturing Data Pipeline

## 5.1 Equipment Telemetry

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

In production:

```text
Machine / PLC / Sensor
          ↓
SCADA / Historian / IoT Gateway
          ↓
Kafka / Azure Event Hubs
          ↓
Data Platform
```

Raw observations would first be persisted in an append-only **Bronze layer**.

Keeping the original events supports:

* late-arriving data handling
* data-quality investigation
* auditability
* feature recomputation
* historical replay

---

# 6. PySpark Data Engineering Layer

The optional PySpark layer demonstrates how telemetry processing could scale beyond a local Pandas workflow.

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

A production architecture could evolve toward:

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

---

# 7. Synthetic Feature Engineering

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

---

# 8. Temporal Validation

Predictive-maintenance data is time-dependent.

A random train/test split can allow future machine behaviour to influence model development.

The synthetic experiment therefore uses:

```text
Past ------------------------------------------→ Future

|            Train            | Validation | Test |
```

A **24-hour purge** is applied before relevant split boundaries.

This prevents observations whose labels look 24 hours into the future from referring to failures occurring inside the next partition.

Each partition has a distinct purpose:

```text
TRAIN
  ↓
Fit LightGBM

VALIDATION
  ↓
Select Alert Threshold

TEST
  ↓
Final Evaluation
```

This more closely approximates future deployment than randomly mixing historical observations.

---

# 9. Training and Inference

Training and inference are intentionally separated.

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
Save Model Bundle
```

The saved model bundle contains:

* trained LightGBM model
* ordered feature contract
* selected threshold
* failure horizon
* model version

The approved model can subsequently score new telemetry without retraining every time new data arrives.

## Batch / Online Inference

```text
New Data
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

Retraining is a separate lifecycle process that can be triggered when sufficient new labelled data becomes available or monitoring indicates model degradation.

---

# 10. Risk-to-Action Layer

The model generates a **failure-risk score**.

The operating alert threshold is selected using validation data only.

For the synthetic benchmark:

```text
Selected threshold = 0.10
```

The output contains:

* `failure_risk` — model-estimated risk score
* `alert` — whether the score exceeds the selected threshold
* `risk_band` — LOW, MEDIUM, or HIGH
* SHAP contributions — features pushing the score higher or lower

Example demonstration bands:

```text
risk < selected threshold
        → LOW

selected threshold ≤ risk < 0.60
        → MEDIUM

risk ≥ 0.60
        → HIGH
```

The HIGH cutoff is a demonstration value.

In production, thresholds should consider:

* missed-failure cost
* unnecessary inspection cost
* equipment criticality
* downtime cost
* technician capacity
* spare-parts availability
* production constraints

---

# 11. SHAP Explainability

Maintenance engineers need more than a risk score.

They also need to understand:

> **Why did the model assign this equipment a high risk score?**

The project uses **TreeSHAP** for both global and local explanations.

## Global SHAP

Generated artifacts include:

```text
artifacts/shap_global_importance.csv
artifacts/shap_beeswarm.png
artifacts/shap_bar.png

artifacts/ims_shap_global_importance.csv
artifacts/ims_shap_beeswarm.png
artifacts/ims_shap_bar.png
```

Global SHAP helps answer:

> **Which features does the model generally rely on?**

## Local SHAP

Local SHAP explains individual predictions.

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

Local explanation files retain:

* machine or bearing context
* predicted risk
* observed label
* feature value
* SHAP contribution
* contribution direction
* strongest model drivers

The API exposes explanations through:

```text
POST /predict/explain
```

SHAP explains **model behaviour**, not physical causality.

A positive SHAP contribution indicates that a feature pushed the model toward a higher risk score. It does not prove that the feature is the physical root cause of failure.

---

# 12. Batch Scoring

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

A production scheduler could run this process periodically and publish the results to a dashboard, MES, CMMS, or maintenance-planning system.

---

# 13. FastAPI Model Serving

The trained model can be consumed programmatically through **FastAPI**.

Start the API:

```bash
uvicorn api.app:app --reload
```

Swagger documentation is available at:

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

# 14. Manufacturing Dashboard

Launch the synthetic manufacturing dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard provides a human decision-support layer.

A reliability engineer or maintenance planner can:

1. Review fleet-level machine counts and active alerts.
2. Identify high-risk assets.
3. Rank machines by risk score.
4. Select an individual machine.
5. Inspect recent sensor trends.
6. Review local SHAP explanations.
7. Compare individual drivers with global model behaviour.
8. Decide whether to inspect, schedule diagnostics, or continue monitoring.

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

# 15. MLflow Experiment Tracking

Training experiments are tracked using **MLflow**.

The pipeline records:

* ROC-AUC
* PR-AUC
* precision
* recall
* F1
* selected threshold
* failure horizon
* feature count
* model configuration
* model artifacts

Start MLflow:

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

This provides traceability between model configuration, evaluation results, and saved artifacts.

---

# 16. NASA IMS Bearing Experiment

The repository also contains a separate experiment using the **NASA IMS Bearings dataset**, which contains vibration measurements from physical bearing run-to-failure experiments.

The objective is:

> **Can vibration characteristics identify that a bearing is approaching a defined failure window before documented run end?**

The NASA workflow is:

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

# 17. NASA Experimental Split

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
   │
   ▼
Training

Set 2
2nd_test
   │
   ▼
Threshold Selection

Set 3
4th_test
   │
   ▼
Original Held-Out Evaluation
```

Holding out complete runs provides a stronger evaluation than randomly mixing observations from the same degradation trajectory.

> **Dataset naming:** NASA's third experimental set is stored in a directory named `4th_test`. The project therefore uses `4th_test` internally for Set 3; this does not represent a fourth experimental set.

For the held-out experiment, Bearing 3 is the documented failed bearing.

Bearings 1, 2, and 4 do not have documented failures by experiment end and are therefore treated as **right-censored controls**, rather than assumed to be permanently healthy.

---

# 18. NASA Vibration Feature Engineering

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

For each bearing, the pipeline calculates the **mean and maximum** of the 13 base features across the available vibration channels.

This creates:

## **26 base numerical features**

and provides a common model schema across the different IMS sensor configurations.

---

# 19. Failure Labelling and Leakage Prevention

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

Several metadata fields are explicitly excluded from model inputs:

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

The LightGBM model therefore learns from **vibration-derived condition features**, rather than bearing identity or explicit knowledge of future failure timing.

---

# 20. Original NASA Held-Out Evaluation

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

Because failure-window observations are rare, accuracy alone would provide a misleading view of performance.

**ROC-AUC** measures ranking quality across thresholds.

**PR-AUC** is particularly informative because the positive pre-failure class is uncommon.

At the selected threshold, the original model achieved approximately **70% precision and 69% recall**.

---

# 21. What the Held-Out Run Revealed

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

Physically, however, some may represent valid detection of a **system-level mechanical response propagating through the shared rig**.

The useful interpretation is:

* **Bearing 3 is the likely fault origin**, because its degradation is strongest and appears earliest.
* **Bearings 1, 2, and 4 show secondary vibration responses** later in the run.
* The original classifier detects abnormal bearing-local vibration but does not explicitly localize the originating component.

This distinction between **fault detection** and **fault localization** motivated the next feature-engineering step.

---

# 22. Cross-Bearing Relative Feature Engineering

The observation above suggested that absolute vibration magnitude alone could partially represent shared rig-level vibration.

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

Two relative transformations are created:

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
                │                         │
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

# 23. Evaluation Protocol and Test-Set Feedback

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

Set 1 therefore supports the physical plausibility of the feature, but because it is training data it is not independent evidence of generalization.

The updated model was then:

```text
Set 1 → Training
Set 2 → Threshold Selection
Set 3 → Exploratory Re-Evaluation
```

At this point, Set 3 was no longer a pristine unseen test set.

This is:

## **Test-set feedback bias**


The feature itself uses only contemporaneous measurements and treats bearings symmetrically. No future observations, run-end timing, bearing identity, or failure labels are used to calculate the cross-bearing features.

However, because observations from Set 3 influenced the decision to create those features, the updated Set 3 metrics are no longer an unbiased estimate of generalization performance.

A rigorous next experiment should therefore:

1. freeze the cross-bearing feature definition;
2. freeze model-selection decisions;
3. evaluate on independent run-to-failure data.

Potential datasets include FEMTO/PRONOSTIA or XJTU-SY.

---

# 24. Cross-Bearing Exploratory Results

The updated model was trained on Set 1, tuned on Set 2, and re-evaluated on Set 3.

The exploratory results were:

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

The cross-bearing model therefore detected more of Bearing 3's pre-failure observations while reducing alerts on the other bearings.

These results support the cross-bearing feature engineering as a promising modelling direction.

However:

> **The 0.986 ROC-AUC and 0.889 PR-AUC should not be interpreted as performance on a never-seen test set.**

They are exploratory results on a reused evaluation run.

---

# 25. Early Warning Analysis

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
 / Bearing Failure
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

# 26. NASA IMS Dashboard

Launch the dedicated NASA dashboard:

```bash
streamlit run dashboard/ims_app.py --server.port=8502
```

Then open:

```text
http://localhost:8502
```

The dashboard provides several analytical views.

## Fleet Overview

Displays the latest:

* risk score
* alert state
* bearing status

for all four bearings.

## Bearing Drill-Down

Allows inspection of:

* full risk trajectory
* final 24-hour failure window
* vibration-feature trends
* alert timing

## Fault Localization

Compares:

* model risk
* absolute vibration severity
* peer-relative vibration severity

across bearings.

## Explainability

Displays:

* local SHAP drivers for the selected bearing
* global model feature drivers

## Model Evaluation

Displays:

* ROC-AUC
* PR-AUC
* precision
* recall
* F1
* confusion matrix
* non-failed-bearing alert rate
* warning timing

The dashboard therefore connects:

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

# 27. Maintenance Feedback Loop

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
* model comparison
* maintenance-policy optimization

This feedback loop separates a one-time ML model from a continuously improving maintenance system.

---

# 28. Quick Start

## 28.1 Create Environment

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

## 28.2 Run the Synthetic Pipeline

Generate telemetry:

```bash
python -m src.generate_data
```

Train the model:

```bash
python -m src.train
```

Generate current machine risk ranking:

```bash
python -m src.batch_score
```

Launch the manufacturing dashboard:

```bash
streamlit run dashboard/app.py
```

Launch FastAPI:

```bash
uvicorn api.app:app --reload
```

Launch MLflow:

```bash
mlflow ui
```

---

## 28.3 Run the NASA IMS Pipeline

Download and extract the dataset:

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

For a quick preprocessing check:

```bash
python -m src.ims_features \
    --max-files 20 \
    --output data/processed/ims_features_smoke.parquet
```

Launch the NASA IMS dashboard:

```bash
streamlit run dashboard/ims_app.py --server.port=8502
```

---

# 29. Suggested Portfolio Demonstration

A concise interview demonstration can be structured around five questions.

## 1. What is the business problem?

Unexpected failures cause downtime and maintenance cost.

The objective is to identify equipment showing pre-failure behaviour early enough for engineers to investigate.

## 2. How did you prevent leakage?

Show:

* chronological synthetic split
* 24-hour purge
* complete-run NASA separation
* exclusion of bearing identity and `hours_to_run_end`

## 3. What did you discover?

Show the final-24-hour bearing comparison.

Explain that vibration increased across multiple bearings because the components share the same mechanical rig, but Bearing 3 showed the strongest and earliest degradation.

## 4. What did you do about it?

Explain the peer-relative features:

```text
Bearing Signal
      ↓
Compare Against Simultaneous Peers
      ↓
Relative Difference / Ratio
      ↓
Fault-Localization Features
```

Then explicitly mention that because Set 3 informed this idea, the improved results are exploratory.

## 5. How would engineers use the model?

Show:

* risk trajectory
* threshold
* peer-bearing comparison
* SHAP
* warning timing
* maintenance decision

The complete story becomes:

```text
Business Problem
      ↓
Data Engineering
      ↓
Initial Model
      ↓
Held-Out Evaluation
      ↓
EDA Observation
      ↓
Physical Hypothesis
      ↓
Feature Engineering
      ↓
Exploratory Re-Evaluation
      ↓
Explainability
      ↓
Maintenance Decision
```

---

# 30. Repository Structure

```text
predictive-maintenance-project/
│
├── api/
│   └── app.py
│
├── artifacts/
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
├── models/
│
├── src/
│   ├── generate_data.py
│   ├── features.py
│   ├── train.py
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
└── README.md
```

---

# 31. Technology Stack

| Layer                   | Technologies                                             |
| ----------------------- | -------------------------------------------------------- |
| **Data Science**        | Python, Pandas, NumPy, LightGBM, scikit-learn            |
| **Signal Processing**   | FFT, time-domain and frequency-domain vibration features |
| **Feature Engineering** | Rolling telemetry + cross-bearing relative features      |
| **Explainability**      | SHAP / TreeSHAP                                          |
| **Data Engineering**    | PySpark, Parquet                                         |
| **Experiment Tracking** | MLflow                                                   |
| **Model Serving**       | FastAPI                                                  |
| **Visualization**       | Streamlit, Plotly                                        |
| **Deployment**          | Docker, Docker Compose                                   |
| **Testing**             | pytest                                                   |

---

# 32. Production Extensions

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
5. Maintenance-cost-based threshold calibration.
6. Data, feature, prediction, and model drift monitoring.
7. Maintenance-outcome capture for supervised retraining.
8. Model registry and champion/challenger deployment.
9. CI/CD for automated testing and deployment.
10. Independent validation of the cross-bearing features.
11. Remaining Useful Life modelling using an appropriate dataset such as NASA C-MAPSS.

---

# 33. Next Validation Experiment

The most important modelling extension is **not another algorithm**.

It is independent validation of the feature-engineering hypothesis.

The current cross-bearing features were designed after examining Set 3.

The next experiment should therefore freeze:

* feature definitions
* preprocessing
* model family
* hyperparameter search process
* threshold-selection procedure
* evaluation metrics

and apply them to new run-to-failure data.

Potential datasets include:

### FEMTO / PRONOSTIA

Provides accelerated bearing degradation trajectories and is suitable for evaluating whether relative vibration features generalize across independent bearing experiments.

### XJTU-SY

Provides multiple bearing run-to-failure trajectories under different operating conditions.

### NASA C-MAPSS

Provides multiple turbofan degradation trajectories and is particularly suitable for a separate **Remaining Useful Life** experiment.

This would allow the project to progress from:

```text
Promising Exploratory Feature Engineering
                 ↓
Independent Validation
                 ↓
Evidence of Generalization
```

---

# 34. Limitations

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
* the improved cross-bearing metrics therefore represent exploratory re-evaluation rather than an unbiased final test;
* the ~21.8-hour warning result comes from the documented failed Bearing 3 in Set 3;
* SHAP explains model behaviour rather than physical causality;
* the synthetic benchmark does not represent expected production performance;
* production alert thresholds require calibration against equipment criticality and maintenance economics.

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

# 35. Key Takeaway

This project demonstrates more than training a classifier.

It connects:

```text
Sensor Data
    ↓
Data Engineering
    ↓
Signal Processing
    ↓
Feature Engineering
    ↓
Machine Learning
    ↓
Temporal / Run-Separated Validation
    ↓
Explainability
    ↓
Model Serving
    ↓
Operational Dashboard
    ↓
Maintenance Decision
    ↓
Outcome Feedback
```

The **NASA IMS experiment** demonstrates how physical vibration measurements can be converted into interpretable condition-monitoring features.

The **held-out analysis** demonstrates how model errors can reveal useful information about the underlying physical system.

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

The **synthetic manufacturing pipeline** demonstrates how predictive-maintenance modelling could fit into a broader Data Engineering and MLOps architecture.

The central question is therefore not only:

> **Can the model detect failure risk?**

but also:

> **Can the signal be interpreted, validated, explained, operationalized, and translated into a useful maintenance decision?**

---

# Disclaimer

This project is intended for **educational and portfolio purposes**.

It combines reproducible synthetic manufacturing telemetry with physical vibration measurements from the NASA IMS accelerated bearing-life experiments.

The original NASA Set 3 result represents a run-separated held-out evaluation. Later cross-bearing feature engineering was informed by analysis of that test run, so the improved cross-bearing results are explicitly reported as **exploratory re-evaluation rather than independent test performance**.

Neither the synthetic benchmark nor the NASA experiment represents validation under ordinary factory operating conditions.

Model outputs should not be used for real maintenance or safety decisions without prospective equipment-specific validation, appropriate engineering review, and integration with established operational procedures.
