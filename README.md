# Student Predictive Analytics & Smart Attendance System

**Smart India Hackathon 2025 – National Finalist** · Team VictoReign

> ### Read this first: what is original and what is new
> | Component | Status |
> |---|---|
> | Face-recognition attendance (InsightFace Buffalo_L, OpenCV, RTSP, MongoDB) | **Original project** (`Face detction.py`) |
> | Node/Express backend, React dashboards, SendGrid notifications | **Original project** |
> | Academic risk score `1 - (0.6·marks + 0.4·attendance)` with LOW / MEDIUM / HIGH | **Original project – rule-based, *not* a trained ML model** (`calculateRisk.js`) |
> | Trained ML academic-performance classifier (this README's ML sections) | **Newly reconstructed, trained on a public Kaggle dataset.** It was **not** part of the original SIH 2025 submission. |

## 1. Project overview
The system combines (1) computer-vision attendance, (2) academic risk assessment, (3) faculty/admin dashboards and (4) automated risk notifications. This repository additionally contains a reproducible machine-learning component that predicts a student's **academic performance band (Low / Middle / High)** from engagement, attendance-related and demographic indicators.

## 2. Problem statement
Attendance and academic records live in separate silos, and struggling students are usually identified only after poor results. The goal is early identification of students likely to be in the *Low* band – with particular attention to **not missing** such students (Low-class recall).

## 3. System architecture
```
 RTSP cameras ──> Face detction.py ──> MongoDB (attendance) ─┐
                                                              ├─> Node/Express backend ──> React dashboards
 Faculty input (marks) ───────────────> MongoDB (students) ──┘          │            │
                                                                        │            └─> SendGrid alerts
                                  calculateRisk.js  (rule-based baseline, always computed)
                                                                        │
                       backend_integration/mlClient.js ──HTTP──> app/model_api.py (FastAPI)
                                                                        │
                                          models/academic_performance_model.pkl (Joblib)
```
The ML service is optional. If it is down, or a student record lacks the features the model needs, the backend **falls back to the original rule-based score** and says so in the response (`source: "rule_based_fallback"`).

## 4. Technologies
Python, OpenCV, InsightFace, NumPy, pandas, scikit-learn, Joblib, FastAPI, Node.js, Express.js, React.js, MongoDB, SendGrid.

## 5. Dataset (ML component)
* **Name:** Students' Academic Performance Dataset (xAPI-Edu-Data) · **Kaggle:** <https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data> (`aljarah/xAPI-Edu-Data`).
* **Attribution:** collected from the Kalboard 360 e-learning platform using the Experience API (xAPI); credit the original authors and the Kaggle uploader (the authors are, to the best of my knowledge, Amrieh, Hamtini & Aljarah – verify on the Kaggle page). Check the license/terms on the Kaggle page yourself – **the raw CSV is deliberately not committed** (`dataset/raw/` is git-ignored) because redistribution terms were not verified.
* **Size after loading:** 480 rows × 17 columns (16 features + target `Class`), no missing values, 2 exact duplicate rows (dropped in memory → 478 rows used; the file on disk is untouched).
* **Target `Class`:** L (Low) 127 · M (Middle) 211 · H (High) 142 (before de-duplication).
* **Numeric features (0-100 counters):** `raisedhands`, `VisITedResources`, `AnnouncementsView`, `Discussion`.
* **Categorical features:** `gender`, `NationalITy`, `PlaceofBirth`, `StageID`, `GradeID`, `SectionID`, `Topic`, `Semester`, `Relation`, `ParentAnsweringSurvey`, `ParentschoolSatisfaction`, `StudentAbsenceDays` (`Under-7` / `Above-7`).
* Full EDA with plots: [`notebooks/student_performance_analysis.ipynb`](notebooks/student_performance_analysis.ipynb).

### Dataset download
```bash
pip install kaggle
# Create a token at Kaggle > Settings > API Tokens, then store it OUTSIDE the repo:
mkdir -p ~/.kaggle && echo "<YOUR_TOKEN>" > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
kaggle datasets download -d aljarah/xAPI-Edu-Data
mkdir -p dataset/raw && unzip xAPI-Edu-Data.zip -d dataset/raw/
```
Never commit tokens or `kaggle.json`.

## 6. Machine-learning methodology
**Preprocessing** (`src/preprocessing.py`): no imputation needed; exact duplicates dropped before splitting; one-hot encoding of categoricals inside a `ColumnTransformer` (`handle_unknown="ignore"`); numerics standardised **only for Logistic Regression** (tree models get raw values); the whole preprocessor lives inside a scikit-learn `Pipeline`, so it is fitted on training data/folds only (no leakage).

**Feature engineering:** none beyond encoding – no features were created or imputed.

**Training** (`src/train.py`): stratified 80/20 split (seed 42; 382 train / 96 test). Each model is tuned with 5-fold stratified `GridSearchCV` (macro-F1) on the training set. Models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost (**only if the `xgboost` package is installed – it was not installed in the environment used for the results below, so XGBoost is not in the reported comparison**).

**Selection rule (fixed before looking at results):** `0.5 × CV macro-F1 + 0.5 × CV recall of the Low class`, computed from cross-validated predictions on the training set. The test set is used once for reporting and never for selection.

## 7. Results (held-out test set, 96 students)
| Model | CV macro-F1 | CV Low recall | Selection score | Test acc | Macro P | Macro R | Macro F1 | Weighted F1 | Test Low recall | ROC-AUC (OvR macro) |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.768 | 0.810 | 0.789 | 0.729 | 0.733 | 0.745 | 0.737 | 0.727 | 0.880 | 0.895 |
| Decision Tree | 0.737 | 0.830 | 0.784 | 0.708 | 0.717 | 0.719 | 0.714 | 0.708 | 0.800 | 0.820 |
| Random Forest | 0.801 | 0.820 | 0.810 | 0.781 | 0.794 | 0.783 | 0.787 | 0.781 | 0.840 | 0.915 |
| Gradient Boosting | 0.769 | 0.810 | 0.789 | 0.760 | 0.774 | 0.766 | 0.766 | 0.759 | 0.880 | 0.891 |

**Selected: Random Forest** (best selection score; best CV macro-F1, test accuracy and ROC-AUC). Best hyper-parameters: `{'clf__class_weight': 'None', 'clf__max_depth': 'None', 'clf__min_samples_leaf': 2, 'clf__n_estimators': 200}`.

Per-class results for Random Forest (test):
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| L (Low) | 0.840 | 0.840 | 0.840 | 25 |
| M (Middle) | 0.733 | 0.786 | 0.759 | 42 |
| H (High) | 0.808 | 0.724 | 0.764 | 29 |

Confusion matrix (rows = true, columns = predicted, order L, M, H): `[[21, 4, 0], [4, 33, 5], [0, 8, 21]]`.

Top features (permutation importance, drop in macro-F1):
| Feature | Importance |
|---|---|
| StudentAbsenceDays | 0.212 |
| VisITedResources | 0.051 |
| raisedhands | 0.042 |
| ParentAnsweringSurvey | 0.029 |
| gender | 0.022 |
| Topic | 0.021 |

**Interpretation and honest caveats**
* The test set has only 96 students (25 Low), so one student moves Low recall by 4 points. Logistic Regression and Gradient Boosting reached higher Low recall on the test set (0.880 vs 0.840), but that gap is one student and the models were selected on cross-validation, not on test results.
* Tuned-model cross-validation scores are slightly optimistic because the grid search and the CV estimate use the same folds; no nested CV was run.
* The dataset is small, from one platform, and `Class` is a performance *band* label – results do not transfer automatically to another institution.
* `StudentAbsenceDays` dominates; `Relation` (Father/Mum) also associates with the label for reasons that are probably collection artefacts.
* Confusion-matrix plots: `reports/figures/`. All metrics: `reports/evaluation_results.json`.

## 8. Integration with the existing application
The model needs **all 16 features**. The existing application only stores `marks` and `attendance` per student. There is no defensible mapping from those two numbers to e.g. `raisedhands` or `StudentAbsenceDays` (the latter is a count of absence *days*, which cannot be recovered from an attendance percentage without the number of sessions), so **no values are invented**. Integration strategy:
1. The backend always computes the original rule-based score (`calculateRisk.js`).
2. `backend_integration/mlClient.js` calls the ML service **only when all 16 features are available** for a student; otherwise it returns the rule-based result with `missing_features` listed.
3. Both results are returned side by side; the rule-based score is never silently replaced.
Collecting the missing engagement fields in the app (LMS activity, parent survey, absence days) is the prerequisite for using the model on live students.

## 9. API documentation
Start: `uvicorn app.model_api:app --port 8000`
* `GET /health` → `{"status":"ok","model_loaded":true}` (503 if artefacts are missing).
* `GET /model-info` → model name, features, allowed categorical values, CV and test metrics.
* `POST /predict` – JSON body with all 16 features → `{"predicted_class","predicted_label","probabilities":{"L","M","H"},"low_performance_risk","model"}`. Missing / non-numeric / out-of-range (0-100) / unknown-category fields return **422** listing every problem.

Python usage without the API: `from src.predict import predict_one`.

## 10. Installation and execution
```bash
pip install -r requirements.txt
python -m src.train            # tune, compare, save models/ + reports/
python -m src.evaluate         # re-evaluate the saved model on the held-out split
python -m src.predict          # demo prediction
uvicorn app.model_api:app --port 8000
python -m pytest tests -q      # Python tests (API tests need fastapi + httpx)
node tests/test_mlClient.js    # Node client test (stub server)
```
Existing backend/frontend: `npm install && npm start` (see their folders).

## 11. Limitations and future work
* **API service (`app/model_api.py`) was written but not executed** in the environment used to build this component, because `fastapi`/`httpx` could not be installed there; run the API tests locally before relying on it.
* XGBoost was not evaluated (package unavailable there).
* The Node backend in the repository is incomplete (e.g. `server.js` requires `src/config/db` and `src/routes/studentRoutes`, which are missing), so the client was tested against a stub server and not against the real routes.
* The Face-recognition script contains a hard-coded MongoDB connection string. **Rotate that password and load it from an environment variable.**
* Future: collect the missing features in-app, train on institution-specific labelled outcomes, nested CV, probability calibration, fairness review (the dataset includes nationality and gender).
