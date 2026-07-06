
# Student Predictive Analytics System

### **Smart India Hackathon (SIH) 2025 Finalist**

## Executive Summary

The **Student Predictive Analytics System** is an enterprise-grade, web-based platform engineered to optimize student retention and academic outcomes. Developed for the national finals of the **Smart India Hackathon (SIH) 2025**, the system seamlessly bridges the gap between administrative data collection and proactive academic intervention.

By integrating automated automated attendance tracking via IP camera streams with predictive machine learning models, the platform equips educational institutions with real-time, actionable insights to identify at-risk students and drive data-informed decision-making.

---

## The Problem Statement

Modern educational institutions generate vast amounts of disparate student data. However, traditional systems relegate attendance tracking to a passive administrative chore rather than leveraging it as an active metric for engagement.

### **Key Challenges Addressed:**

* **Data Silos:** Attendance logs and academic records exist independently, obscuring the correlation between presence and performance.
* **Reactive Interventions:** Faculty typically identify struggling students only *after* poor examination results, missing critical opportunities for early support.
* **Administrative Overhead:** Manual attendance logging is prone to human error and consumes valuable instructional time.

---

## Core Objectives

* **Automated Data Ingestion:** Streamline attendance capture utilizing computer vision ready IP camera integrations.
* **Centralized Data Warehousing:** Maintain structured, highly relational records of student engagement and historical grades.
* **Predictive Risk Modeling:** Deploy machine learning pipelines to forecast academic outcomes and flag early warning signs.
* **Actionable Business Intelligence:** Deliver high-fidelity, interactive visualization dashboards tailored for institutional leadership and faculty.

---

## Platform Architecture & Features

### 🛠️ Core Capabilities

* **Automated Attendance Lifecycle:** End-to-end automation from video stream capture to database persistence.
* **Predictive Analytics Pipeline:** Multi-variant ML models that compute risk scores based on holistic student metrics.
* **Granular Search & Filtering:** High-performance data querying across departments, cohorts, and individual student profiles.
* **Automated Report Generation:** One-click compliance and performance reporting for internal reviews and accreditation auditing.

### 📊 Dashboard Interfaces

* **Executive View:** High-level institutional analytics, department-wise trends, and macro-level performance metrics.
* **Faculty View:** Granular roster management, real-time alert triggers for at-risk students, and individual intervention tracking.

---

## Enterprise Technology Stack

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           APPLICATION LAYERS                            │
├───────────────────┬────────────────────────┬────────────────────────────┤
│   FRONTEND        │   BACKEND & DATABASE   │   MACHINE LEARNING         │
├───────────────────┼────────────────────────┼────────────────────────────┤
│ • React.js        │ • Node.js / Express.js │ • Python                   │
│ • JavaScript (ES6)│ • MongoDB (NoSQL)      │ • Scikit-learn             │
│ • Bootstrap 5     │ • RESTful API Design   │ • Pandas / NumPy           │
└───────────────────┴────────────────────────┴────────────────────────────┘

```

* **DevOps & Tools:** Git, GitHub, Postman API Client

---

## System Architecture Workflow

```
[ IP Camera Stream ] ────> [ Automated Capture ] ────> [ MongoDB Database ]
                                                              │
[ Faculty Input ] ───────> [ Academic Records ]  ───────> [ Data Processing ]
                                                              │
[ Dashboard UI ]  <─────── [ Predictive Insights ] <───── [ ML Model Pipeline ]

```

1. **Ingestion:** Attendance data is programmatically processed via edge-ready IP stream inputs.
2. **Storage:** Structured payloads are securely routed through an Express.js backend and stored in MongoDB.
3. **Synthesis:** Python-driven data pipelines aggregate attendance trends alongside core academic variables.
4. **Inference:** Predictive engines compute performance probabilities.
5. **Visualization:** React-powered dashboards present data through optimized charts and reporting widgets.

---

## Directory Structure

```filepath
Student-Predictive-Analytics/
├── client/                  # React.js Frontend Application
│   ├── public/              # Static assets and index.html
│   ├── src/                 # Component architecture & state management
│   └── package.json
│
├── server/                  # Node.js/Express REST API
│   ├── controllers/         # Business logic & request handling
│   ├── models/              # MongoDB schemas
│   ├── routes/              # Secure API routing endpoints
│   ├── middleware/          # Auth & validation layers
│   └── server.js            # Application entry point
│
├── ml-model/                # Predictive Analytics Engine
│   ├── dataset/             # Training and validation data
│   ├── train_model.py       # Model training pipeline
│   ├── predict.py           # Inference script for production api
│   └── model.pkl            # Serialized model artifact
│
├── README.md
└── package.json

```

---

## Data Science & Predictive Modeling

The predictive core leverages multivariate algorithms to forecast student success. The models process features engineered across key performance indicators:

$$\text{Risk Score} = f(\text{Attendance Rate}, \text{Internal Assessments}, \text{Assignment Velocity}, \text{Historical GPA})$$

> **Impact Metric:** Rather than relying solely on summative final grades, the model updates continuously based on formative inputs, enabling faculty to orchestrate target interventions *weeks before* examination cycles.

---

## Roadmap & Future Enhancements

* **Advanced Biometrics:** Edge-deployed face recognition algorithms optimized for dense classroom environments.
* **Automated Communication Gateway:** Automated SMS/Email alerts via Twilio/SendGrid integration for parental notifications.
* **Cross-Platform Portability:** Native iOS and Android mobile companion applications for faculty-on-the-go.
* **Cloud Scalability:** Migration to microservices architecture utilizing AWS/Azure deployment with Docker containerization.

---

## Deployment & Installation

### Prerequisites

* Node.js (v18+)
* MongoDB (v6+)
* Python (v3.9+)

### Repository Initialization

```bash
git clone https://github.com/your-username/student-predictive-analytics.git
cd student-predictive-analytics

```

### Micro-Frontend Setup

```bash
cd client
npm install --legacy-peer-deps
npm start

```

### Micro-Backend Setup

```bash
cd ../server
npm install
npm run dev

```

### Machine Learning Environment Setup

```bash
cd ../ml-model
pip install -r requirements.txt
python train_model.py

```

---

## Project Governance & Credits

* **Status:** Prototype verified at the National Finals of Smart India Hackathon (SIH) 2025.
* **Core Contributors:** Jeffrin Jijo & Team VictoReign
* **Intellectual Property / License:** Released under the MIT Research & Educational Purpose License.
