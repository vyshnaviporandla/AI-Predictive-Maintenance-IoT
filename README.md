# AI-Powered Predictive Maintenance System for IoT Devices

An experimental machine-learning dashboard that analyzes synthetic industrial IoT sensor data to demonstrate predictive maintenance and equipment-failure risk assessment.

## Live Demo

[Launch the Streamlit App](https://ai-predictive-maintenance-iot-bcmdjxg9bpmwfzup96dv5s.streamlit.app/)

## GitHub Repository

[View Source Code](https://github.com/vyshnaviporandla/AI-Predictive-Maintenance-IoT)

## Technologies Used

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn

## Key Features

- Interactive predictive-maintenance dashboard
- Sensor-data analysis and feature engineering
- Machine-learning-based failure-risk assessment
- Data visualization and model evaluation
- Synthetic industrial IoT dataset

## Developer

Vyshnavi Porandla

---

## 📌 1. Project Overview

Predictive maintenance uses equipment data and analytical models to estimate the likelihood of machine failure and help maintenance teams investigate potential problems before an unexpected breakdown.

This project demonstrates a software-only predictive maintenance workflow without requiring physical IoT devices or industrial hardware.

The application uses simulated sensor readings, prepares data for machine learning, trains a Random Forest classifier, and presents predictions and model information through a Streamlit dashboard.

### Problem Statement

Unexpected equipment failures can interrupt production, increase maintenance costs, and reduce operational efficiency. Traditional maintenance approaches may rely on fixed schedules or action after a failure occurs.

This project demonstrates how sensor data and machine learning can support an earlier assessment of equipment failure risk.

### Objectives

- Simulate industrial IoT sensor measurements.
- Explore and preprocess sensor data.
- Engineer equipment-health indicators.
- Train and evaluate a classification model.
- Estimate machine failure risk from sensor readings.
- Display predictions and maintenance alerts.
- Visualize model metrics and feature importance.
- Document the complete workflow for learning and portfolio demonstration.

---

## ✨ 2. Key Features

- Synthetic industrial sensor data generation
- Ten equipment and sensor input features
- Four engineered machine-health features
- Random Forest classification with class balancing
- Stratified training and testing workflow
- Accuracy, precision, recall, and F1-score reporting
- Interactive Streamlit dashboard
- Equipment failure-risk assessment
- Probability visualization and feature-importance charts
- Threshold-based maintenance alerts
- Dataset exploration and outcome filtering
- Command-line training and prediction workflow
- Saved model metrics and prediction reports
- Optional local Ollama maintenance explanations
- Docker configuration and Jupyter notebook support

Feature availability and generated outputs depend on the selected workflow and local configuration.

---

## 🏭 3. Industry Applications

Predictive maintenance concepts are relevant to:

- **Manufacturing:** Monitoring equipment operating conditions.
- **Factories:** Identifying unusual machine behavior for investigation.
- **Power generation:** Studying equipment operating trends.
- **Automotive:** Analyzing motor and component sensor measurements.
- **Aviation:** Supporting data-driven maintenance planning.

These are potential application areas. This educational prototype has not been validated on real industrial equipment.

---

## 📊 4. Sensor Data

The simulated dataset contains the following input features:

| Feature | Description |
|---|---|
| Temperature | Equipment operating temperature |
| Vibration | Machine vibration level |
| Pressure | Operating pressure |
| RPM | Rotational speed |
| Current | Electrical current |
| Humidity | Ambient humidity |
| Coolant temperature | Cooling-system temperature |
| Power draw | Equipment power consumption |
| Device age | Accumulated operating hours |
| Maintenance gap | Days since previous maintenance |

The preprocessing workflow also derives four indicators:

- Temperature-to-pressure ratio
- Vibration-load index
- RPM-to-current ratio
- Combined stress index

These values are simulated for experimentation, not measurements collected from actual machines.

---

## 🧠 5. Machine Learning Workflow

The project uses a scikit-learn pipeline with a StandardScaler and a class-balanced Random Forest classifier.

### Workflow

1. Generate or load synthetic sensor data.
2. Clean and prepare the dataset.
3. Prepare the failure labels.
4. Engineer equipment-health features.
5. Split the dataset into training and testing subsets.
6. Train the Random Forest classifier.
7. Evaluate its predictions.
8. Save the model and evaluation artifacts.
9. Enter sensor readings in the dashboard.
10. Display the predicted class, failure probability, and corresponding alert.

### Evaluation Metrics

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted failures that are actual failures in the evaluation dataset.
- **Recall:** Proportion of actual failures detected by the model.
- **F1-score:** Balance between precision and recall.

Metrics calculated on synthetic data do not establish real-world predictive performance.

---

## 🏗️ 6. System Architecture

```text
Synthetic IoT Sensor Data
          |
          v
Data Loading and Cleaning
          |
          v
Feature Engineering
          |
          v
Training and Testing Split
          |
          v
Random Forest Classifier
          |
          v
Model Evaluation and Persistence
          |
          v
Streamlit Dashboard
          |
          +--> Sensor Data Assessment
          |
          +--> Failure Probability
          |
          +--> Maintenance Alerts
          |
          +--> Model Visualization
          |
          +--> Optional Local Ollama Explanation
```

---

## 🛠️ 7. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Matplotlib | Charts and visualization |
| Seaborn | Statistical visualization |
| Streamlit | Interactive dashboard |
| Joblib | Model serialization |
| Jupyter Notebook | Data exploration |
| Git and GitHub | Version control and project publishing |
| Docker | Optional containerization |
| Ollama | Optional local language-model explanations |

---

## 📁 8. Project Structure

```text
AI-Predictive-Maintenance-IoT/
├── .dockerignore
├── .gitignore
├── .streamlit/
├── Dockerfile
├── docker-compose.yml
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── data/
├── docs/
├── images/
├── models/
├── notebooks/
├── outputs/
├── screenshots/
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── llm_assistant.py
│   ├── model_building.py
│   ├── prediction.py
│   ├── preprocessing.py
│   └── visualization.py
└── tests/
```

- `app.py` — Streamlit dashboard.
- `main.py` — Command-line workflow.
- `src/` — Data processing, model training, predictions, and supporting logic.
- `data/` — Synthetic and processed datasets.
- `models/` — Model and evaluation artifacts.
- `images/` — Generated charts.
- `screenshots/` — Project demonstration evidence.
- `notebooks/` — Exploratory analysis.
- `outputs/` — Prediction reports and summaries.
- `docs/` — Architecture and project documentation.
- `tests/` — Automated tests.

Some artifacts are generated when the application or training workflow runs.

---

## 💻 9. Installation and Execution

### Prerequisites

- Python 3.13
- pip
- Git (for repository management)

### Windows Setup

Open Command Prompt in the project directory.

Create a virtual environment if one does not already exist:

```bat
py -m venv venv
```

Activate it:

```bat
venv\Scripts\activate
```

Install dependencies:

```bat
python -m pip install -r requirements.txt
```

If the `python` command is unavailable, use:

```bat
py -m pip install -r requirements.txt
```

### Run the Dashboard

```bat
streamlit run app.py
```

Open the following address in your browser:

http://localhost:8501

### Run the Command-Line Workflow

```bat
python main.py
```

This runs the workflow implemented by the project's command-line entry point.

### Run the Tests

```bat
python -m unittest discover -s tests -v
```

### Optional Docker Execution

```bash
docker compose up --build -d
```

Docker and the optional Ollama assistant are not required for the basic local dashboard.

---

## 📈 10. Results and Outputs

The project is designed to produce or display:

- Model evaluation metrics
- A confusion matrix
- Feature-importance visualization
- Failure-probability visualization
- Processed sensor data
- Predicted machine condition
- Maintenance alert information
- A saved prediction report
- A maintenance summary

The original repository includes a recorded synthetic-data evaluation snapshot:

| Metric | Previously Recorded Value |
|---|---:|
| Accuracy | 89.50% |
| Precision | 72.44% |
| Recall | 76.67% |
| F1-score | 74.49% |

---

## 📸 11. Screenshots and Demonstration

Save your own screenshots under `screenshots/`.

Suggested filenames:

```text
screenshots/
├── 01_dashboard_overview.png
├── 02_model_metrics.png
├── 03_sensor_assessment.png
├── 04_failure_prediction.png
├── 05_dataset_explorer.png
└── 06_github_repository.png
```

Capture screenshots from the application and workflows you actually run. Do not use placeholder images as evidence of successful execution.

To display a screenshot in this README after saving it, use:

```markdown
![Dashboard Overview](screenshots/01_dashboard_overview.png)
```

---

## ⚠️ 12. Limitations

- The dataset uses synthetic sensor readings.
- The model has not been validated using real industrial telemetry.
- No physical sensors or machines are connected.
- No live IoT data ingestion is implemented in this software-only demonstration.
- Synthetic-data metrics do not guarantee performance on actual equipment.
- Predictions and maintenance recommendations are educational outputs, not safety decisions.
- The optional Ollama assistant requires a compatible local service and model.
- Production deployment would require additional security, monitoring, validation, and operational controls.

---

## 🚀 13. Future Enhancements

- Integrate real or publicly available industrial datasets.
- Add a simulated real-time sensor stream.
- Compare Random Forest with Logistic Regression and other classifiers.
- Explore time-series models for sequential sensor readings.
- Store prediction history in a database.
- Add configurable alert thresholds and downloadable reports.
- Implement authenticated access and deployment monitoring.
- Validate model performance on representative real-world data.

---

## 🎓 14. Learning Outcomes

This project provides practical exposure to:

- Python application development
- Data cleaning and feature engineering
- Supervised machine learning
- Classification metrics and model evaluation
- Probability-based risk communication
- Interactive data visualization
- Modular project organization
- Model persistence and testing
- GitHub version control and technical documentation

---

## 👩‍💻 15. Developer

**Vyshnavi Porandla**

B.Tech Computer Science and Engineering student

Interested in Python development, machine learning, data science, and practical software engineering.

This project is part of my learning and portfolio development journey.

---

## Conclusion

The AI-Powered Predictive Maintenance System demonstrates an end-to-end educational workflow combining synthetic IoT data, feature engineering, machine learning, failure-risk assessment, and interactive visualization.

It provides a foundation for further experimentation with industrial analytics and predictive maintenance.

**Project classification:** Software prototype | Synthetic data | Machine learning | Streamlit dashboard | No physical hardware validation
