# Skanownik - Vehicle Undercarriage Anomaly Detection System

An advanced AI-powered system designed for analyzing X-ray scans of vehicle undercarriages. The system automatically classifies vehicle types (passenger vs. delivery vans) and detects potential anomalies (e.g., contraband, structural modifications) using state-of-the-art Computer Vision algorithms.

## 🚀 Quick Start (Docker) - Recommended for Demonstrations

The most efficient way to deploy the full stack (Frontend + Backend) without manual environment configuration.

### Prerequisites

*   **Download the [model .cpkt file](https://drive.google.com/file/d/1FdS9yK-cGAaH_ukQ8JSXYjrDzz8ny7E8/view?usp=sharing) and place it in [backend/trained_models/](./backend/trained_models) OR execute `uv run src/train.py` from the [backend/](./backend/) directory** to generate the model file
*   **Docker**
*   **Docker Compose**

### Deployment

1.  Open a terminal in the project's root directory.
2.  Execute the following command:
    ```bash
    docker compose up --build
    ```
    > **⚠️ Important Note:** The initial setup process may take a significant amount of time (several minutes) as it requires downloading base images and installing all system dependencies. Please be patient.

3.  Wait for the containers to build and start successfully.

### Accessing the Application

*   **Frontend (Web Interface):** [http://localhost:5173](http://localhost:5173)
    *   Upload X-ray scans here to visualize analysis results.
*   **Backend API (Documentation):** [http://localhost:8000/docs](http://localhost:8000/docs)
    *   Interactive Swagger UI for testing API endpoints directly.

### Example file

There is a sample file that can be uploaded to the web UI in the project directory:

- [auto.czarno.bmp](./auto.czarno.bmp)

---

## 🐍 Standalone Backend Setup

For development purposes or testing the backend in isolation without Docker.

### Prerequisites

*   Python 3.12+
*   `uv` (Python package manager)

### Installation

1.  Install `uv` (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  Navigate to the backend directory:
    ```bash
    cd backend
    ```
3.  Install dependencies:
    ```bash
    uv sync
    ```

### Running the API

```bash
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

The server will be accessible at `http://localhost:8000`.

### Additional Scripts (Training & Evaluation)

The `backend/` directory contains scripts for model training and evaluation. Please refer to `backend/README.md` for detailed instructions.

---

## 🏗️ System Architecture

*   **Frontend:** Minimalist web interface (HTML/JS) served via Nginx.
*   **Backend:** High-performance REST API built with FastAPI (Python).
*   **AI/ML Core:**
    *   **Vehicle Classifier:** ResNet18-based model for distinguishing between passenger cars and delivery vans.
    *   **Anomaly Detector:** Feature extraction pipeline (ResNet18) combined with a Memory Bank (KNN) approach to identify deviations from the norm.
