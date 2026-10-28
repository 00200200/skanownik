# Skanownik - Vehicle Undercarriage Scan Analysis

A prototype for analyzing X-ray scans of vehicle undercarriages. A ResNet18 classifier predicts one of two vehicle classes: passenger car (`osobowe`) or delivery van (`dostawcze`). A separate pipeline compares image patches with a reference memory bank and flags patches whose feature distance exceeds a configured threshold.

An anomaly score indicates a difference from reference patches; it does not identify contraband, explain a structural modification, or give a calibrated probability of a threat. The API currently highlights at most one patch per image. This is a demonstration project; operational screening would require separate validation on representative scans.

## Links

- [Showcase video](./showcase.mkv)
- [Presentation](./skanownik-prezentacja.pdf)

## Quick Start (Docker)

Docker Compose runs the frontend and backend for local demonstrations. The configuration uses host networking and a Vite development server.

### Prerequisites

*   **Download the [vehicle classifier .ckpt file](https://drive.google.com/file/d/1FdS9yK-cGAaH_ukQ8JSXYjrDzz8ny7E8/view?usp=sharing)** and place it at `backend/trained_models/vehicle-classifier-epoch=19-val_loss=0.19.ckpt`, the path currently used by the API. Alternatively, run `uv run src/train.py` from [backend/](./backend/) with training data in `backend/data/` (not included). Training may produce a different checkpoint filename; the API path must match it.
*   Keep the two `knn_*.pkl` files and their `config_*.json` files in `backend/trained_models_anomaly/`; the classifier checkpoint alone is not sufficient for anomaly inference.
*   **Docker**
*   **Docker Compose**

### Deployment

1.  Open a terminal in the project's root directory.
2.  Execute the following command:
    ```bash
    docker compose up --build
    ```
    > The first build downloads base images and dependencies. Runtime model initialization may also download pretrained ResNet18 weights if they are not cached.

3.  Wait for the containers to build and start successfully.

### Accessing the Application

*   **Frontend (Web Interface):** [http://localhost:5173](http://localhost:5173)
    *   Upload X-ray scans here to visualize analysis results.
*   **Backend API (Documentation):** [http://localhost:8000/docs](http://localhost:8000/docs)
    *   Interactive Swagger UI for testing API endpoints directly.

### Example file

There is a sample file that can be uploaded to the web UI in the project directory:

- [auto.czarno.bmp](./auto.czarno.bmp)

The `/analyze` endpoint processes only filenames ending in `czarno.bmp` (case-insensitive); other names are returned as skipped.

---

## Standalone Backend Setup

For development purposes or testing the backend in isolation without Docker.

### Prerequisites

*   Python 3.12 (`backend/pyproject.toml` requires `>=3.12,<3.13`)
*   `uv` (Python package manager)
*   The classifier checkpoint and anomaly files described above

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

The anomaly code also imports `scikit-learn` and `joblib`, which are not declared in `backend/pyproject.toml`. They must be available in the runtime environment; the current dependency list alone does not specify the full inference environment.

### Running the API

```bash
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

The server will be accessible at `http://localhost:8000`.

### Additional Scripts (Training & Evaluation)

The `backend/` directory contains scripts for model training and evaluation. Please refer to `backend/README.md` for detailed instructions.

The anomaly scripts build the memory bank from 90% of clean images, then choose thresholds and calculate metrics on the remaining clean images plus the anomalous images. Because threshold selection and scoring share that set, the resulting F1 is a tuning result. Assessing performance on new scans requires a separate test set with the thresholds fixed in advance.

---

## System Architecture

*   **Frontend:** React interface served by Vite in the current Docker configuration.
*   **Backend:** REST API built with FastAPI (Python).
*   **AI/ML Core:**
    *   **Vehicle Classifier:** ResNet18-based model for distinguishing between passenger cars and delivery vans.
    *   **Anomaly Scoring:** Frozen pretrained ResNet18 features and nearest-neighbor distances to a memory bank of reference patches, with a threshold selected for each vehicle class.
