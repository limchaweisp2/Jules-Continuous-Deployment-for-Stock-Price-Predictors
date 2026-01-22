# My CICD Project

This project demonstrates a fully automated machine learning training pipeline using DVC for pipeline management and CML for reporting metrics and plots to GitHub Pull Requests.

## Repository Structure

- `train.py`: Python script to generate synthetic data and train a RandomForest model.
- `dvc.yaml`: DVC pipeline configuration.
- `.github/workflows/cml.yaml`: GitHub Actions workflow for CI/CD and CML reporting.
- `requirements.txt`: Python dependencies.
- `data_baseline.csv`: Normal synthetic training data.
- `data_drift.csv`: Shifted synthetic data for testing model robustness.

## Testing Data Sets

To help you test your MLOps pipeline, I have provided two sets of data:

1.  **`data_baseline.csv` (The Control Group):**
    - **Purpose:** This file contains "normal" data that follows the patterns the model is expected to learn.
    - **Use Case:** Use this to train your initial model and establish a "baseline" performance metric (Accuracy, F1, etc.). This is your "Gold Standard" for comparison.

2.  **`data_drift.csv` (The Stress Test):**
    - **Purpose:** This file contains data where the underlying distribution has shifted (features have different means or ranges).
    - **Use Case:** Use this to simulate "Data Drift" (e.g., changing market conditions or sensor failure). When you run the training script with this file (`python train.py --data data_drift.csv`), you will see the metrics drop. This demonstrates why automated deployment can be risky—a model trained on one set of conditions may fail when the data "drifts" in the real world.

## Local Setup and Initialization

To initialize this repository locally, follow these steps:

1. **Initialize Git repository:**
   ```bash
   git init
   ```

2. **Initialize DVC:**
   ```bash
   dvc init
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add files to Git:**
   ```bash
   git add .
   git commit -m "Initial commit: Set up DVC pipeline and CML workflow"
   ```

5. **Run the pipeline locally (optional):**
   ```bash
   dvc repro
   ```

## Triggering the CI/CD Pipeline

1. **Create a new branch:**
   ```bash
   git checkout -b feature/experiment
   ```

2. **Make changes to `train.py` or other files.**

3. **Commit and push to GitHub:**
   ```bash
   git add .
   git commit -m "Update training logic"
   git push origin feature/experiment
   ```

4. **Open a Pull Request** on GitHub. The GitHub Action will trigger, run the DVC pipeline, and CML will post a comment with the metrics and the confusion matrix plot directly on your PR.
