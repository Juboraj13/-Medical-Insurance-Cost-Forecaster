# -Medical-Insurance-Cost-Forecaster


Natural Language Processing Toxicity Inference Engine

This project implements a production-ready text classification pipeline designed to analyze online comments and predict whether a text payload is safe or toxic. The system utilizes a machine learning architecture trained on real-world community commentary data and exposes its prediction capabilities via a web API interface.

Project Structure

notebook.ipynb: The primary Jupyter Notebook containing data preprocessing, exploratory data analysis, and the model optimization pipeline. Running this notebook trains the model and generates the necessary serialization assets.
app.py: A modular FastAPI script that loads the serialized model assets and establishes the API routing infrastructure.
requirements.txt: The text file specifying the exact system dependencies and library versions required for replication.

Note: The models/ directory and its underlying tracking files (complex_model.pkl and tfidf_vectorizer.pkl) are generated automatically on your local machine after executing the training notebook.

System Installation and Local Setup

To deploy the text classification infrastructure locally on your machine, follow these execution steps:

1. Project Directory Preparation
Clone or download the repository files (app.py, requirements.txt, and notebook.ipynb) into a dedicated workspace folder on your local file system.

2. Environment Dependency Installation
Open your operating system terminal, navigate directly into your project root directory, and run the following command to install the required Python libraries:

pip install -r requirements.txt
