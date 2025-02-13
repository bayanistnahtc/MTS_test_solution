# RAG Service with Query Classification

This repository contains an implementation of a Retrieval-Augmented Generation (RAG) service with an integrated query classification module. The solution is inspired by the EMNLP 2024 paper [Link](https://aclanthology.org/2024.emnlp-main.981.pdf) and demonstrates an end-to-end pipeline for determining when to use retrieval augmentation and generating responses via a RAG workflow.

## Overview

The solution consists of:
- **Query Classification Module:** Determines if a query requires retrieval augmentation using a BERT-like model.
- **Data Labeling Pipeline:** Uses LLM-generated labels on a subset of the Dolly 15K dataset to create a training set.
- **Model Training:** Fine-tunes models (e.g., `roberta-base`) using both full-model and LoRA strategies.
- **RAG Pipeline:** Integrates query classification, document retrieval (using Haystack), and response generation.
- **API Service:** A FastAPI-based REST API that provides endpoints for query processing and health checks.
- **Deployment:** Dockerized setup with docker-compose for easy deployment.

## Architecture

The RAG workflow is structured as follows:

```
               [Input Query]
                     │
             [Query Classifier]
       ┌─────────────┴─────────────┐
[LLM Generation]       [Document Retrieval]
                                │
                       [Context Augmentation]
                                │
                          [LLM Generation]
```

- **Query Classification:** Routes queries either directly to the LLM or through retrieval modules.
- **Document Retrieval:** Uses a dense retrieval to fetch relevant documents.
- **Response Generation:** Combines retrieved context with the original query for final answer generation.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/bayanistnahtc/rag-service.git
   cd rag-service
   ```

2. **Set Up Environment:**
   Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the project root with necessary settings:
   ```env
   API_HOST=0.0.0.0
   API_PORT=8000
   CLASSIFIER_PATH=path/to/your/fine-tuned-model
   DEBUG=true
   MISTRAL_API_KEY=<your-mistral-api-key>

   ```

## Running the Service

### Locally
Run the FastAPI service using Uvicorn:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Using Docker
1. **Build and Run the Container:**
   ```bash
   docker-compose up --build
   ```

2. **Health Check:**
   Visit `http://localhost:8000/health` to verify the service status.


## API Endpoints

- **GET /health:** Service health check.
- **POST /ask:** Process a query through the RAG pipeline.

## Training the Query Classifier

- **Data Labeling:** Run `labeling.py` to generate labels for the dataset.
- **Training:** Execute `training_lora.py` to fine-tune the query classification model using the labeled dataset.

## Future Improvements

- Enhance data labeling quality and expand the training dataset.
- Experiment with alternative models and fine-tuning strategies.
- Extend retrieval methods (e.g., hybrid, multi-modal) and improve vector database configuration.
- Implement unit tests and CI/CD pipelines for continuous integration.
