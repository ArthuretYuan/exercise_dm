# Contextual Semantic Similarity
This repository includes training and evaluation scripts for models to determine the semantic similarity between key expressions in technical documents while considering their contextual domain. The primary goal is to facilitate efficient document retrieval by recognizing equivalent or related expressions based on their domain-specific meanings.

To tackle this challenge, four different model architectures were explored to integrate the context information to the model’s input.

## Folder Structure
```
exercise_dm/
│── models/ # Directory for storing trained models (ignored)
│── data/ # Directory for dataset (ignored)
│── src/
│   │── exercise_dm
│       │── core
│       │   │── training
│       │   │   │── train_concat_bert.py
│       │   │   │── train_gpt2.py
│       │   │   │── train_sbert_ce.py
│       │   │   │── train_sbert_st.py
│       │   │── evaluation
│       │       │── evaluate_concat_bert.py
│       │       │── evaluate_gpt2.py
│       │       │── evaluate_sbert_ce.py
│       │       │── evaluate_sbert_st.py
│       │── jobs
│       │   │── build_api.py
│       │   │── infer_concat_bert.py
│       │   │── infer_sbert_ce.py
│       │   │── infer_sbert_st.py
│       │   │── infer_gpt2.py
│       │   │── load_models.py
│       │── utils
│       │   │── map_context_code.py
│       │   │── measure_false_degree.py
│       │   │── quantize_similarity_score.py
│       │   │── split_test_data.py
│       │── settings.py
│── .gitignore
│── README.md
│── pyproject.toml
│── install.sh
```

## Model Approaches
- Sentence BERT - Sentence Transformer
- Sentence BERT - Cross Encoder
- Concatenated BERT
- GPT2

For a comprehensive explanation of the methodology and performance analysis, please visit the following link.

Report link: https://drive.google.com/file/d/1zSsjEYS-GWJgynUFXWAKRfjRhYx-GKBw/view?usp=sharing

This README file provides instructions on using the model for inference via an API.

# Similarity Score API

## Overview
This API provides similarity scoring between two text inputs within the specific context. There are 4 optional trained machine learning models as mentioned above. It exposes a FastAPI-based endpoint where users can send text pairs and context, then receive similarity predictions.

## Requirements
- python 3.11+
- pyTorch
- transformers
- fastAPI
- uvicorn
- sentence-transformers
- scikit-learn
- datasets

The version of the dependencies are specified in pyproject.toml file.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/similarity-service.git
    cd similarity-service
    ```
2. Install dependencies:
    ```bash
    pip install setuptools
    poetry install
    ```

## Download models

***IMPORTANT!*** 

This service requires a model to function. Please download the model folder before using the API.

Model download link: https://drive.google.com/drive/folders/10XjJU_eIH92HFXwczLfyNnOfPUjIQwC_?usp=sharing

## Runing the API
Start the FastAPI server with Uvicorn:
```bash
uvicorn exercise_dm.jobs.build_api:app --host 0.0.0.0 --port 8000
```

## API Usage
### Endpoint: [/predict](http://localhost:8000/predict)
### Method: POST
### Request format
```json
{
	"model": "concat_bert",
    "context_mode": "concat_bert",
    "text1": "cutting tools",
    "text2": "laser",
    "context": "C09",
    "true_similarity": "0.5"
}
```

#### model options
- 'sbert_st': sentence bert - sentence transformer
- 'sbert_ce': sentence bert - cross encoder
- 'concat_bert': concatenated bert
- 'gpt2': gpt2 model

#### context_mode options
- 'code': When using context code, plesse enter a code (e.g., A01) to the "context" field
- 'text': When using context text, please provide a meaningful description (e.g., for code ‘A01’, the corresponding text is ‘AGRICULTURE; FORESTRY; ANIMAL HUSBANDRY; HUNTING; TRAPPING; FISHING’, case insensitive).

### Response format
```json
{
    "predicted_similarity": [
        0.3867394030094147
    ]
}
```

## Contributors
- Yaxiong YUAN