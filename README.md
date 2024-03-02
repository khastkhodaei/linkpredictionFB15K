# Knowledge Graph Link Prediction

This repository contains implementations for link prediction in knowledge graphs using the TransE model.

## Description
Link prediction is a fundamental task in knowledge graph completion, where the goal is to predict missing links between entities. In this project, we employ the TransE model, a popular approach for link prediction in knowledge graphs. The TransE model learns entity and relation embeddings and measures the plausibility of a triple by calculating the distance between embeddings. During training, the model aims to minimize the distance between the embeddings of the head and tail entities given a relation.

## Course Information
This project serves as the final project for the Information Networks course taught by Dr. Moosavi at Shiraz University, completed in March 2024.

## Datasets
The project includes three datasets:
- FB15K
- FB15K237
- WN18RR

## Files
- `dataset.py`: Contains the implementation of the dataset class for loading and preprocessing data.
- `models.py`: Defines the TransE model architecture.
- `FB15Klinkpre.py`: Script for link prediction using the TransE model on the FB15K dataset.
- `FB15K237linkpre.py`: Script for link prediction using the TransE model on the FB15K237 dataset.
- `WN18RRlinkpre.py`: Script for link prediction using the TransE model on the WN18RR dataset.

## Usage
To run link prediction for each dataset:
1. Install the required dependencies: `pip install -r requirements.txt`
2. Execute the corresponding Python script for the dataset of interest.

## Results
The TransE model is trained on each dataset, and the training loss is logged for each epoch. The trained models are saved in the `saved_models` directory for future use. Additionally, the preprocessing steps, such as label encoding and data normalization, are applied to ensure the data is suitable for training.

## Conclusion
This project provides a comprehensive framework for link prediction in knowledge graphs using the TransE model. By leveraging entity and relation embeddings, the model can effectively predict missing links in knowledge graphs. The flexibility of the implementation allows for easy extension to other datasets and models for further experimentation and analysis.