# Use the following command to run the script
# N is the number of GPUs on your VM
# torchrun --nproc_per_node=4 script3_hyperparameter_search_optuna-v3_Contrastive_loss.py



# NOTE
# first run script_1_building_hyperparameter_tuning_sampled_dataset
# then run this script

import math
import torch
import json
import time
import torch
import pickle  
import optuna
import random
import logging
import numpy as np
import transformers
import pandas as pd
from tqdm import tqdm
from torch import optim
from datasets import Dataset
from datetime import datetime
from collections import Counter
from datasets import load_from_disk
from get_logging import logger_object
from types import SimpleNamespace
from transformers import AutoConfig
from torch.utils.data import DataLoader
from collections import defaultdict
from datasets import Dataset, DatasetDict
from transformers import EarlyStoppingCallback
from transformers import AdamW, get_scheduler
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction, BinaryClassificationEvaluator, SentenceEvaluator
from sentence_transformers.losses import TripletLoss, MultipleNegativesRankingLoss, GISTEmbedLoss, ContrastiveLoss
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, util, SentenceTransformerTrainingArguments, SentenceTransformerTrainer, InputExample

logger = logger_object()


# 1 - Load the Dataset
def load_dataset_locally():

    logger.debug('loading from disk')
    train_dataset = pd.read_csv('../src/datasetbuilder/code-final-icd-cpt-dataset-builder/prepared_datasets/train_dataset_hyperparameter_constrastive_loss.csv').iloc[:,1:]
    eval_dataset = pd.read_csv('../src/datasetbuilder/code-final-icd-cpt-dataset-builder/prepared_datasets/eval_dataset_hyperparameter_constrastive_loss.csv').iloc[:,1:]
    
    train_dataset = [
        
        InputExample(
        texts=[row['sentence1'], row['sentence2']],
            label=float(row['label'])
        ) for idx,row in train_dataset.iterrows()
        
    ]

    return train_dataset, eval_dataset


# 2 - Prepare the validation data for Recall@3 evaluation

def get_relevant_documents(eval_dataset):
    logger.debug('Generating Relevant Documents Per Query')
    
    # assign unique ids to queries and documents
    queries = {}
    corpus = {}
    relevant_docs = defaultdict(set)

    query_id_map = {}
    doc_id_map = {}
    query_id_counter = 0
    doc_id_counter = 0

    for idx, row in eval_dataset.iterrows():
        sentence1 = row['sentence1']
        sentence2 = row['sentence2']
        label = int(row['label'])

        # Assign query IDs
        if sentence1 not in query_id_map:
            query_id = f'q{query_id_counter}'
            query_id_map[sentence1] = query_id
            queries[query_id] = sentence1
            query_id_counter += 1
        else:
            query_id = query_id_map[sentence1]

        # Assign document IDs
        if sentence2 not in doc_id_map:
            doc_id = f'd{doc_id_counter}'
            doc_id_map[sentence2] = doc_id
            corpus[doc_id] = sentence2
            doc_id_counter += 1
        else:
            doc_id = doc_id_map[sentence2]

        # Map relevant documents
        if label == 1:
            relevant_docs[query_id].add(doc_id)

    # Convert relevant_docs to a regular dict
    relevant_docs = dict(relevant_docs)
    
    return queries, corpus, relevant_docs



# 3 - Implement Custom Recall@3 Evaluator

class RecallAtKEvaluator(SentenceEvaluator):
    """
    Custom evaluator that computes mean Recall@K.
    """
    def __init__(self, queries, corpus, relevant_docs, k=3, name=''):
        self.queries = queries
        self.corpus = corpus
        self.relevant_docs = relevant_docs
        self.k = k
        self.name = name

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        # Compute embeddings for queries and corpus
        query_ids = list(self.queries.keys())
        corpus_ids = list(self.corpus.keys())
        query_texts = [self.queries[qid] for qid in query_ids]
        corpus_texts = [self.corpus[cid] for cid in corpus_ids]
        
        query_embeddings = model.encode(query_texts, convert_to_tensor=True, show_progress_bar=False)
        corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=False)
        
        # Compute similarity scores
        cos_scores = torch.matmul(query_embeddings, corpus_embeddings.T)
        
        # For each query, compute Recall@K
        recall_scores = []
        for idx, query_id in enumerate(query_ids):
            relevant = self.relevant_docs.get(query_id, set())
            scores = cos_scores[idx]
            top_k = torch.topk(scores, k=self.k, largest=True)
            top_k_indices = top_k.indices.cpu().numpy()
            retrieved_doc_ids = [corpus_ids[i] for i in top_k_indices]
            num_relevant_retrieved = len(set(retrieved_doc_ids).intersection(relevant))
            num_relevant = len(relevant)
            if num_relevant > 0:
                recall = num_relevant_retrieved / num_relevant
                recall_scores.append(recall)
            else:
                continue  # Skip if no relevant documents
        
        mean_recall = np.mean(recall_scores) if recall_scores else 0.0
        return mean_recall


# 2 - Define the Evaluator

def get_dev_evaluator(eval_dataset):
    
    dev_evaluator_name = "finetuned_pubmed_bert_dev"
    
    # dev_evaluator = TripletEvaluator(
    #                 anchors=eval_dataset["anchor"],
    #                 positives=eval_dataset["positive"],
    #                 negatives=eval_dataset["negative"],
    #                 name=dev_evaluator_name,)
    
    queries, corpus, relevant_docs = get_relevant_documents(eval_dataset)
    

    dev_evaluator = RecallAtKEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
         k=3,
        name="recall_at_3_evaluator",
    )
    
    
    return dev_evaluator


# 3 - Define the Objective Function For Optuna

train_dataset, eval_dataset = load_dataset_locally()
dev_evaluator = get_dev_evaluator(eval_dataset)

def objective(trial):
    # Suggest Hyperparameters To Optimize
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    num_epochs = trial.suggest_int('num_epochs',1,10)
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    margin = trial.suggest_float('margin',0.2,1.0)
    
    # Create a data loader with the suggested batch size
    train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size = batch_size)
    
    # initialize the sentence transformer model
    model_name = "../model/NeuML_pubmedbert-base-embeddings"
    model = SentenceTransformer(model_name)
    
    # define the loss function with the suggested margin
    train_loss = losses.ContrastiveLoss(model = model, margin = margin)
    
    # train the model with the suggested hyperparameters
    model.fit(
        train_objectives = [(train_dataloader, train_loss)],
        evaluator = dev_evaluator,
        epochs = num_epochs,
        optimizer_params = {'lr' : learning_rate},
        show_progress_bar = False,
        evaluation_steps = 1000,
        output_path = None
    
    )
    
    # evaluate the model on the validation set
    recall_at_3 = dev_evaluator(model)
    return recall_at_3

# 4 - Run the Hyperparameter Optimization

# create a study object with direction = 'maximize' since we want to maximize Recall@3
study = optuna.create_study(direction = 'maximize')
# Start the optimization process
study.optimize(objective, n_trials = 10)

# 5 - Retrieve and Print the Best Hyperparameters
logger.debug('Best Trial : ')
trial = study.best_trial
logger.debug(f'Recall@3 : {trial.value}')
logger.debug(f'Best Hyperparameters:')
for key,value in trial.params.items():
    logger.debug(f' {key}:{value}')