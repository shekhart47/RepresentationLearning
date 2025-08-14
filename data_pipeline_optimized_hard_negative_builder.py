# python3 optimized_hard_negative_builder.py
import os
import json
import gzip
import random
import joblib

import torch
import pickle
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import faiss
import logging

# Configure logging
from typing import Dict, Union, List
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

    
class Distributed_Sentence_Embeddings:
    """ Generates distributed embedding for descriptions
        Args:
            model_path : path to the Sentence Transformer Model
            batch_size : batch size to be used for building the index of code : embedding pairs
            text_descriptions : all anchor positive negative text
        Returns:
            text_sentence_embedding_dictionary : dictionary with all sentences ( anchor + positives + negatives ) and respective embedding
    """
    
    def __init__(self, model_path : str, batch_size : int):
        """ initialize model_path and reference code_lookup dataset
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.text_sentence_embedding_dictionary : dict =  {}
        self.model = None
    def load_model(self):
        """ loads SentenceTransformer model
        """
        self.model = SentenceTransformer(model_path, trust_remote_code=True).cuda()
        return self.model
        
    def setup_multi_processing_pool(self):
        """ determines the number of GPUs available on the compute
        """
        
        print("Setting Up Multi Processing Pool")
        device_count = torch.cuda.device_count()
        pool_list = [f"cuda:{device}" for device in range(device_count)]
        return pool_list
    
    
    def generate_sentence_embedding_pair(self, text_descriptions : List[str]):
        """ Creates a dictionary of code and respective embeddings
        """
        
        print("Generating Embeddings")

        torch.cuda.empty_cache()

        model = self.load_model()

        pool_list = self.setup_multi_processing_pool()

        pool = self.model.start_multi_process_pool(pool_list)

        print("Computing embeddings using the multi-process pool")
        embeddings = self.model.encode_multi_process(sentences = text_descriptions, pool = pool, batch_size = self.batch_size)

        self.model.stop_multi_process_pool(pool)
        
        for sentence, embedding in zip(text_descriptions, embeddings):
            self.text_sentence_embedding_dictionary[sentence] = embedding

        print("Sentence : Embedding Mapping Created")
        return self.text_sentence_embedding_dictionary


class OptimizedTripletMiner:
    """
    Optimized triplet mining using vectorized operations and FAISS for similarity search
    """
    
    def __init__(self, embedding_dim: int = 1024, use_gpu: bool = True):
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.faiss_index = None
        self.embedding_matrix = None
        self.id_to_idx = {}
        self.idx_to_id = {}
        
    def build_faiss_index(self, embeddings_dict: Dict[str, np.ndarray]):
        """
        Build FAISS index for fast similarity search
        """
        logger.info("Building FAISS index...")
        
        # Convert embeddings dict to matrix
        ids = list(embeddings_dict.keys())
        embeddings = np.array([embeddings_dict[id_] for id_ in ids]).astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        self.embedding_matrix = embeddings
        self.id_to_idx = {id_: idx for idx, id_ in enumerate(ids)}
        self.idx_to_id = {idx: id_ for idx, id_ in enumerate(ids)}
        
        # Build FAISS index
        if self.use_gpu:
            # GPU version
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for normalized vectors = cosine similarity
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            # CPU version
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        
        self.faiss_index.add(embeddings)
        logger.info(f"FAISS index built with {len(embeddings)} embeddings")
    
    def get_vectorized_negatives_fast(self, query_embedding: np.ndarray, 
                                    positives: List[str], 
                                    search_pool: List[str], 
                                    num_to_select: int,
                                    similarity_threshold_percentile: float = 0.7) -> List[str]:
        """
        Fast vectorized approach for hard negative mining using FAISS
        """
        # Get indices for search pool
        search_indices = [self.id_to_idx[id_] for id_ in search_pool if id_ in self.id_to_idx]
        
        if len(search_indices) == 0:
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding.copy().astype('float32')
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search for most similar items in the search pool
        k = min(len(search_indices), num_to_select * 10)  # Get more candidates than needed
        similarities, indices = self.faiss_index.search(query_embedding, len(self.embedding_matrix))
        
        # Filter to only include items in search pool
        search_indices_set = set(search_indices)
        filtered_results = []
        
        for sim, idx in zip(similarities[0], indices[0]):
            if idx in search_indices_set:
                filtered_results.append((sim, self.idx_to_id[idx]))
            if len(filtered_results) >= k:
                break
        
        if len(filtered_results) == 0:
            return []
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(filtered_results, columns=['similarity', 'id'])
        
        # Calculate threshold based on percentile
        threshold = np.percentile(df['similarity'], similarity_threshold_percentile * 100)
        
        # Select hard negatives (high similarity but not positive)
        hard_negatives = df[df['similarity'] >= threshold]['id'].tolist()
        
        # Select remaining from lower similarity ranges
        easy_negatives = df[df['similarity'] < threshold]['id'].tolist()
        
        # Balance selection
        num_hard = min(len(hard_negatives), num_to_select // 2)
        num_easy = min(len(easy_negatives), num_to_select - num_hard)
        
        selected_negatives = []
        if num_hard > 0:
            selected_negatives.extend(random.sample(hard_negatives, num_hard))
        if num_easy > 0:
            selected_negatives.extend(random.sample(easy_negatives, num_easy))
        
        return selected_negatives

    def get_batch_negatives(self, queries_data: List[Dict], 
                          search_pool: List[str],
                          query_embeddings: Dict[str, np.ndarray],
                          num_to_select: int = 50) -> List[Dict]:
        """
        Process multiple queries in batch for better efficiency
        """
        results = []
        
        for query_data in tqdm(queries_data, desc="Processing queries"):
            query = query_data['query']
            positives = query_data['positives']
            
            if query not in query_embeddings:
                logger.warning(f"Query embedding not found for: {query}")
                continue
            
            query_embedding = query_embeddings[query]
            
            # Get candidate negatives (excluding positives)
            candidate_negatives = [id_ for id_ in search_pool if id_ not in positives]
            
            if len(candidate_negatives) < num_to_select:
                logger.warning(f"Not enough candidates for query: {query}")
                hard_negatives = candidate_negatives
            else:
                hard_negatives = self.get_vectorized_negatives_fast(
                    query_embedding, positives, candidate_negatives, num_to_select
                )
            
            results.append({
                'query': query,
                'positives': positives,
                'negatives': hard_negatives
            })
        
        return results

def construct_triplet_dataset_optimized(filtered_data_list: List[Dict], 
                                      icd_reference_lookup: Dict,
                                      text_sentence_embedding_dictionary: Dict[str, np.ndarray],
                                      query_embeddings: Dict[str, np.ndarray],
                                      top_k: int = 5,
                                      num_to_select: int = 50,
                                      use_gpu: bool = True) -> List[Dict]:
    """
    Optimized triplet dataset construction
    """
    # Initialize the miner
    miner = OptimizedTripletMiner(use_gpu=use_gpu)
    
    # Build FAISS index once
    miner.build_faiss_index(text_sentence_embedding_dictionary)
    
    # Prepare search pool (all ICD descriptions)
    search_pool = list(icd_reference_lookup.values())
    
    hard_negative_triplets = []
    
    # Process data in chunks by specialty
    for i in tqdm(range(len(filtered_data_list)), desc="Processing specialties"):
        specialty_data = filtered_data_list[i]
        specialty = list(specialty_data.keys())[0]
        
        logger.info(f'Processing Specialty: {specialty}')
        
        query_positives_dataset = specialty_data.get(specialty)
        
        # Prepare queries data for batch processing
        queries_data = []
        for query, positives in tqdm(query_positives_dataset.items(), desc=f"Preparing {specialty}"):
            positives = [description for description in positives if description != '']
            positives = positives[:top_k]
            
            if len(positives) > 0:
                queries_data.append({
                    'query': query,
                    'positives': positives
                })
        
        # Process all queries for this specialty in batch
        if queries_data:
            batch_results = miner.get_batch_negatives(
                queries_data, search_pool, query_embeddings, num_to_select
            )
            
            # Format results
            for result in batch_results:
                hard_negative_triplets.append({
                    "specialty": specialty,
                    "anchor": result['query'],
                    "positives": result['positives'],
                    "negatives": result['negatives']
                })
    
    return hard_negative_triplets

# Alternative approach using pure NumPy vectorization (if FAISS is not available)
def get_negatives_vectorized_numpy(query_embedding: np.ndarray,
                                 positives: List[str],
                                 search_pool: List[str],
                                 text_embeddings: Dict[str, np.ndarray],
                                 num_to_select: int) -> List[str]:
    """
    Vectorized negative mining using pure NumPy (fallback if FAISS unavailable)
    """
    # Filter search pool to exclude positives
    candidate_negatives = [id_ for id_ in search_pool if id_ not in positives and id_ in text_embeddings]
    
    if len(candidate_negatives) < num_to_select:
        return candidate_negatives
    
    # Stack all candidate embeddings
    candidate_embeddings = np.stack([text_embeddings[id_] for id_ in candidate_negatives])
    
    # Compute cosine similarity in batch
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    candidate_norms = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
    similarities = np.dot(candidate_norms, query_norm)
    
    # Create DataFrame for easier manipulation
    df = pd.DataFrame({
        'id': candidate_negatives,
        'similarity': similarities
    })
    
    # Sort by similarity (descending)
    df = df.sort_values('similarity', ascending=False)
    
    # Apply percentile-based selection
    similarity_70th = np.percentile(similarities, 70)
    similarity_90th = np.percentile(similarities, 90)
    
    # Hard negatives: between 70th-90th percentile
    hard_negatives = df[(df['similarity'] >= similarity_70th) & 
                       (df['similarity'] < similarity_90th)]['id'].tolist()
    
    # Easy negatives: below 70th percentile
    easy_negatives = df[df['similarity'] < similarity_70th]['id'].tolist()
    
    # Balance selection
    num_hard = min(len(hard_negatives), num_to_select // 2)
    num_easy = min(len(easy_negatives), num_to_select - num_hard)
    
    selected_negatives = []
    if num_hard > 0:
        selected_negatives.extend(np.random.choice(hard_negatives, num_hard, replace=False))
    if num_easy > 0:
        selected_negatives.extend(np.random.choice(easy_negatives, num_easy, replace=False))
    
    return selected_negatives

# Usage example:

def icd_embedding_loader():
    
    @staticmethod
    def load_pickle(filepath: str) -> Dict[str, np.ndarray]:

        filepath = Path(filepath)
        # Try compressed file first
        compressed_path = filepath.with_suffix('.pkl.gz') if filepath.suffix != '.gz' else filepath
        regular_path = filepath.with_suffix('.pkl') if filepath.suffix != '.pkl' else filepath
        
        try:
            if compressed_path.exists():
                with gzip.open(compressed_path, 'rb') as f:
                    embeddings_dict = pickle.load(f)
                logger.info(f"Loaded compressed embeddings from {compressed_path}")
                return embeddings_dict
            elif regular_path.exists():
                with open(regular_path, 'rb') as f:
                    embeddings_dict = pickle.load(f)
                logger.info(f"Loaded embeddings from {regular_path}")
                return embeddings_dict
            else:
                raise FileNotFoundError(f"No embeddings file found at {filepath}")
                
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise
    
    file_path = '../../datasets/icd_embeddings/icd_embeddings_dictionary'
    loaded_embeddings = load_pickle(file_path)
    print(f"Loaded {len(loaded_embeddings)} embeddings with pickle")
    return loaded_embeddings


def query_embedding_loader():
    
    @staticmethod
    def load_pickle(filepath: str) -> Dict[str, np.ndarray]:
        
        filepath = Path(filepath)
        
        # Try compressed file first
        compressed_path = filepath.with_suffix('.pkl.gz') if filepath.suffix != '.gz' else filepath
        regular_path = filepath.with_suffix('.pkl') if filepath.suffix != '.pkl' else filepath
        
        try:
            if compressed_path.exists():
                with gzip.open(compressed_path, 'rb') as f:
                    embeddings_dict = pickle.load(f)
                logger.info(f"Loaded compressed embeddings from {compressed_path}")
                return embeddings_dict
            elif regular_path.exists():
                with open(regular_path, 'rb') as f:
                    embeddings_dict = pickle.load(f)
                logger.info(f"Loaded embeddings from {regular_path}")
                return embeddings_dict
            else:
                raise FileNotFoundError(f"No embeddings file found at {filepath}")
                
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise
        
    file_path = '../../datasets/query_embeddings/set3/query_embeddings_dictionary_v2'
    query_embeddings = load_pickle(file_path)
    print(f"Loaded {len(query_embeddings)} embeddings with pickle")
    return query_embeddings

def load_annotated_files(file_type):
    
    
    if file_type == '.json':

        path = '../../datasets/datasets_augmented/'

        all_files = [path + file for file in os.listdir(path) if '.json' in file]
        data_list = []
        for file_path in all_files:

            with open(file_path, 'r') as file:
                data = json.load(file)
            data_list.append(data)
            
    elif file_type == '.pkl':
        
        with open('../../datasets/datasets_augmented/06122025_final_augmented_set/consolidated_data.pkl', 'rb') as file:
            data_list = pickle.load(file)
        
        
    def select_description(df):
    
        max_length_indices = df.groupby('codes')['length'].idxmax()

        return df.loc[max_length_indices].reset_index(drop = True)


    icd_dataset = pd.read_csv('../../../shekhar_tanwar/ICD-ICD-Triplet/dataset/icd10.csv').iloc[:,14:16]
    icd_dataset.columns = ['codes','description']
    icd_dataset = icd_dataset.drop_duplicates()

    icd_dataset['description'] = icd_dataset['description'].apply(lambda x : x.strip())
    icd_dataset['length'] = icd_dataset['description'].apply(lambda x : len(x))

    icd_dataset_filtered = select_description(icd_dataset)

    icd_reference_lookup = {}
    for row in icd_dataset.itertuples():
        icd_reference_lookup[row.codes.strip()] = row.description.strip()

    return data_list,  icd_reference_lookup

def filter_dataset(data_list):
    filtered_data_list = []
    for i in range(len(data_list)):
        data_specialty = data_list[i]

        specialty = list(data_specialty.keys())[0]
        query_codes = data_specialty.get(specialty)
        query_code_filtered = {}


        for query, code_descriptions in query_codes.items():
            if len(code_descriptions) != 1:
                code_descriptions_filtered = [code for code in code_descriptions if code != ''] 
                query_code_filtered[query] = code_descriptions_filtered
            else:
                continue

        data_specialty_filtered = {}
        data_specialty_filtered[specialty] = query_code_filtered
        filtered_data_list.append(data_specialty_filtered)
        
    return filtered_data_list


if __name__ == "__main__":
    """
    
    The script loads the prepared dataset in format { specialty : {query : [icd description1, icd description2,...]} , ... } 
    
    It loads the embeddings of the ICD Descriptions
    
    And loads the embeddings of the queries used in the prepared dataset
    The query embeddings have to be re-computed everytime new queries are added.
    
    It then filters the dataset to remove queries with no codes, and then builds the triplet dataset
    
    """

    
    # Load your data
    file_type = '.pkl'
    # loads the specialty, query, icd description dataset and the icd reference lookup
    data_list, icd_reference_lookup = load_annotated_files(file_type)
    # loads the icd description embeddings
    text_sentence_embedding_dictionary = icd_embedding_loader()
    # loads the query embeddings ( queries in the datatset ) 
    query_embeddings = query_embedding_loader()

    # Filter dataset
    filtered_data_list = filter_dataset(data_list)

    # Build optimized triplet dataset
    triplet_dataset = construct_triplet_dataset_optimized(
        filtered_data_list=filtered_data_list,
        icd_reference_lookup=icd_reference_lookup,
        text_sentence_embedding_dictionary=text_sentence_embedding_dictionary,
        query_embeddings=query_embeddings,
        top_k=10,
        num_to_select=50,
        use_gpu=False  # Set to False if no GPU available
    )
    
    path = '../../datasets/triplets_v45_240_queries_10positives_50hn/icd_icd_hard_negative_triplets_p5_hn50_06192025.jsonl'
    with open(path, 'w') as outfile:
        for entry in  triplet_dataset:
            json.dump(entry, outfile)
            outfile.write('\n')
    
    
    print(f"Generated {len(triplet_dataset)} triplets")

