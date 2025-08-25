import pandas as pd
from datasets import Dataset
from datasets import Dataset, DatasetDict


def load_datasets(train_dataset_path, eval_dataset_path):
    
    train_df = pd.read_csv(train_dataset_path).iloc[:,1:]
    eval_df = pd.read_csv(eval_dataset_path).iloc[:,1:]

    return train_df,eval_df
    
def generate_samples_train(data, fraction):
    result = data['anchor'].value_counts().reset_index()
    df_sample = data.groupby('anchor', group_keys=False).apply(lambda x: x.sample(frac=fraction))
    assert df_sample['anchor'].nunique() == result.shape[0]
    return df_sample

# 1 - Load the Dataset
def load_dataset_pandas(TRAIN_DATASET_PATH, EVAL_DATASET_PATH) -> pd.DataFrame:
    
    train_data = pd.read_csv(TRAIN_DATASET_PATH).iloc[:,1:]
    eval_data = pd.read_csv(EVAL_DATASET_PATH).iloc[:,1:]
    # random shuffle
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    eval_data = eval_data.sample(frac=1).reset_index(drop=True)

    return train_data, eval_data

def build_huggingface_dataset(TRAIN_DATASET_PATH, EVAL_DATASET_PATH):
    
    train_data, eval_data = load_dataset_pandas(TRAIN_DATASET_PATH, EVAL_DATASET_PATH)
    train_data = Dataset.from_pandas(pd.DataFrame(data=train_data))
    eval_data = Dataset.from_pandas(pd.DataFrame(data=eval_data))
    ds = DatasetDict()
    ds['train'] = train_data
    ds['validation'] = eval_data
    
    return ds


if __name__ == "__main__":
    
    # these are the original training and eval dataset paths ( hard negative mined datasets )
    train_dataset_path = '../src/datasetbuilder/code-final-icd-cpt-dataset-builder/prepared_datasets/train_dataset_hyperparameter_constrastive_loss.csv'
    eval_dataset_path = '../src/datasetbuilder/code-final-icd-cpt-dataset-builder/prepared_datasets/eval_dataset_hyperparameter_constrastive_loss.csv'
    #test_dataset_path = '../src/datasetbuilder/code-final-icd-cpt-dataset-builder/test_set_icd_cpt_hn_50.csv'

    # this is the path where we want to save the subsampled datasets of train and eval
    SAMPLE_TRAIN_DATASET_PATH = '../dataset/hyperparameter_subsample/hyperparameter_samples_training.csv' # change the dataset type here
    SAMPLE_EVAL_DATASET_PATH = '../dataset/hyperparameter_subsample/hyperparameter_samples_eval.csv'
    
    # we load the dataset
    train_df,eval_df = load_datasets(train_dataset_path, eval_dataset_path)

    # this is the fraction by which we want to sample our data. note that we have to tune the samples
    # to make sure that we have the same number of anchors randomly selected as the original train and eval dataset
    # the samples are selcted through stratificaiton
    fraction = 0.011
    df_sample_train = generate_samples_train(train_df, fraction)
    df_sample_eval = generate_samples_train(eval_df, fraction)
    df_sample_train = df_sample_train.reset_index().iloc[:,1:]
    df_sample_eval = df_sample_eval.reset_index().iloc[:,1:]
    
    # we save the sampled train and eval datasets
    df_sample_train.to_csv(SAMPLE_TRAIN_DATASET_PATH)
    df_sample_eval.to_csv(SAMPLE_EVAL_DATASET_PATH)
    
    # we convert the sampled train and eval as Huggingface datasets
    ds = build_huggingface_dataset(SAMPLE_TRAIN_DATASET_PATH, SAMPLE_EVAL_DATASET_PATH)
    
    # we save the datasets to disk
    ds.save_to_disk("../dataset/huggingface/hf_10242024")
