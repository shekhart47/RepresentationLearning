# example 
# python3 stage3_step2_icd_verification_gpt4o_gpt41-v2.py

import os
import json
import time
import openai
import pickle
import warnings
import requests
import pandas as pd
from tqdm import tqdm
from typing import List
from collections import defaultdict
from azureml.core import Workspace
from azure.identity import DefaultAzureCredential
from azureml.core.authentication import ServicePrincipalAuthentication

from langchain.chains import LLMChain
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import  PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts.few_shot import FewShotPromptTemplate

from langchain.output_parsers import PydanticOutputParser, JsonOutputKeyToolsParser, CommaSeparatedListOutputParser
pd.set_option('display.max_rows', None)
warnings.filterwarnings("ignore")
ws = Workspace.from_config()

class BearerAuth(requests.auth.AuthBase):
    def __init__(self, token):
        self.token = token
    def __call__(self, r):
        r.headers["authorization"] = "Bearer " + self.token
        return r

def initialize_llm(model_name) -> AzureChatOpenAI:
    ws = Workspace.from_config()
    keyvault = ws.get_default_keyvault()
    credential = DefaultAzureCredential()
    workspacename = keyvault.get_secret("project-workspace-name")
    access_token = credential.get_token("https://cognitiveservices.azure.com/.default")
    os.environ["AZURE_OPENAI_KEY"] = access_token.token
    openai.api_type = "azure_ad"
    os.environ["AZURE_OPENAI_ENDPOINT"] = f"https://{workspacename}openai.openai.azure.com/"
    openai.api_version = "2023-07-01-preview"
    subscriptionId = keyvault.get_secret("project-subscription-id")
    # Ensure you have these environment variables set up with your Azure OpenAI credentials
    os.environ["AZURE_OPENAI_API_KEY"] = "ee0dd46654bd4427ba4f5580b5a0db0a"
    os.environ["AZURE_OPENAI_API_BASE"] = "https://xqrojjmb2wjlqopopenai.openai.azure.com/"

    if model_name == "gpt-4o":
        os.environ["AZURE_OPENAI_API_VERSION"] = "2024-05-01-preview"
        os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "gpt-4o"
    

        subscriptionId = keyvault.get_secret("project-subscription-id")
        apiVersion = "2023-10-01-preview"
        url = f"https://management.azure.com/subscriptions/{subscriptionId}/resourceGroups/{workspacename}-common/providers/Microsoft.CognitiveServices/accounts/{workspacename}openai/deployments?api-version={apiVersion}"
        accessToken = credential.get_token("https://management.azure.com/.default")
        response = requests.get(url, auth=BearerAuth(accessToken.token))
        print(f'Initializing Model : {os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]}')
        model = AzureChatOpenAI(
                    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
                    azure_endpoint=os.environ["AZURE_OPENAI_API_BASE"],
                    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
                    max_tokens=4000,
                    temperature=0.9,
                    model_kwargs={"seed": 1337}
                )
        
        print(f'Model {model_name} Initialized')

    elif model_name == "gpt-4.1":

        os.environ["AZURE_OPENAI_API_VERSION"] = "2024-12-01-preview"
        os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "gpt-4.1"
    
    
        subscriptionId = keyvault.get_secret("project-subscription-id")
        apiVersion = "2024-12-01-preview"
        url = f"https://management.azure.com/subscriptions/{subscriptionId}/resourceGroups/{workspacename}-common/providers/Microsoft.CognitiveServices/accounts/{workspacename}openai/deployments?api-version={apiVersion}"
        accessToken = credential.get_token("https://management.azure.com/.default")
        response = requests.get(url, auth=BearerAuth(accessToken.token));
    
        print(f'Initializing Model : {os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]}')
        model = AzureChatOpenAI(
                    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
                    azure_endpoint=os.environ["AZURE_OPENAI_API_BASE"],
                    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                    openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
                    max_tokens=4000,
                    temperature=0.9,
                    model_kwargs={"seed": 1337}
                )
        
        print(f'Model {model_name} Initialized')
        
    
    return model   

def get_filtered_icd_codes(model : AzureChatOpenAI, medical_specialty_subspecialty : str, medical_query : str, icd_code_description_list : list):


    class SpecialtiesResponse(BaseModel):
        queries: List[str] = Field(description="List of queries corresponding to user provided medical specialty")

    # Set up the PydanticOutputParser with the SpecialtiesResponse model
    output_parser = CommaSeparatedListOutputParser()


    system_prompt = """ You are a certified medical coder who assigns ICD-10 codes.

        Goal  
        Given (1) a medical search query, (2) a medical **specialty_subspecialty**, and (3) a user-supplied list of **ICD-10 code : description** pairs, identify which codes are **non-relevant**—i.e., either too generic for the query’s intent or unrelated to the stated specialty_subspecialty.
        
        How to decide relevance  
        • Understand the clinical intent expressed in the query.  
        • Align that intent with the clinical scope of the specialty_subspecialty.  
        • Examine each ICD-10 description in the list.  
          – If it does **not** match both the query intent **and** the specialty_subspecialty with reasonable clinical specificity, mark it non-relevant.  
          – Otherwise, treat it as relevant (do *not* output it).
        
        Response format (strict)  
        Return **only** the non-relevant ICD-10 codes, separated by commas.  
        Example: `A00.1,Y38.2`  
        If every code is relevant, return `[]` (just the two bracket characters).  
        Do **not** include explanations, headings, or extra text.
        
        Inputs (to be injected at runtime)  
        medical_query: {medical_query}
        medical_specialty_subspecialty: {medical_specialty_subspecialty}
        icd_code_description_list: {icd_code_description_list}
        
        Few-shot guidance 
        Example 1
        medical_query: swelling
        medical_specialty_subspecialty: acupuncturist_acupuncturist
        icd_code_description_list: [“R60.9: Fluid retention NOS”, “Y38.2: Terrorism involving other explosions”, “A00.1: Cholera due to Vibrio cholerae”]
        Expected output: Y38.2,A00.1
        
        Example 2
        medical_query: eye socket tumour
        medical_specialty_subspecialty: cliniccenter_oral_and_maxillofacial_surgery
        icd_code_description_list: [“C41.0: Malignant neoplasm of skull bones”, “D3A.01: Benign carcinoid tumour of small intestine”]
        Expected output: D3A.01
        
        Remember: output **only** the comma-separated ICD-10 codes (or `[]`). Do not add any other text.

    """

    # Define the prompt template
    prompt_template = PromptTemplate.from_template(
        template=system_prompt
    )
    
    
    chain = prompt_template | model | output_parser
    result = chain.invoke(input={"medical_specialty_subspecialty": medical_specialty_subspecialty, "medical_query": medical_query,"icd_code_description_list" : icd_code_description_list, "format_instructions" : output_parser.get_format_instructions()})
    return result    


def get_icd_dataset(icd_reference_file : str):
    icd_reference_file = '../../../../shekhar_tanwar/ICD-ICD-Triplet/dataset/icd10.csv'
    dataset_icd = pd.read_csv(icd_reference_file).iloc[:,1:]
    
    dataset_icd = dataset_icd.drop_duplicates()
    dataset_icd = dataset_icd.iloc[:,13:15]
    dataset_icd.columns = ['ICD_Codes','Description']
    dataset_icd['ICD_Codes'] = dataset_icd['ICD_Codes'].apply(lambda x : x.strip())
    dataset_icd['Description'] = dataset_icd['Description'].apply(lambda x : x.strip())
    dataset_icd = dataset_icd.drop_duplicates(subset = ['ICD_Codes'], keep = 'first')

    icd_reference_lookup = {}

    for index, row in dataset_icd.iterrows():
        icd_reference_lookup[row.ICD_Codes] = row.Description

    return dataset_icd, icd_reference_lookup


def combined_gpt_4o_gpt41_dataset(icd_reference_lookup : dict, gpt_4o_path : str, gpt_41_path : str, input_file_specialty_query_code_dict : str, input_file_specialty_query_code_desciption_dict : str):
    
    all_files_gpt4o = [gpt_4o_path +  file for file in os.listdir(gpt_4o_path) if '.json' in file]
    all_files_gpt41 = [gpt_41_path +  file for file in os.listdir(gpt_41_path) if '.json' in file]
    
    specialty_query_dict = {}
    specialty_query_code_desciption_dict = {}

    problematic_specialty_list = []
    problematic_query_list = []
        
    for key in tqdm(range(len(all_files_gpt41))):
        
        file_path = all_files_gpt4o[key]
        medical_specialty_subspecialty = file_path.split('/')[-1].split('.')[0]
        
        with open(all_files_gpt4o[key], 'r') as file:
            data_gpt4o = json.load(file)
    
        with open(all_files_gpt41[key], 'r') as file:
            data_gpt41 = json.load(file)    

        try:
        
            queries = list(data_gpt4o.get(medical_specialty_subspecialty).keys())
            
            query_code_dict = {}
            query_code_description_dict = {}
        
            for medical_query in queries:

                try:
                
                    retrieved_codes_gpt4o = data_gpt4o.get(medical_specialty_subspecialty).get(medical_query)
                    retrieved_codes_gpt41 = data_gpt41.get(medical_specialty_subspecialty).get(medical_query)
            
                    retrieved_codes = list(set(retrieved_codes_gpt4o + retrieved_codes_gpt41))
                    # final retrieved_codes
                    retrieved_codes = [code[:-1] if len(code) >=2 and code[-1] == '0' and code[-2] == '0' else code for code in retrieved_codes ] 
            
                    icd_code_description_list = []
                    icd_code_list = []
            
                    # final selected_codes
                    for code in retrieved_codes:
                        if code in icd_reference_lookup:
                            icd_code_list.append(code)
                            icd_code_description_list.append(code + " : " + icd_reference_lookup.get(code))
            
                    query_code_description_dict[medical_query] = icd_code_description_list
                    query_code_dict[medical_query] = icd_code_list

                except Exception as e:
                    problematic_query_list.append(medical_specialty_subspecialty + " : " + medical_query)

                    
                
            specialty_query_code_desciption_dict[medical_specialty_subspecialty] = query_code_description_dict
            specialty_query_dict[medical_specialty_subspecialty] = query_code_dict

        except Exception as e: 
            problematic_specialty_list.append(medical_specialty_subspecialty)
            
    with open('../../../datasets/datasets_augmented/final_dataset_v40/icd_filtered/specialty_query_dict.json', 'w') as file:
        json.dump(specialty_query_dict, file, indent = 4)

    with open('../../../datasets/datasets_augmented/final_dataset_v40/icd_filtered/specialty_query_code_desciption_dict.json', 'w') as file:
        json.dump(specialty_query_code_desciption_dict, file, indent = 4)    
    
    return specialty_query_dict, specialty_query_code_desciption_dict, problematic_specialty_list, problematic_query_list


def split_input_data(input_file : str, output_dir : str, num_chunks : int):
    
    splits_dir = f"{output_dir}splits/"
    all_file_paths = [splits_dir + file for file in splits_dir if 'json' in file]
    if len(all_file_paths) == num_chunks:
        all_split_files = []
        
        for input_file in all_file_paths:
            with open(input_file, 'r') as f:
                specialty_data = json.load(f)
            
            all_split_files.append(specialty_data)
            
    else:
    
        # Create output directory
        splits_dir = f"{output_dir}splits/"
        os.makedirs(splits_dir, exist_ok=True)

        # Load input data
        with open(input_file, 'r') as f:
            specialty_data = json.load(f)

        # Get list of specialties
        specialties = list(specialty_data.keys())
        chunk_size = len(specialties) // num_chunks + (1 if len(specialties) % num_chunks else 0)

        all_split_files = []
        # Create chunks
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(specialties))

            chunk_specialties = specialties[start_idx:end_idx]
            chunk_data = {specialty: specialty_data[specialty] for specialty in chunk_specialties}

            # Save chunk
            filename = input_file.split('/')[-1].split('.')[0]
            chunk_file = f"{splits_dir}{filename}_split_{i}.json"
            all_split_files.append(chunk_data)
            with open(chunk_file, 'w') as f:
                json.dump(chunk_data, f, indent=4)

            #print(f"Chunk {i}: {len(chunk_specialties)} specialties, saved to {chunk_file}")

    return all_split_files    


def load_splits_chunks(file_path : str):

    file_path_dict = file_path + '/splits/'
    all_files_dict = [file_path_dict + file for file in os.listdir(file_path_dict) if '.json' in file]

    all_split_files_dict = []
    for file_path in all_files_dict:
        with open(file_path, 'r') as file:
            data = json.load(file)
        all_split_files_dict.append(data)

    return all_split_files_dict

def get_icd_code_processor(model_name : str, chunk_specialty_query_code_dict : dict, chunk_specialty_query_code_description_dict : dict):

    # get a list of all specialties
    
    model = initialize_llm(model_name = model_name)
    all_specialties = list(chunk_specialty_query_code_dict.keys())

    # dict to save the result by each specialty
    filtered_medical_code_dict = {}
    retry_dict = defaultdict(list)

    processed_specialties_path = "../../../datasets/datasets_augmented/final_dataset_v40/icd_filtered/filtered_icd_codes/"
    processed_specialties_gpt41 = [file.split('.json')[0] for file in os.listdir(processed_specialties_path) if '.json' in file]
    
    for i in tqdm(range(len(all_specialties))):

        # get the medical specialty
        medical_specialty_subspecialty = all_specialties[i]

        if medical_specialty_subspecialty in processed_specialties_gpt41:
            continue

        else:

            print(f'Processing Specialty : {medical_specialty_subspecialty}')
    
            # get the query code dictionary with respect to the medical_specialty_subspecialty
            # get the query code description dictionary with respect to the medical_specialty_subspecialty
            
            query_code_dict = chunk_specialty_query_code_dict.get(medical_specialty_subspecialty)
            query_code_description_dict = chunk_specialty_query_code_description_dict.get(medical_specialty_subspecialty)
    
            # dicionary to save the final code set
            filtered_query_code_dict = {}
            for medical_query, icd_code_description_list in tqdm(query_code_description_dict.items()):
    
                try:
    
                    unrelated_codes = get_filtered_icd_codes(model = model, medical_specialty_subspecialty = medical_specialty_subspecialty, medical_query = medical_query, icd_code_description_list = icd_code_description_list)
        
                    actual_codes = query_code_dict.get(medical_query)
        
                    final_code_set = list(set(actual_codes).difference(set(unrelated_codes)))
        
                    filtered_query_code_dict[medical_query] = final_code_set
    
                except Exception as e:
                        
                        print(e)
                        retry_dict[medical_specialty_subspecialty].append(medical_query)
                    
                
            filtered_medical_code_dict[medical_specialty_subspecialty] = filtered_query_code_dict
            
            output_path = f"../../../datasets/datasets_augmented/final_dataset_v40/icd_filtered/filtered_icd_codes/{medical_specialty_subspecialty}.json"
            with open(output_path, 'w') as file:
                json.dump(filtered_medical_code_dict, file, indent = 4)
            
            filtered_medical_code_dict = {}
            print(f'Processed Specialty : {medical_specialty_subspecialty}')
        
            time.sleep(0.1)

    return retry_dict


if __name__ == "__main__":

    file_index = 3 # this is the split to process , slits- > 0,1,2,3

    model_name = "gpt-4.1"

    icd_reference_file = '../../../../shekhar_tanwar/ICD-ICD-Triplet/dataset/icd10.csv'    
    gpt_4o_path = '../../../datasets/datasets_augmented/icd_sets_v40/gpt_4o_results/'
    gpt_41_path = '../../../datasets/datasets_augmented/icd_sets_v40/gpt_41_results/'

    input_file_specialty_query_code_dict = '../../../datasets/datasets_augmented/final_dataset_v40/icd_filtered/specialty_query_dict.json'
    output_dir_specialty_query_code_dict = '../../../datasets/datasets_augmented/final_dataset_v40/icd_filtered/specialty_query_splits/'
    
    input_file_specialty_query_code_desciption_dict = '../../../datasets/datasets_augmented/final_dataset_v40/icd_filtered/specialty_query_code_desciption_dict.json'
    output_dir_specialty_query_code_desciption_dict = '../../../datasets/datasets_augmented/final_dataset_v40/icd_filtered/specialty_query_code_desciption_splits/'
    
    
    dataset_icd, icd_reference_lookup = get_icd_dataset(icd_reference_file = icd_reference_file) 
    
    combine_results = False
    
    if combine_results:
        
        specialty_query_dict, specialty_query_code_desciption_dict, _ , _ = combined_gpt_4o_gpt41_dataset(icd_reference_lookup = icd_reference_lookup, gpt_4o_path = gpt_4o_path,  gpt_41_path = gpt_41_path, input_file_specialty_query_code_dict = input_file_specialty_query_code_dict, input_file_specialty_query_code_desciption_dict = input_file_specialty_query_code_desciption_dict)


    create_splits = False
    
    if create_splits:
        # splits for specialty_query_dict
        num_chunks = 4
        print(f'Creating Split for specialty_query_dict')
        all_split_files_specialty_query_code_dict = split_input_data(input_file = input_file_specialty_query_code_dict, output_dir = output_dir_specialty_query_dict , num_chunks = num_chunks)

        # splits for specialty_query_code_desciption_dict
        
        print(f'Creating Split for specialty_query_code_desciption_dict')
        all_split_files_specialty_query_code_desciption_dict = split_input_data(input_file = input_file_specialty_query_code_desciption_dict, output_dir = output_dir_specialty_query_code_desciption_dict, num_chunks = num_chunks)

        
    load_splits = True
    if load_splits:
        print(f'Loading Chunks from {output_dir_specialty_query_code_dict}')
        data_list_specialty_query_code_dict = load_splits_chunks(file_path = output_dir_specialty_query_code_dict)            
        
        print(f'Loading Chunks from {output_dir_specialty_query_code_desciption_dict}')
        data_list_specialty_query_code_description_dict = load_splits_chunks(file_path = output_dir_specialty_query_code_desciption_dict)

    print(f'Loading Chunk : {file_index}')
    chunk_specialty_query_code_dict = data_list_specialty_query_code_dict[file_index]
    chunk_specialty_query_code_description_dict = data_list_specialty_query_code_description_dict[file_index]
    print("#" * 30)
    print('Initializing ICD Filter Component')
    retry_dict = get_icd_code_processor(model_name = model_name, chunk_specialty_query_code_dict = chunk_specialty_query_code_dict, chunk_specialty_query_code_description_dict = chunk_specialty_query_code_description_dict)

    with open('../../../datasets/datasets_augmented/final_dataset_v40/icd_filtered/retry_dict_filtered_icd_codes.json', 'w') as file:
        json.dump(retry_dict, file, indent = 4)

    print(f'All Specialties Processed in Chunk {file_index}!')

    



            





            

        



    

    

    



