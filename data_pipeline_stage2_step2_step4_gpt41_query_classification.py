# python3 stage2_step2_step4_gpt41_query_classification.py

import os
import json

import time
import json
import openai

import pickle
import requests
import itertools 
import warnings
import pandas as pd
from tqdm import tqdm
from typing import List,Optional
from azureml.core import Workspace
from collections import defaultdict
from azure.identity import DefaultAzureCredential
from azureml.core.authentication import ServicePrincipalAuthentication

from langchain.chains import LLMChain
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import  PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.prompts.few_shot import FewShotPromptTemplate

from langchain.output_parsers import PydanticOutputParser, JsonOutputKeyToolsParser, CommaSeparatedListOutputParser
# pd.set_option('display.max_rows', None)
warnings.filterwarnings("ignore")
ws = Workspace.from_config()

def load_datasets(batch_mode : bool, batch_path : Optional[str], single_file_path : str) -> dict:
    if batch_mode:
        print('Loading Synthetic Queries Specialty Classification')
        path = '../../datasets/datasets_augmented/augmentation_set3/'
        all_files = [path + file for file in os.listdir(path) if '.json' in file]
    
        specialty_query_dict = {}
    
        for file_path in all_files:
    
            with open(file_path, 'r') as file:
                data = json.load(file)
    
            for specialty, queries in data.items():
                specialty_query_dict[specialty] = queries

        print(f'Total Specialties : {len(list(specialty_query_dict.keys()))}')
                
        return specialty_query_dict
        
    else:
        print('Loading UES Queries Specialty Classification')
        
        with open(single_file_path, 'r') as file:
            specialty_query_dict = json.load(file)
            
        assert type(specialty_query_dict) == dict
        print(f'Total Specialties : {len(list(specialty_query_dict.keys()))}')
        
        return specialty_query_dict
        

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

def classify_queries(model : AzureChatOpenAI, medical_specialty : str, medical_query : str):
    

    class ClassificationResponse(BaseModel):
        classification: str = Field(description="One of ['diagnostic', 'procedural', 'exclude']")
        #reason : str = Field(...,description="Short explaination for why the query was assigned this classification")

    # Set up the PydanticOutputParser with the SpecialtiesResponse model
    output_parser = PydanticOutputParser(pydantic_object = ClassificationResponse)

    
    system_prompt =    """You are a senior medical coding expert specializing in both diagnostic (ICD) and procedural (CPT) classification. Your task is to analyze a user medical query and its associated specialty, extract medically relevant information, and classify the query as either:

                        - diagnostic: for queries where the user's intent is to identify or describe a medical condition.
                        - procedural: for queries where the user's intent involves a medical procedure or treatment.
                        - exclude: for ambiguous, non-medical, or multi-intent queries where confident classification is not possible.
                        
                        Please follow this chain-of-thought reasoning process:
                        
                        Step 1: **Preprocessing**
                        - Remove names (e.g., “Sara Moore”) and address/location fields (e.g., “Santa Monica”) from the query.
                        - Normalize terms and extract only medically relevant tokens.
                        
                        Step 2: **Medical Intent Extraction**
                        - Identify whether the query expresses:
                          - symptoms (e.g., chest pain, headache, back pain)
                          - conditions (e.g., hypertension, arthritis)
                          - diagnostic tests (e.g., blood test, MRI)
                          - procedures or interventions (e.g., surgery, therapy, replacement)
                        - Analyze whether the medical target specialty or subspecialty aligns with the query terms.
                        
                        Step 3: **Code Type Determination**
                        - If the query asks about a condition, diagnosis, or symptom: assign **diagnostic**.
                        - If the query refers to a procedure, surgery, or therapeutic action: assign **procedural**.
                        - If both types of intents are present (e.g., mentions both symptoms and surgery), assign **exclude**
                        
                        Step 4: **Confidence Check**
                        - Only assign "diagnostic" or "procedural" if the query intent is clearly aligned with one category.
                        - If unsure or if multiple intents are present, return "exclude".
                        
                        Final Output Format:
                        
                        classification: <diagnostic|procedural|exclude>
                        
                        
                        ### Examples
                        
                        **Example 1:**
                        medical_query: "Chest pain"
                        target_specialty: "cardiology"
                        → classification: diagnostic
                        → reason: Mentions a symptom (chest pain) aligned with cardiology; indicates diagnostic evaluation.
                        
                        **Example 2:**
                        medical_query: "ACL reconstruction surgery"
                        target_specialty: "orthopedic_surgery"
                        → classification: procedural
                        → reason: Clearly indicates a surgical procedure aligned with orthopedics.
                        
                        **Example 3:**
                        medical_query: "Dr. Moore for knee replacement"
                        target_specialty: "orthopedics"
                        → classification: exclude
                        → reason: Query mixes a name with procedural content but lacks clarity about user intent or relevant symptoms.
                        
                        Do not generate ICD or CPT codes. Your only task is query classification. Please only output the label associated with the query, no other text.

                        Format Instructions: {format_instructions}  
                        medical_query: {medical_query}
                        target_specialty: {medical_specialty}
                        
                                           
    """

    # Define the prompt template
    prompt_template = PromptTemplate.from_template(
        template=system_prompt,
        #input_variables = ['medical_specialty','medical_query','format_instructions']
    )
    
    
    chain = prompt_template | model | output_parser
    result = chain.invoke(input={"medical_specialty": medical_specialty, "medical_query": medical_query, "format_instructions" : output_parser.get_format_instructions()})
    return result

def get_query_labels(specialty_query_dict : dict, validate_already_classified : bool, classified_specialties_path : Optional[str], output_path : str) -> dict:

    model = initialize_llm(model_name="gpt-4.1")
    
    specialty_query_class = {}

    specialty_query_metrics = {}

    if validate_already_classified:
        if len(classified_specialties_path) == '':
            print('No Path Specified')
        else:
            classified_specialties = [file.split('.')[0] for file in os.listdir(classified_specialties_path) if '.json' in file]
    
    assert type(specialty_query_dict) == dict
    for medical_specialty, queries in tqdm(specialty_query_dict.items()):

        if validate_already_classified:
            if medical_specialty in classified_specialties:
                continue

        print(f'Processing Specialty : {medical_specialty}')
    
        filtered_query_list = []
        for i in tqdm(range(len(queries))):
            medical_query = queries[i]
            result = classify_queries(model = model, medical_specialty = medical_specialty, medical_query = medical_query)        
    
            if result.classification == 'diagnostic':
                filtered_query_list.append(medical_query)
        
        specialty_query_class[medical_specialty] = filtered_query_list

        specialty_query_metrics[medical_specialty] = ((len(queries) - len(filtered_query_list))/(len(queries))) * 100
        
        with open(f'{output_path}/{medical_specialty}.json', 'w') as file:
            json.dump(specialty_query_class, file, indent = 4)

        print(f'Processed Specialty : {medical_specialty}')
        specialty_query_class = {}
                
        time.sleep(0.01)

    return specialty_query_metrics
    
if __name__ == "__main__":
    print(f'Loading Dataset')
    batch_mode = False
    batch_path = '../../datasets/datasets_augmented/augmentation_set3/'
    single_file_path = '../../datasets/datasets_augmented/augmentation_set3/ues_keyword_nucc_classification/specialty_subspecialty_dict.json'
    
    specialty_query_dict = load_datasets(batch_mode = batch_mode, batch_path = batch_path, single_file_path = single_file_path)

    print(f'Filtering Queries By Specialty')
    validate_already_classified = False
    classified_specialties_path = '../../datasets/datasets_augmented/augmentation_set3/ues_keyword_nucc_classification/'
    
    output_path = '../../datasets/datasets_augmented/augmentation_set3/ues_keyword_nucc_classification/nucc_classification_by_specialties' 
    
    specialty_query_metrics = get_query_labels(specialty_query_dict = specialty_query_dict, validate_already_classified = validate_already_classified, classified_specialties_path = '', output_path = output_path)    
    print(f'Saving Classification Metrics')
    with open(f'../../datasets/datasets_augmented/augmentation_set3/ues_keyword_nucc_classification/nucc_classification_by_specialties_metrics.json', 'w') as file:
            json.dump(specialty_query_metrics, file, indent = 4)


    
    