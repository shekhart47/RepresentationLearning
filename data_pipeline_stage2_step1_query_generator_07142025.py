# usage example:
# python3 Stage_2_step1_query_generator_07142025.py

import os
import json
import pandas as pd

import os
import time
import json
import openai
import pickle
import requests
import itertools 
import warnings
import pandas as pd
from tqdm import tqdm
from typing import List, Optional
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


def load_datasets() -> tuple:
    
    path_1 = '../../../datasets/datasets_error_analysis/result_evaluation_inhouse_gpt/gpt_specialty_query_dict_1.json'
    path_2 = '../../../datasets/datasets_error_analysis/result_evaluation_inhouse_gpt/gpt_specialty_query_dict_2.json'
    path_3 = '../../../datasets/datasets_error_analysis/result_evaluation_inhouse_gpt/gpt_specialty_query_dict_3.json'
    path_4 = '../../../datasets/datasets_error_analysis/result_evaluation_inhouse_gpt/gpt_specialty_query_dict_4.json'
    
    
    with open(path_1, 'r') as file:
        data_1 = json.load(file)
    
    with open(path_2, 'r') as file:
        data_2 = json.load(file)
    
    with open(path_3, 'r') as file:
        data_3 = json.load(file)
    
    with open(path_4, 'r') as file:
        data_4 = json.load(file)
    

    gpt_combined_data = {}

    for data in [data_1, data_2, data_3, data_4]:
        for specialty , queries in data.items():
            if specialty in gpt_combined_data:
                gpt_combined_data[specialty].update(queries)
            else:
                gpt_combined_data[specialty] = queries.copy()

    all_specialties = list(gpt_combined_data.keys())
    print(f'Total Specialties Covered : {len(all_specialties)}')


    specialty_query_dict = {}


    for specialty in all_specialties:
        data_specialty = gpt_combined_data.get(specialty, [])
        query_list = []
        for item in data_specialty:
            query = list(item.keys())[0]
            query_list.append(query)

        specialty_query_dict[specialty] = query_list

    all_specialties = list(specialty_query_dict.keys())
    
    return all_specialties


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

    elif model_name == "o4-mini":

        os.environ["AZURE_OPENAI_API_VERSION"] = "2024-12-01-preview"
        os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "o4-mini"
    
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


def query_generator(model : AzureChatOpenAI, medical_specialty : str):
    
    # this version of the prompt can handle queries with location based information added in
    # eg : 
    # 1 - arm pain doctor in redmond, washington : list of ICD codes
    # 2 - Dr Sara Moore : no results
    # 3 - Dr Sara Moore in Redmond, Washington : no results
    # 4 - Dr Sara Moore physiotherapist in Redmond, Waington : list of ICD codes


    class SpecialtiesResponse(BaseModel):
        queries: List[str] = Field(description="List of queries corresponding to user provided medical specialty")

    # Set up the PydanticOutputParser with the SpecialtiesResponse model
    output_parser = CommaSeparatedListOutputParser()



    #NOTE : Feedback From Martin
    
    # omit the treatment and proceddures that are related to broad range of specialties and treated by broad range of medical specalists
    # avoid procedural and treamtnet queries that would result in very broad range of possible diagnosis codes when billed in a claim
    # avoid procedural and treamtnet queries that are related to a very broad range of condition and diseases that may have very broad range of possible diagnosis codes
    
    
    
    system_prompt =    """
            You are a clinical data generation expert specializing in constructing medically valid queries for natural language medical search engines.
            
            Your task is to generate realistic queries that a patient might enter into a healthcare application or search engine. The queries must reflect **symptoms, conditions, procedures and treatments, or diagnostic concerns** related to the target medical specialty and subspecialty.
            
            Instructions:
            1. avoid procedural and treamtnet queries that would result in very broad range of possible diagnosis codes when billed in a claim
            2. avoid procedural and treamtnet queries that are related to a very broad range of condition and diseases that may have very broad range of possible diagnosis codes
            3. Do NOT include any procedural language (e.g., "therapy", "surgery", "MRI", "replacement", "treatment", "rehab").
            4. Focus on real-world concerns a patient might ask when unsure of their condition 
            5. Include both short-form and long-form natural language queries, i.e 2 to 8 words long queries.
            6. Do not include doctor names or clinic addresses.
            7. Use diverse phrasing and vocabulary across examples.
            

            Task:
            Generate 200 queries related to the above specialty.

            Only output the query, no other text.

            Format Instructions:
            {format_instructions}
                            
            medical_specialty: {medical_specialty}
               
    """

    # Define the prompt template
    prompt_template = PromptTemplate.from_template(
        template=system_prompt,
        #input_variables = ['medical_specialty','medical_query','format_instructions']
    )
    
    
    chain = prompt_template | model | output_parser
    result = chain.invoke(input={"medical_specialty": medical_specialty, "format_instructions" : output_parser.get_format_instructions()})
    return result

import random
def generate_queries_specialty(load_retry : bool, dataset_type : Optional[str] ) -> list:

    model_name = "gpt-4o" #Options : { gpt-4.1 , gpt-4o }, gpt-4.1 leads to problems when creating queries, gpt-4o is better at text generation
    model = initialize_llm(model_name = model_name)
    
    if load_retry:

        if dataset_type == 'csv':
            dataset_path = './final_stats_less_200_diagnostic_quries.csv'
            all_specialties = list(pd.read_csv(dataset_path).iloc[:,1:]['Specialties'])

        else:
            print('Loading Retry List From Pickle')
            path = '../../../datasets/datasets_augmented/augmentation_set4/retry_list_v1.pkl'
    
            with open(path, 'rb') as file:
                all_specialties = pickle.load(file)
    else:
        print(f'Generating specialty list from load_datasets()')
        all_specialties = load_datasets()
   
    
    #generated_files_path = '../../../datasets/datasets_augmented/augmentation_set3/iteration1/'
    generated_files_path = '../../../datasets/datasets_augmented/augmentation_set4/iteration1/'
    generated_files = [file.split('.')[0] for file in os.listdir(generated_files_path) if '.json' in file]

    all_specialties = list(set(all_specialties).difference(set(generated_files)))

    
    specialty_query_dict = {}
    retry_list = []
    
    for i in tqdm(range(len(all_specialties))):
    
        medical_specialty = all_specialties[i]
        print(f'Processing Specialty : {medical_specialty}')

        try:
            
            queries_1 = query_generator(model = model, medical_specialty = medical_specialty)
            queries_2 = query_generator(model = model, medical_specialty = medical_specialty)
    
            final_query_set = list(set(queries_1 + queries_2)) #random.sample(list(set(queries_1 + queries_2)), 250)
            print(f'Processed Specialty : {medical_specialty} : Total Queries Generated : {len(final_query_set)}')
            
            specialty_query_dict[medical_specialty] = final_query_set
    
            with open(f'../../../datasets/datasets_augmented/augmentation_set4/iteration1/{medical_specialty}.json', 'w') as file:
                json.dump(specialty_query_dict, file, indent = 4)
    
            specialty_query_dict = {}
    
            time.sleep(0.01)

        except Exception as e:
            print(e)
            print(f'Regenerate Queries For Specialty : {medical_specialty}')
            retry_list.append(medical_specialty)

    return retry_list

if __name__ == '__main__':
        
    load_retry = True
    retry_list = generate_queries_specialty(load_retry = load_retry, dataset_type = 'pickle')

    path  = '../../../datasets/datasets_augmented/augmentation_set4/retry_list_v1.pkl'
    
    with open(path, 'wb') as file:
        pickle.dump(retry_list, file)

    