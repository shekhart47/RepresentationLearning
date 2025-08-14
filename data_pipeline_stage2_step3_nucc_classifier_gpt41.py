# Usage Example
# python3 nucc_classifier_gpt41.py

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

def get_specialty_for_query_nucc_labels(model : AzureChatOpenAI, specialty_list : List, user_query : str):


    class SpecialtiesResponse(BaseModel):
        queries: List[str] = Field(description="List of medical specialties related to a search query")

    # Set up the PydanticOutputParser with the SpecialtiesResponse model
    output_parser = CommaSeparatedListOutputParser()

    system_prompt =    """
                        You are a helpful AI assistant specializing in healthcare.
                        Your task is to identify the top 3 most relevant medical specialties and sub-specialties for a given user search query. You must select only from the list of valid medical_specialty_subspecialty values provided by the user.

                        CRITICAL INSTRUCTIONS:
                        1. Only select labels from the exact list provided: {specialty_list}
                        2. Choose exactly 3 labels that are the most relevant to the user query.
                        3. If there is no exact match, choose the most semantically or clinically appropriate labels from the list.
                        4. Do not modify the names in the list. Return the values exactly as they appear.
                        5. Never invent or create new specialty names.
                        6. Return only the top 3 labels based on relevance to the query.
                        7. Do not return any explanation, formatting, or text outside the selected labels.

                        EXAMPLES:

                        Query: long term facility  
                        specialty_list: [geriatrics_physical therapist, general practice_registered nurse, nursing_home_physician, ...]  
                        → geriatrics_physical therapist, general practice_registered nurse, nursing_home_physician

                        Query: comfort care  
                        specialty_list: [palliative_care_specialist, general practice_registered nurse, geriatrics_physical therapist, ...]  
                        → palliative_care_specialist, general practice_registered nurse, geriatrics_physical therapist

                        INSTRUCTIONS RECAP:
                        - Analyze the meaning and intent behind the user query
                        - Choose the 3 most relevant medical_specialty_subspecialty entries from the given list
                        - Return only the chosen 3 labels with no additional commentary or structure

                        Inputs:
                        user_query: {user_query}  
                        specialty_list: {specialty_list}
                        """

    # add more diverse examples (as part teaching the model)of other specialties and provide 3-5 examples
    
    
    # Define the prompt template
    prompt_template = PromptTemplate.from_template(
        template=system_prompt
    )
    
    
    chain = prompt_template | model | output_parser
    result = chain.invoke(input={"specialty_list": specialty_list, "user_query" : user_query, "format_instructions" : output_parser.get_format_instructions()})
    return result


def load_datasets() -> tuple:
    
    path_1 = '../../datasets/datasets_error_analysis/result_evaluation_inhouse_gpt/gpt_specialty_query_dict_1.json'
    path_2 = '../../datasets/datasets_error_analysis/result_evaluation_inhouse_gpt/gpt_specialty_query_dict_2.json'
    path_3 = '../../datasets/datasets_error_analysis/result_evaluation_inhouse_gpt/gpt_specialty_query_dict_3.json'
    path_4 = '../../datasets/datasets_error_analysis/result_evaluation_inhouse_gpt/gpt_specialty_query_dict_4.json'
    
    
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


def load_ues_keywords_list():

    path = '../../datasets/UES_Keywords/ues_keywords_part2.csv'
    ues_keywords = pd.read_csv(path)
    data_set2 = list(ues_keywords['Keywords'])
    
    with open('../../datasets/UES_Keywords/ues_keywords_in_api_from_audit_file_20250408 1 (1).json', 'r') as file:
        data_set1 = json.load(file)
    
    
    final_keywords = data_set1 + data_set2
    final_keywords = pd.DataFrame(list(zip(final_keywords)), columns = ['final_keywords'])
    final_keywords['sequence_length'] = final_keywords['final_keywords'].apply(lambda x : len(x.split()))    

    return final_keywords


if __name__ == "__main__":
    query_specialty_dict = {}
    
    model_name = "gpt-4o" # "gpt-4o" , "gpt-4.1"
    model = initialize_llm(model_name = model_name)
    

    nucc_specialties = load_datasets()
    final_keywords = load_ues_keywords_list()
    query_list = final_keywords['final_keywords']
    print(f'Total Queries : {len(query_list)}')

    for i in tqdm(range(len(query_list))):
        user_query = query_list[i]
        predictions = get_specialty_for_query_nucc_labels(model = model, specialty_list = nucc_specialties, user_query = user_query)
        query_specialty_dict[user_query] = predictions # creates top 3 medical specialty_subspecialty per query
        time.sleep(0.1)
    
    with open(f'../../datasets/datasets_augmented/augmentation_set3/ues_keyword_nucc_classification/ues_keyword_nucc_classification_{model_name}.json', 'w') as file:
        json.dump(query_specialty_dict, file, indent = 4)
    
    
# Plan Of Action
# create top 3 medical specialty the query could fall in

# q1 : [s1_ms1, s2_ms2, s3_ms3]
# q2 : [s1_ms1, s2_ms2, s3_ms3]


# s1_ms1 : q1
# s2_ms2 : q1
# s3_ms3 : q1
# s1_ms1 : q2
# s2_ms2 : q2
# s3_ms3 : q2

# then for each specualty_subspecialty and query combination genarate a paraphrased version of the query

# s1_ms1 : q1'
# s2_ms2 : q1'
# s3_ms3 : q1'
# s1_ms1 : q2'
# s2_ms2 : q2'
# s3_ms3 : q2'


# then add The paraphrased queries back to the training set
# and then add the original queries to the eval dataset
