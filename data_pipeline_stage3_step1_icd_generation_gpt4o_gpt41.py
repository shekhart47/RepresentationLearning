# example
# python3 stage3_step1_icd_generation_gpt4o_gpt41.py

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


def get_query_icd_codes_v4(model : AzureChatOpenAI, medical_specialty : str, medical_query : str):


    class SpecialtiesResponse(BaseModel):
        queries: List[str] = Field(description="List of queries corresponding to user provided medical specialty")

    # Set up the PydanticOutputParser with the SpecialtiesResponse model
    output_parser = CommaSeparatedListOutputParser()

    system_prompt =    """You are a certified medical coder specializing in ICD-10 code assignment for search queries.
                        Your task is to generate correct and relevant ICD-10 codes based on a medical query and medical_specialty provided by the user. 
                        
                        Please note that the medical_specialty provided by the user contains a specialty and a subspecialty as follows : specialty_subspecialty.
                        For example if the specialty is internal medicine and subscpecialty is endocrinology then the user input would be internal medicine_endocrinology
                        
                        Please develop an understanding of the intent of the user provided medical search query with respect to the medical_specialty
                        and adhere to the following guidelines to generate the ICD codes:
                        
                        Strict Guideline:  
                        - Return only valid ICD-10 codes separated by comma , without additional text.
                        - Generate upto 10 ICD-10 codes that are most relevant to the medical_specialty for the identified medical terms
                        - Do not suggest too generic or very broad range of ICD codes which can be mapped to multiple medical_specialties for the given query, instead suggest codes which are highly specific to the query under the given medical_specialty
                        - If there are no ICD codes for the medical query, return "[]" to indicate an empty list.
                        
                        Query Analysis Rules:
                        1. If the query contains ONLY names or locations without medical terms example santa monica, bellevue, return "[]".
                        2. If the query contains the medical_specialty itself (e.g. "cardiologist" for Cardiology) generate top relevant ICD-10 codes for the medical_specialty
                        3. If the query contain conditions/procedures related to the medical_specialty, generate top relevant ICD-10 codes for the searhch query for that medical_specialty
                        4. For any query mentioning a medical profession role ( e.g. "physiotherapist", "cardiologist","oncologist", "dentist" etc ), extract the specialty
                        and generate common ICD-10 codes for conditions typically treated by that specialty. 
                           Exaple : sanata monda orthopedics group. Instruction : Extract orthopedics from the query and generate ICD-10 codes for that
                        4. Always prioritize specific medical conditions mentioned in the query (e.g. "knee pain","back spasm") over generic specialty terms.
                        5. If the query contains both a specialty term AND a specific condition, prioritize codes for the specific condition.
                        6. Analyze the search query and extract relevant tokens from the search query which can be mapped to ICD-10 codes, after extraction return the relevant ICD-10 codes

                        
                        To help you understand the queries with the strict guideline 1, I am providing the following examples:

                        Example 1 : 
                        Medical Query : Early pregnancy signs
                        Medical Specialty : Gynecology
                        ICD_CODES : Z34.91, Z34.01, Z34.81
                        Reason For ICD codes : The query is medical in nature and when analyzed from the perspective of the medical_specialty Gynecology can be associated with the above listed ICD Codes.
                        
                        Example 2 : 
                        Medical Query : chest pain
                        Medical Specialty : Cardiology
                        ICD_CODES : R07.9, I20.9, R07.89, I21.3,I25.10, R07.1, R07.2, I151.9 
                        Reason : the query is medical in nature and can be linked to medical codes
                        Reason For ICD codes : The query is medical in nature and when analyzed from the perspective of the medical_specialty Cardiology can be associated with the above listed ICD Codes.
                        
                        Example 3 : 
                        Medical Query : Sara Moore
                        Medical Specialty : Neurology
                        ICD_CODES : []
                        Reason For ICD codes :  The seach query contains no medical term(s) and thus as per the Strict Guidlines should not have medical codes
                        
                        Example 4 : 
                        Medical Query : Physical Therapist near Highland Ave
                        Medical Specialty : Physiotherapy
                        ICD_CODES : [M54.5, M25.50, M62.81,S33,.5XXA, M79.1, M62.830, M54.2, M54.16, Z96.641, Z47.1]
                        Reason For ICD codes :  Some parts of the searh query are medical in nature, and thus from the search query we can extract 'Physical Therapist' and relate it with the 
                        medical_specialty Physiotherapy to generate top relevant ICD Codes for the extracted terms
                        
                        Example 4 : 
                        Medical Query : Dr Smith Orthopedic surgeon knee replacement
                        Medical Specialty : Orthopedics
                        ICD_CODES : [M17.0, M17.11, M17.12, Z96.651, Z96.652, Z47.1, M25.561, M25.562, M23.50, M79.604]
                        Reason For ICD codes :  Some parts of the searh query are medical in nature, and thus from the search query we can extract 'Orthopedic surgeon knee replacement' and related it with the medical
                        specialty Orthopedics to generate top relevant ICD Codes for the extracted terms
                        
                        Example 5 : 
                        Medical Query : Headache Specialist
                        Medical Specialty : Neurology
                        ICD_CODES : [G43.909, G44.209, R51.9, G44.009, G44.319, G43.709, G44.89, R22.0, G44.019, G43.009]
                        Reason For ICD codes :  The query is medical in nature, and thus we can relate it with the medical_specialty Neurology to generate top relevant ICD Codes for the search query
                        
                        Example 6 : 
                        Medical Query : James Young physiotherapist in Baltimore for back pain
                        Medical Specialty : Physiotherapist 
                        ICD_CODES : [M54.5, M54.4, M54.16, M51.26, M51.27, M47.26, M47.27, M47.28, M54.89, M54.9]
                        Reason For ICD codes :  Some parts of the searh query are medical in nature, and thus from the search query we can extract 'physiotherapist , 'back pain' and related them with 
                        the medical_specialty Physiotherapist to generate top relevant ICD Codes for the extracted terms
                        
                        Example 7 : 
                        Medical Query : Santa Monica Orthopedics group
                        Medical Specialty : Orthopedics 
                        ICD_CODES : ['M17.0', 'M17.11', 'M17.12', 'M25.50', 'M54.5', 'M79.606', 'M25.561', 'M25.562', 'M23.50', 'M79.604]
                        Reason For ICD codes :  Some parts of the searh query are medical in nature, and thus from the search query we can extract Orthopedics and related it with the medical_specialty Orthopedics to 
                        generate top relevant ICD Codes for the extracted terms
                        
                        Example 8 : 
                        Medical Query : Blood Labs
                        Medical Specialty : pathology_blood banking & transfusion medicine 
                        ICD_CODES : ['D50.9' , 'D64.9' ,'D69.6' ,'R73.01' ,'R73.09' ,'R73.9' , 'R79.9' , 'Z13.0' ,'Z01.812']
                        Reason For ICD codes :  Some parts of the searh query are medical in nature, and thus from the search query we extracted blood and related it with the medical 
                        specialty pathology_blood banking & transfusion medicine to generated top relevant ICD Codes for the extracted terms
                       
                        IMPORTANT:
                        PLEASE TAKE YOUR TIME IN UNDERSTANDING THE MEDICAL QUERY WITH RESPECT TO Medical specialty. Your response must contain
                        ONLY upto 10 relevant ICD-10 codes separated by commas, or "[]" if no codes apply. Please make sure that the codes you suggest are limited to the
                        search query and medical_specialty, Do not suggest too generic or very broad range of ICD codes which can be mapped to multiple medical_specialties for the given query
                        
                        Do not include any explanations, headers, or additional text in your response.
                        PLEASE DO NOT DEVIATE FROM THE ABOVE ASSUMPTION. 
                                            
                        Task:  
                        Identify the correct ICD codes for the following medical query.  
                        
                        Format Instructions:
                        {format_instructions}
                        
                        medical_query: {medical_query}
                        medical_specialty: {medical_specialty}
                                    
    """

    # Define the prompt template
    prompt_template = PromptTemplate.from_template(
        template=system_prompt
    )
    
    
    chain = prompt_template | model | output_parser
    result = chain.invoke(input={"medical_specialty": medical_specialty, "medical_query": medical_query, "format_instructions" : output_parser.get_format_instructions()})
    return result

def load_medical_specialty_dataset():
    path = '../../../datasets/datasets_augmented/final_dataset_v40/splits/splits/'
    all_files = [path + file for file in os.listdir(path) if '.json' in file]

    data_list = []
    for file_path in all_files:
        with open(file_path, 'r') as file:
            medical_specialty_dataset = json.load(file)
        data_list.append(medical_specialty_dataset)
    return data_list

def generate_icd_codes(medical_specialty_dataset : dict) -> dict:

    model_name = "gpt-4.1"
    model_gpt41 = initialize_llm(model_name = model_name)

    model_name = "gpt-4o"
    model_gpt4o = initialize_llm(model_name = model_name)

    medical_specialty_query_gpt_4o = {}
    medical_specialty_query_gpt_41 = {}

    retry_dict = defaultdict(list)
    
    all_specialties = list(medical_specialty_dataset.keys())    

    gpt_4o_path = '../../../datasets/datasets_augmented/icd_sets_v40/gpt_4o_results/'
    gpt_41_path = '../../../datasets/datasets_augmented/icd_sets_v40/gpt_41_results/'
    
    processed_specialties_gpt4o = [file.split('.json')[0] for file in os.listdir(gpt_4o_path) if '.json' in file]
    processed_specialties_gpt41 = [file.split('.json')[0] for file in os.listdir(gpt_41_path) if '.json' in file]
    

    for i in tqdm(range(len(all_specialties))):
        medical_specialty = all_specialties[i]


        if (medical_specialty in processed_specialties_gpt4o) or (medical_specialty in processed_specialties_gpt41):
            continue

        else:

            print(f'Processing Specialty : {medical_specialty}')

            medical_queries = medical_specialty_dataset.get(medical_specialty)
    
            query_set_gpt_41 = {}
            query_set_gpt_4o = {}
            
            for k in tqdm(range(len(medical_queries))):
    
                medical_query = medical_queries[k]
                try:
    
                    icd_codes_gpt_41 = get_query_icd_codes_v4(model = model_gpt41, medical_specialty = medical_specialty, medical_query = medical_query)
                    icd_codes_gpt_4o = get_query_icd_codes_v4(model = model_gpt4o, medical_specialty = medical_specialty, medical_query = medical_query)
        
                    query_set_gpt_41[medical_query] = icd_codes_gpt_41
                    query_set_gpt_4o[medical_query] = icd_codes_gpt_4o
    
                except Exception as e:
                    
                    print(e)
                    retry_dict[medical_specialty].append(medical_query)
                    
            medical_specialty_query_gpt_4o[medical_specialty] = query_set_gpt_4o    
            medical_specialty_query_gpt_41[medical_specialty] = query_set_gpt_41
    
            with open(f'../../../datasets/datasets_augmented/icd_sets_v40/gpt_4o_results/{medical_specialty}.json', 'w') as file:
                json.dump(medical_specialty_query_gpt_4o, file, indent = 4)
    
            with open(f'../../../datasets/datasets_augmented/icd_sets_v40/gpt_41_results/{medical_specialty}.json', 'w') as file:
                json.dump(medical_specialty_query_gpt_41, file, indent = 4)
    
            # empty the medical specialty dictionaries : medical_specialty_query_gpt_4o and medical_specialty_query_gpt_41 to 
            # process next specialty
            medical_specialty_query_gpt_4o = {}
            medical_specialty_query_gpt_41 = {}
            print(f'Processed Specialty : {medical_specialty}')
    
            time.sleep(0.1)

    return retry_dict

if __name__ == "__main__":
    chunk = 3 # 0,1,2,3
    
    data_list = load_medical_specialty_dataset()
    medical_specialty_dataset = data_list[chunk]
    retry_dict = generate_icd_codes(medical_specialty_dataset = medical_specialty_dataset)

    with open('../../../datasets/datasets_augmented/icd_sets_v40/retry_dict.json', 'w') as file:
        json.dump(retry_dict, file, indent = 4)