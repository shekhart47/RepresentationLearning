# python3 gpt_specialty_detection_pipeline.py
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

def initialize_llm() -> AzureChatOpenAI:
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
    
    print('Model Initialized')
    
    return model


def get_specialty_for_query_nucc_labels(model : AzureChatOpenAI, specialty_list : List, user_query : str):


    class SpecialtiesResponse(BaseModel):
        queries: List[str] = Field(description="List of queries corresponding to user provided medical specialty")

    # Set up the PydanticOutputParser with the SpecialtiesResponse model
    output_parser = CommaSeparatedListOutputParser()

    system_prompt =    """You are a helpful AI assistant specializing in healthcare. 
                        Your task is to identify the medical specialty and sub specialty associated with with a user provided search query from the specialty_list provided by 
                        the user.
                        
                        Please combine the two results as follows : medicalspecialty_subspecialty.
                        
                        For example if the detected medical specialty is Psychiatry & Neurology, and the subspecialty is Neurodevelopmental Disabilities
                        Then the final answer becomes Psychiatry & Neurology_Neurodevelopmental Disabilities
                        
                        Please provide only upto the top 5 comma separated list of medicalspecialty_subspecialty values for each user provided query.

                    
                        For your reference, I am providing the following examples:

                        Example 1 : 
                        Query : long term facility
                        Medical Specialty : geriatrics
                        Subspecialty : physical therapist
                        Final Answer : geriatrics_physical therapist

                        Example 2 : 
                        Query : comfort care
                        Medical Specialty : general practice
                        Subspecialty : registered nurse
                        Final Answer : general practice_registered nurse
                        
                        Based on the user query, return ranked list of medicalspecialty_subspecialty only from the list provided by the user as a comma separated list in 
                        order of relevance. 
                        
                        PLEASE TAKE YOUR TIME IN UNDERSTADING THE SEARCH QUERY AND CONSIDER ALL MEDICAL SPECALTIES AND SUBSPECIALTIES THE QUERY COULD FALL IN.
                        
                        PLEASE ONLY OUTPUT THE MEDICALSPECALTY_SUBSPECIALTY AND NO OTHER TEXT, IF YOU CANNOT FIND A RELEVANT MEDICALSPECALTY_SUBSPECIALTY OUTPUT 'NO'.
                        PLEASE DO NOT DEVIATE FROM THE ABOVE ASSUMPTION. 
                        
                        
                        Format Instructions:
                        {format_instructions}
                        user_query : {user_query}
                        specialty_list: {specialty_list}"""
                        
    # add more diverse examples (as part teaching the model)of other specialties and provide 3-5 examples
    
    
    # Define the prompt template
    prompt_template = PromptTemplate.from_template(
        template=system_prompt
    )
    
    
    chain = prompt_template | model | output_parser
    result = chain.invoke(input={"specialty_list": specialty_list, "user_query" : user_query, "format_instructions" : output_parser.get_format_instructions()})
    return result

def process_nucc_dataset():
    nucc_dataset = pd.read_csv('../../../../../../udit_saini/provider_search_2025/nucc_taxonomy_250.csv')
    nucc_dataset=nucc_dataset[nucc_dataset.Grouping!='Group']
    nucc_dataset.Specialization=nucc_dataset.Specialization.fillna(nucc_dataset.Classification)
    nucc_dataset = nucc_dataset[['Classification','Specialization']]
    nucc_dataset['Classification'] = nucc_dataset['Classification'].apply(lambda x : x.lower())
    nucc_dataset['Specialization'] = nucc_dataset['Specialization'].apply(lambda x : x.lower())
    nucc_dataset['Classification'] = nucc_dataset['Classification'].apply(lambda x : x.replace('/','') if '/' in x else x)
    nucc_dataset['Specialization'] = nucc_dataset['Specialization'].apply(lambda x : x.replace('/','') if '/' in x else x)
    labels_specialties_subspecialties = []
    
    
    for row in nucc_dataset.itertuples():
        labels_specialties_subspecialties.append(row.Classification + '_' + row.Specialization)
    
    
    nucc_dataset['labels_specialties_subspecialties'] = labels_specialties_subspecialties    
    
    return nucc_dataset

def load_query_dataset():
    query_data_train = pd.read_excel('../datasets_udit/train_data_multilabel.xlsx').iloc[:,1:]
    query_data_test = pd.read_excel('../datasets_udit/test_data_multilabel.xlsx').iloc[:,1:]
    query_data = pd.concat([query_data_train, query_data_test])
    query_data = query_data.reset_index().iloc[:,1:]
    
    nucc_dataset = process_nucc_dataset()
    labels_specialties_subspecialties = list(nucc_dataset['labels_specialties_subspecialties'])
    
    
    return labels_specialties_subspecialties, query_data

if __name__ == "__main__":
    labels_specialties_subspecialties, query_data = load_query_dataset()
    user_queries = list(query_data['user_query'])
    
    user_query_specialties = {}
    
   
    model = initialize_llm()
    for i in tqdm(range(len(user_queries))):
        user_query = user_queries[i]
        detected_specialties_subspecialties = get_specialty_for_query_nucc_labels(model = model, specialty_list = labels_specialties_subspecialties, user_query = user_query)
        
        user_query_specialties[user_query] = detected_specialties_subspecialties
        time.sleep(0.1)

        
    
    with open('../datasets_udit/specialty_multilabel_predictions_gpt.json', 'w') as file:
        json.dump(user_query_specialties, file, indent=4)
