# example 
# python3 stage_4_step_1_specialty_verification-v2.py

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


def load_filtered_gpt_41_processed_dataset(file_path : str, output_file_path : str):
    
    all_files = [file_path + file for file in os.listdir(file_path) if '.json' in file]

    all_split_files_dict = []
    for file_path in all_files:
        with open(file_path, 'r') as file:
            data = json.load(file)
        all_split_files_dict.append(data)


    # COMBINING LIST OF DICTIONARY INTO SINGLE DICTIONARY AND FILTER OUT QUERIES WITH NO ICD CODES
    specialty_query_codes_dict = {}
    for data in all_split_files_dict:
        specialty = list(data.keys())[0]
        query_codes_dict = list(data.values())[0]
    
        query_codes = {}
        for query, codes in query_codes_dict.items():
            if len(codes) == 0:
                continue
            else:
                query_codes[query] = codes
    
        specialty_query_codes_dict[specialty] = query_codes


    with open(output_file_path, 'w') as file:
        json.dump(specialty_query_codes_dict, file, indent = 4)

    return all_split_files_dict , specialty_query_codes_dict


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



def get_query_icd_code_description_dataset(icd_reference_lookup : dict,  specialty_query_codes_dict : dict):
    
    # path to the filtered specialt query code dict files
    # file_path = '../../../datasets/datasets_augmented/final_dataset_v40/icd_filtered/filtered_icd_codes/'
    # read all the files in the file_path


    specialties = list(specialty_query_codes_dict.keys())
    specialty_query_code_desciption_dict = {}
    problematic_specialty_list = []
    
    for i in tqdm(range(len(specialties))):
        medical_specialty_subspecialty = specialties[i]

        try:
            query_codes_dict = specialty_query_codes_dict.get(medical_specialty_subspecialty)
            query_code_description_dict = {}
            
            
            for medical_query, retrieved_codes_gpt41 in query_codes_dict.items():
                icd_code_description_list = []
                # final selected_codes
                for code in retrieved_codes_gpt41:
                    if code in icd_reference_lookup:
                        icd_code_description_list.append(code + " : " + icd_reference_lookup.get(code))
        
                query_code_description_dict[medical_query] = icd_code_description_list
    
            specialty_query_code_desciption_dict[medical_specialty_subspecialty] = query_code_description_dict

        except:
            problematic_specialty_list.append(specialty)
    
    with open('../../../datasets/datasets_augmented/final_dataset_v40/icd_filtered/specialty_query_code_desciption_splits/filtered_specialty_query_code_desciption_dict.json', 'w') as file:
       json.dump(specialty_query_code_desciption_dict, file, indent = 4)             

    return specialty_query_code_desciption_dict, problematic_specialty_list


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

def get_specialty_verification(model : AzureChatOpenAI, medical_specialty_subspecialty : str, medical_query : str, icd_code_description_list : list):


    class SpecialtiesResponse(BaseModel):
        queries: List[str] = Field(description="List of queries corresponding to user provided medical specialty")

    # Set up the PydanticOutputParser with the SpecialtiesResponse model
    output_parser = CommaSeparatedListOutputParser()


    system_prompt = """ You are a certified medical coder who assigns ICD-10 codes.

            **Goal:**  
            Given (1) a medical search query, (2) a user-supplied list of **ICD-10 code: description** pairs, and (3) a reference medical **specialty_subspecialty**, identify whether the reference **specialty_subspecialty** is **relevant** or **non-relevant** to the medical query and the ICD-10 code-description list. The result should be **non-relevant** if the reference **specialty_subspecialty** is either too generic for the query/ICD-10 list’s intent or clearly unrelated to that query and code list.
            
            **How to decide relevance:**  
            - **Understand the query’s clinical intent:** Determine what condition, symptom, or scenario the query is describing.  
            - **Consider the ICD-10 code list context:** The ICD-10 codes and descriptions are chosen to be consistent with each other and with the query, representing a specific clinical scenario. Use this combined context to inform your decision.  
            - **Match with the specialty_subspecialty:** Evaluate whether a practitioner of the reference **specialty_subspecialty** typically addresses the query’s scenario:  
              - The specialty_subspecialty pair may consist of a broad specialty and a more focused subspecialty. Emphasize the general **specialty** domain. If the scenario falls under that general field (even if not an exact subspecialty match), it can be considered relevant.  
              - If the scenario (query + codes) **does not fall under** the domain of the reference specialty_subspecialty — for example, the specialty_subspecialty is overly broad/vague for this specific case, or it pertains to a different field of medicine — then label it **non-relevant**.  
              - If the scenario **does** fall under the clinical domain of that specialty_subspecialty (i.e. a provider of that type would reasonably handle such cases), then label it **relevant**.  
            - **Multiple possible specialties:** There may be cases where the query and codes could belong to more than one specialty. You are **only** checking the given reference specialty_subspecialty. If the given specialty_subspecialty is one appropriate choice for this scenario, mark it **relevant** (even if other specialties could also be involved).  
            - **If unsure:** If you cannot confidently determine relevance from the information provided, label the result as `CANNOT_DECIDE`.
            
            **Response format (strict):**  
            Return **only** a single label as the answer: `relevant`, `non-relevant`, or `CANNOT_DECIDE` (use `CANNOT_DECIDE` only if you truly cannot decide). Do **not** include explanations, reasoning, or any additional text.
            
            **Inputs (to be inserted at runtime):**  
            medical_query: *{medical_query}*  
            icd_code_description_list: *{icd_code_description_list}*  
            medical_specialty_subspecialty: *{medical_specialty_subspecialty}*  
            
            **Few-shot guidance (examples):**
            
            Example 1:  
            medical_query: **aging and decreased independence and mobility**  
            icd_code_description_list: **['Z74.3 : Need for continuous supervision', 'Z73.89 : Other problems related to life management difficulty', 'Z73.6 : Limitation of activities due to disability', 'Z60.0 : Phase of life problem', 'Z74.2 : Need for assistance at home and no other household member able to render care', 'Z74.1 : Need for assistance with personal care', 'Z74.09 : Other reduced mobility', 'R54 : Senile debility', 'Z91.81 : History of falling']**  
            medical_specialty_subspecialty: **adult companion_adult companion**  
            Expected output: **relevant**
            
            Example 2:  
            medical_query: **acupuncture for headaches**  
            icd_code_description_list: **['R51.9 : Headache, unspecified', 'G44.209 : Tension-type headache, unspecified, not intractable', 'G43.009 : Migraine without aura NOS', 'G44.89 : Other headache syndrome', 'G43.909 : Migraine NOS', 'G43.709 : Chronic migraine without aura NOS']**  
            medical_specialty_subspecialty: **anesthesiology_addiction medicine**  
            Expected output: *non-relevant**
            
            Example 3:  
            medical_query: **specialty biologic and injectable therapies in healthcarer**  
            icd_code_description_list: **['T88.59XA : Other complications of anesthesia, initial encounter', 'T41.1X5A : Adverse effect of intravenous anesthetics, initial encounter', 'T88.7XXA : Unspecified adverse effect of drug or medicament, initial encounter']**  
            medical_specialty_subspecialty: **cliniccenter_student health**  
            Expected output: **non-relevant**
            
            Example 4:  
            medical_query: **itchy scalp after workplace exposure**  
            icd_code_description_list: **['L23.9 : Allergic contact dermatitis, unspecified cause', 'L28.0 : Circumscribed neurodermatitis', 'L25.9 : Unspecified contact dermatitis, unspecified cause', 'L23.8 : Allergic contact dermatitis due to other agents', 'L23.5 : Allergic contact dermatitis due to plastic', 'L29.8 : Other pruritus', 'R21 : Rash and other nonspecific skin eruption', 'L50.9 : Urticaria, unspecified', 'L24.9 : Irritant contact dermatitis, unspecified cause', 'L27.2 : Dermatitis due to ingested food', 'L24.0 : Irritant contact dermatitis due to detergents']**  
            medical_specialty_subspecialty: **cardiologist**  
            Expected output: **non-relevant**
            
            Example 5:  
            medical_query: **acne worsening at job**  
            icd_code_description_list: **['L70.1 : Acne conglobata', 'L21.9 : Seborrheic dermatitis, unspecified', 'L30.9 : Eczema NOS', 'L25.9 : Unspecified contact dermatitis, unspecified cause', 'L70.0 : Acne vulgaris', 'L23.5 : Allergic contact dermatitis due to plastic', 'L71.9 : Rosacea, unspecified', 'L24.9 : Irritant contact dermatitis, unspecified cause', 'L70.8 : Other acne', 'L70.9 : Acne, unspecified', 'L24.0 : Irritant contact dermatitis due to detergents', 'L71.0 : Perioral dermatitis']**  
            medical_specialty_subspecialty: **dermatopathology_occupational medicine**  
            Expected output: **relevant**

            Example 5:  
            medical_query: **how does diabetes affect brain tumour**  
            icd_code_description_list: **['L70.1 : Acne conglobata', 'L21.9 : Seborrheic dermatitis, unspecified', 'L30.9 : Eczema NOS', 'L25.9 : Unspecified contact dermatitis, unspecified cause', 'L70.0 : Acne vulgaris', 'L23.5 : Allergic contact dermatitis due to plastic', 'L71.9 : Rosacea, unspecified', 'L24.9 : Irritant contact dermatitis, unspecified cause', 'L70.8 : Other acne', 'L70.9 : Acne, unspecified', 'L24.0 : Irritant contact dermatitis due to detergents', 'L71.0 : Perioral dermatitis']**  
            medical_specialty_subspecialty: **neurology**  
            Expected output: **CANNOT_DECIDE**
            
            
            **Remember:** Provide **only** the label (`relevant`, `non-relevant`, or `CANNOT_DECIDE`) as the answer. Do not add any explanation or extra text.
    """

    # Define the prompt template
    prompt_template = PromptTemplate.from_template(
        template=system_prompt
    )
    
    
    chain = prompt_template | model | output_parser
    result = chain.invoke(input={"medical_specialty_subspecialty": medical_specialty_subspecialty, "medical_query": medical_query,"icd_code_description_list" : icd_code_description_list, "format_instructions" : output_parser.get_format_instructions()})
    return result    

def load_splits_chunks(file_path : str):

    file_path_dict = file_path + '/splits/'
    all_files_dict = [file_path_dict + file for file in os.listdir(file_path_dict) if '.json' in file]

    all_split_files_dict = []
    for file_path in all_files_dict:
        with open(file_path, 'r') as file:
            data = json.load(file)
        all_split_files_dict.append(data)

    return all_split_files_dict

def get_filtration_results_specialty_verification(model_name : str, chunk_specialty_query_code_dict : dict, chunk_specialty_query_code_description_dict : dict):
    
    model = initialize_llm(model_name = model_name)

    all_specialties = list(chunk_specialty_query_code_dict.keys())

    #filtered_specialty_query_code_results

    processed_specialties_path = "../../../datasets/datasets_augmented/final_dataset_v40/icd_filtered/specialty_query_code_desciption_splits/filtered_specialty_query_code_results/"
    processed_specialties_gpt41 = [file.split('.json')[0] for file in os.listdir(processed_specialties_path) if '.json' in file]
    

    filtered_specialty_query_codes_dict = {}
    retry_dict = defaultdict(list)

    for i in tqdm(range(len(all_specialties))):


        reference_medical_specialty_subspecialty = all_specialties[i]
        if reference_medical_specialty_subspecialty in processed_specialties_gpt41:
            continue

        else:
            print(f'Processing Specialty : {reference_medical_specialty_subspecialty}')

            query_code_description_dict = chunk_specialty_query_code_description_dict.get(reference_medical_specialty_subspecialty)

            query_code_dict = {}
            
            for medical_query, icd_code_description_list in tqdm(query_code_description_dict.items()):

                try:
                
                    result = get_specialty_verification(model = model, medical_specialty_subspecialty = reference_medical_specialty_subspecialty, medical_query = medical_query, icd_code_description_list = icd_code_description_list)
    
                    # for the pairs which are relevant, we only consider those and add to the final query_code_dict
                    if result[0] == 'relevant':
        
                        code_list = chunk_specialty_query_code_dict.get(reference_medical_specialty_subspecialty).get(medical_query)
                        query_code_dict[medical_query] = code_list

                except Exception as e:
                        
                    print(e)
                    retry_dict[reference_medical_specialty_subspecialty].append(medical_query)
    
            filtered_specialty_query_codes_dict[reference_medical_specialty_subspecialty] = query_code_dict

            output_path = f"../../../datasets/datasets_augmented/final_dataset_v40/icd_filtered/specialty_query_code_desciption_splits/filtered_specialty_query_code_results/{reference_medical_specialty_subspecialty}.json"
            with open(output_path, 'w') as file:
                json.dump(filtered_specialty_query_codes_dict, file, indent = 4)
            
            filtered_specialty_query_codes_dict = {}
            print(f'Processed Specialty : {reference_medical_specialty_subspecialty}')
        
            time.sleep(0.1)

    return retry_dict

if __name__ == "__main__":

    file_index = 1 # this is the split to process , slits- > 0,1,2,3

    print('COMBINING LIST OF DICTIONARY INTO SINGLE DICTIONARY')
    file_path = '../../../datasets/datasets_augmented/final_dataset_v40/icd_filtered/filtered_icd_codes/'
    output_file_path = '../../../datasets/datasets_augmented/final_dataset_v40/specialty_verification/filtered_specialty_query_dict.json'
    all_split_files_dict , specialty_query_codes_dict = load_filtered_gpt_41_processed_dataset(file_path = file_path, output_file_path = output_file_path)

    print('CREATING SPECIALTY QUERY CODE DESCRIPTION DICT')
    icd_reference_file = '../../../../shekhar_tanwar/ICD-ICD-Triplet/dataset/icd10.csv'    
    dataset_icd, icd_reference_lookup = get_icd_dataset(icd_reference_file = icd_reference_file) 
    specialty_query_code_desciption_dict, _  = get_query_icd_code_description_dataset(icd_reference_lookup = icd_reference_lookup,  specialty_query_codes_dict = specialty_query_codes_dict)


    #CREATING SPLITS FOR FILTERED DICTIONARY
    create_splits = True

    input_file_filtered_specialty_query_code_dict = '../../../datasets/datasets_augmented/final_dataset_v40/specialty_verification/filtered_specialty_query_dict.json'
    output_dir_filtered_specialty_query_dict = '../../../datasets/datasets_augmented/final_dataset_v40/specialty_verification/filtered_specialty_query_splits/'

    input_file_filtered_specialty_query_code_dict = '../../../datasets/datasets_augmented/final_dataset_v40/icd_filtered/specialty_query_code_desciption_splits/filtered_specialty_query_code_desciption_dict.json'
    output_dir_filtered_specialty_query_dictionption_dict = '../../../datasets/datasets_augmented/final_dataset_v40/icd_filtered/specialty_query_code_desciption_splits/filtered_specialty_query_code_description_splits/'
    num_chunks = 4

    if create_splits:
        print('CREATING SPLITS')

        # splits for filtered_specialty_query_code_dict

        print(f'Creating Split for specialty_query_dict')
        all_split_files_specialty_query_code_dict = split_input_data(input_file = input_file_filtered_specialty_query_code_dict, output_dir = output_dir_filtered_specialty_query_dict , num_chunks = num_chunks)

        # splits for filtered_specialty_query_code_description_dict
        print(f'Creating Split for specialty_query_dict')
        all_split_files_specialty_query_code_desciption_dict = split_input_data(input_file = input_file_filtered_specialty_query_code_dict, output_dir = output_dir_filtered_specialty_query_dictionption_dict , num_chunks = num_chunks)


    load_splits = True

    if load_splits:
        print('LOADING SPLITS')
        
        print(f'Loading Chunks from {output_dir_filtered_specialty_query_dict}')
        data_list_specialty_query_code_dict = load_splits_chunks(file_path = output_dir_filtered_specialty_query_dict)

        print(f'Loading Chunks from {output_dir_filtered_specialty_query_dictionption_dict}')
        data_list_specialty_query_code_description_dict = load_splits_chunks(file_path = output_dir_filtered_specialty_query_dictionption_dict)


    print(f'Loading Chunk : {file_index}')
    chunk_specialty_query_code_dict = data_list_specialty_query_code_dict[file_index]
    chunk_specialty_query_code_description_dict = data_list_specialty_query_code_description_dict[file_index]

    
    # VERIFYING SPECIALTY GIVEN QUERY AND ICD CODE DESCRIPTION DICT
    print("#" * 30)
    print('Initializing SPECIALTY Filter Component')
    model_name = "gpt-4.1"
    retry_dict = get_filtration_results_specialty_verification(model_name = model_name, chunk_specialty_query_code_dict = chunk_specialty_query_code_dict, chunk_specialty_query_code_description_dict = chunk_specialty_query_code_description_dict)

    with open('../../../datasets/datasets_augmented/final_dataset_v40/icd_filtered/specialty_query_code_desciption_splits/retry_dict_filtered_specialties.json', 'w') as file:
        json.dump(retry_dict, file, indent = 4)

    print(f'All Specialties Processed in Chunk {file_index}!')

    

    
