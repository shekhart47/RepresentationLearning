# Steps to Follow to run the Data Pipeline

## Stage 1 : Generate Medical Specialties

### Objective

First focus on the medical specialty_subspecialy list, we want to use to build the dataset. As per the NUCC guidelines, there are 800+ specialties, however some of those specialties are not relevant for the PS 
project, as specialties like 'ambulance' won't be searched using the PS platform, so we focus on those specialties where users are tying to find medical providers. So this step is just about the specialties you 
want to focus the finetuning on. Either create a new list of medical specialties_subspecialties for the current release, or use the latest dataset created ( from the previous release ) and work on improving the 
dataset quality in current iteration ( more on this in Stage 2 Step 1 )

Refer this for all possible NUCC : https://taxonomy.nucc.org/

Specialty Expected format : specialty_subspecialty.

NOTE : All this step does is focus on the medical specialties you want to use for the finetuning task. 


## Stage 2 : Generate Seed Queries Per Specialty

### Objective

This is a 7 step pipeline

The idea here is that we will create synthetic queries per specialty_subspecialty, then create paraphrased version of production queries for each specialty subspecialty, and then combine
the two datasets together with more weightage to paraphrased version of production queries to have N queries per specialty_subspecialty.

NOTE : WE ARE NOT USING THE ACTUAL PRODUCTION QUERIES FOR TRAINING THE MODEL, AS THAT RESULT IN WOULD BE DATA LEAKAGE AND THUS OVERFITTING. The idea is that by creating paraphrased versions
of the UES queries, we are essenstially teaching the model nuances of the different formats in which the paraphrased versin of the actual queries are being asked.

REMEMBER, the actual UES queries are used for evaluation, do not directly use in the training, ever.

### Step 1 : Synthetic Query Genration By Specialty_Subspecialty

First Use GPT 4.1 and create synthetic queries for each search category ( medical specialty_subspecialty )

#### Dataset Input : 

These are sharded specialty_query dictionary files from the previous version of the model. I load these to improve the model over this specialty_subspecialty list.

Users/shekhar_icd/datasets/datasets_error_analysis/result_evaluation_inhouse_gpt/gpt_specialty_query_dict_1.json
Users/shekhar_icd/datasets/datasets_error_analysis/result_evaluation_inhouse_gpt/gpt_specialty_query_dict_2.json
Users/shekhar_icd/datasets/datasets_error_analysis/result_evaluation_inhouse_gpt/gpt_specialty_query_dict_3.json
Users/shekhar_icd/datasets/datasets_error_analysis/result_evaluation_inhouse_gpt/gpt_specialty_query_dict_4.json

NOTE : Since we are about to create synthetic queries for specialty_subspecialty list, Instead of loading the above files, you can also directly pass in any custom specialty_subspecialty list 
you want to focus on, this would be passed in as a file path in the 'load_specialties_list()' function and enable 'load_specialties_user_input' flag to True to enable loading a 
custom specialty_subspecialty list . Refer to the script mentioned in the below Source Code path for reference on how to do that.

#### Dataset Output : 

Users/shekhar_icd/datasets/datasets_augmented/augmentation_set4_v40/iteration1/

NOTE : For a given release, With each variation in prompt, I create a new iteration{x} folder. For example with a version 1 of prompt, we'll have iteration1 , then for version2 of the prompt
we'll have iteration2 folder etc. 

NOTE : For the folder structure name, augmentation_set{x}_v{model_version} folder means that we are creating a augmented dataset for a given model's release. So if we're working on v40 of the 
model, we'll create this folder structure just to make sure the data version control matches with the mode version control. I wish we had mlflow enabled in UAIS to log stuff.

This is a dictionary where specialty_subspecialies are keys are values are a list of synthetic queries.

#### Source Code: Users/shekhar_icd/src/dataset/data_pipeline/Stage_2_step1_query_generator_07142025.py

NOTE : The prompt used here is aligned with the query length from production logs. So whatever production logs come in, measure the length of the input sequence 
( at token level ) and draw a historgram. Find a range of query lengths where the density of queries are. In this case 2 to 8 tokens length.


### Step 2 : Synthetic Query Verification

For the synthetic queries generated in Step 1, verify if the queries are actually diagnositic in nature. If they are, label them as diagnostic, else Procedural or Exclude if the model cannot 
determine. Only diagnostic queries i.e. queries which can have associated ICD-10 codes are considered.


The script used in this step, used for Step 2 and also for Step 4. All we are doing here is given a medical specialty_subspecialty and a query ( synthetic or paraphrased UES query ) 
verify if the qquery is diagnostic in nature.

#### Dataset Input : 

For Step 2, use batch_mode = True in the main function

The output of the Step 1, becomes input of this step.

Users/shekhar_icd/datasets/datasets_augmented/augmentation_set4_v40/iteration1/

NOTE : Eveytime you'll create / test a new prompt, make a new folder to store the data generated, in a separate iteration{x}folder. And with every run, feed that path in the script. 

#### Dataset Output : 
This is where the output of this script is saved. We are essentially filtering out the non-diagnostic queries and saving only the diagnostic ones to this folder.

Users/shekhar_icd/datasets/datasets_augmented/augmentation_set4_v40/ues_keyword_nucc_classification/nucc_classification_by_specialties

The output is a  dictionary where the key is medical specialty_subspecialty and value is a list of all 'synthetically generated' diagnositic queries associated with that.


### Step 3 : Production Query Classification into NUCC specialties.

Then, for the production UES queries , first classify the queries into different NUCC specialties_subspecialty labels. This step is done so that we can then again have a dictionary format 
similar to Step 2, i.e medical specialty subspecialty : list of relatable production queries.

#### Dataset Input :

Again load the specialty_subspecialty list you are focussing on for finetuning task.

Users/shekhar_icd/datasets/datasets_error_analysis/result_evaluation_inhouse_gpt/gpt_specialty_query_dict_1.json
Users/shekhar_icd/datasets/datasets_error_analysis/result_evaluation_inhouse_gpt/gpt_specialty_query_dict_2.json
Users/shekhar_icd/datasets/datasets_error_analysis/result_evaluation_inhouse_gpt/gpt_specialty_query_dict_3.json
Users/shekhar_icd/datasets/datasets_error_analysis/result_evaluation_inhouse_gpt/gpt_specialty_query_dict_4.json

Then load the UES Keywords list, which you need to clasify into the specialty_subspecialty type

Users/shekhar_icd/datasets/UES_Keywords/ues_keywords_part2.csv

#### Dataset Output : The output file is saved here
Users/shekhar_icd/datasets/datasets_augmented/augmentation_set4_v40/ues_keyword_nucc_classification/ues_keyword_nucc_classification_gpt-4o.json

Users/shekhar_icd/datasets/datasets_augmented/augmentation_set3_v40/ues_keyword_nucc_classification/specialty_subspecialty_dict.json

### Step 4 : Production Query Verificaiton

Once the production queries are classified into different specialty_subspecialties, use the below script to filter out non-diagnostic queries.

This time set batch_mode = False, and pass in the path to the output file from Step 3.

#### Dataset Input :

Users/shekhar_icd/datasets/datasets_augmented/augmentation_set4_v40/ues_keyword_nucc_classification/ues_keyword_nucc_classification_gpt-4o.json

#### Dataset Output : 
The result is a folder where for each specialty_subspecialy a new file is created which contains the relevant diagnostic UES queries.

Users/shekhar_icd/datasets/datasets_augmented/augmentation_set3_v40/ues_keyword_nucc_classification/nucc_classification_by_specialties/

### Step 5 : Diverse UES Query Selection

This is a notebook, ( I apologize due to lack of time I couldn't convert it into a script )

 


#### Dataset Input:

- Heading : Load Synthetic Dataset and NUCC classified UES queries Dataset
- 
1 - Load the output of Stage 2 Step 2.
Users/shekhar_icd/datasets/datasets_augmented/augmentation_set3_v40/gpt41_query_clasification_results/


2 - Load the output of Stage 2 Step 4.
Here we load the specialty_subspecialty : list of UES diagnostic queries mapping from Step 4, and then for each medical specialty_subspecialty, filter out highly similar UES
queries ( as described below )

Users/shekhar_icd/datasets/datasets_augmented/augmentation_set3_v40/ues_keyword_nucc_classification/nucc_classification_by_specialties/

 - Heading : Find Distinct UES Queries In Each Specialty Group

The notebook uses Query_Diversity_Selection_Algorithm module, which given a list queries, threshold and the NovaSearch stella model, 
stored here : 
embedding_model = 'Users/shekhar_tanwar/ICD-ICD-Triplet/model/NovaSearch_stella_en_1.5B_v5/' 

and then filters out similar queries with cosine similarities >= 0.95.


So after this step you'll have a dictionary where the key is a medical specialty_subspecialty and value is a list of distinct diagnostic UES queries.

NOTE : We are only filtering out similar UES queries, as using the remaining distinct UES queries we'll create paraphrased version of them and add that to the training set. We don't filter out 
the synthetic queries created in stage_2_step1 as we already have unique queries there, because of the prompt I designed.

### Step 6 : Paraphrase Distinct UES Queries

- Heading : Paraphrase Distinct UES Queries

Using the same notebook as above, use the specialty_subspecialty and distinct diagnostic UES keywords from heading -> 'Find Distinct UES Queries In Each Specialty Group', create paraphrased version of those queries under heading -> Paraphrase Distinct UES Queries

#### Dataset Input :

The output of heading 'Find Distinct UES Queries In Each Specialty Group' is used as input.

#### Dataset Output :

The paraphrased queries by specialty_subspecialty are saved here: 'specialty_paraphrased_queries_dict.json'

- Heading : Generating Summary Stats For Adding Paraphrased Production Queries To Synthetic Queries

  This section basically, combines the synthetic queries and distinct paraphrased UES queries by each specialty_subspecialty to compute metrics around total queries by each type ( paraphrsased vs synthetic ) across medical specialty_subspecialty.

  NOTE : Since I am setting a threshold of 250 queries per specialty_subspecialty type, The stats show that when we do diagnostic based filtration in Step 2, then the total specialties for which the number of queries  > 250 were only 116 out of original 590. So this indicates that we need to rerun stage 2 step 1, and create more queries again, which then led to creation of augmentation_set4_v40.

Basically, the idea here is that you need to combine the synthetic queries and paraphrased queries together for each specialty_subspecialty. If the total query count is < X, where X is some total query count defined by you ( say 100 queries per specialty_subspecialty ), then you would need to regenerate queries again, this time improve the prompt to specifically create diagnostic queries and then rerun stage 2 
step 2 ( to filter out non-diagnostic queries created from the latest round of queries generated ) and then run stage 2 step 6 again to combine the paraphrased UES queries and the new set of synthetic queries together. Overall, this is an iterative process.

NOTE : Under heading : 'Generating Summary Stats For Adding Paraphrased Production Queries To Synthetic Queries' we are saving final_stats_less_200_diagnostic_quries.to_csv('./final_stats_less_200_diagnostic_quries.csv'), this csv file also gives us those specialties for which we need to regenerate queries, which can then be passed in the Stage 2 Step 1.

Once The fresh set of synthetic queries are created, which are vetted using Stage 2 Step 2, then using the heading : 'Adding Synthetic Queries and Paraphrased Production Queries' we can combine the new set of synthetic queries and distinct paraphrased queries we can re-generate summary stats, this time it should generate better results.

Once the two queries list for each medical specialty_subspecialty are combined using the heading : Combine and Sample Queries we can then use the sampling algorighm mentioned there to give more weightage to 
paraphrased production queries and lesser weightage to synthetic queries, the logic can be modified accordingly.

The final file is saved here : Users/shekhar_icd/datasets/datasets_augmented/final_dataset_v40/final_dataset_v40.json

This file will now be used to create ICD codes, it is too big for the LLM to process in one go, and often results in high completion time. A better apprach is to shard the file into
smaller chunks, and then process each chunk separately across different computes and respective terminals, this often avoids Rate Limit Error from OpenAI.

The chunks are saved here : Users/shekhar_icd/datasets/datasets_augmented/final_dataset_v40/splits/

Remember in the next scripts, we will use these chunks now.

## Stage 3 : ICD Code Generation & Verification

### Objective

In this Stage, using the output of the previous step, i.e. Stage 2 Step 6, we will load the chunks one by one, and then will attach ICD Codes to the queries and then we will verify the ICD codes attached for hallunication and specificity to the specialty_subspecialty and the intent of the query.

### Step 1 : Attaching ICD Codes

Here using the output of Stage 2 Step 6, we now have 4 chunks , each consisting of specialty_subspecialty and respective 250 query list.

We now need to attach ICD codes from GPT to these queries. The current step, will take each medical specialty_subspecialty, query and then create 2 set of ICD codes using GPT 4o and GPT 4.1. We intentionally create upto 10 coder per query, as based on previous experiments in Price Care we saw hallucinated codes being created.


#### Dataset Input :
In the load_medical_specialty_dataset() function, Load the split dataset chunks created in Stage 2 Step 6.
Users/shekhar_icd/datasets/datasets_augmented/final_dataset_v40/splits/splits/

Then using generate_icd_codes() function we create ICD codes for all the queries which are present in each chunk. 
We process the files in chunks, as feeding one large file to GPT leads to RateLimiting Error, and thus this sharding approach speeds things up for us.

#### Dataset Output:

The ICD codes generate from GPT 4o and GPT 4.1 are saved here :

Users/shekhar_icd/datasets/datasets_augmented/icd_sets_v40/gpt_4o_results/
Users/shekhar_icd/datasets/datasets_augmented/icd_sets_v40/gpt_41_results/

### Step 2 : Verification Of ICD Codes

Once a list of ICD codes from the two models (GPT 4o and GPT 4.1) are created for each query belonging to a specialty_subspecialty pair, then we need to ensure the following:
1 - Filter out any hallucinated codes, that is ICD codes which are not part of the icd10.csv file
2 - Filter our unrelated, too generic codes and only keeping the ones which are highly specific to a specialty_subspecialty and query pair.

#### Dataset Input :

Refer this for icd10.csv file : Users/shekhar_tanwar/ICD-ICD-Triplet/dataset/icd10.csv'
gpt_4o_path : Users/shekhar_icd/datasets/datasets_augmented/icd_sets_v40/gpt_4o_results/
gpt_41_path : Users/shekhar_icd/datasets/datasets_augmented/icd_sets_v40/gpt_41_results/


#### Instructions:

The script is designed to achieve 3 things, Follow the below instructions to run the script and generate output:

1 - In the current script there are 3 flags : combine_results,  create_splits and load_splits ( please locate them in the code first )

2 - Enable all flags to False, then for the 'combine_results' set that to True. Enbling this flag and running the script using 'python3 stage3_step2_icd_verification_gpt4o_gpt41.py' will load the files in gpt_4o_results and gpt_41_results, will combined them on medical specialty_subspecialty, then will apply related filtration using icd10.csv and GPT. 

After the filtration is complete, we will have a dictionary where for each specialty_subspecialty we will have a 250 queries and related ICD 10 codes. The script will then make a duplicate out this output, and will then attach ICD 10 code's description to each ICD code and thus will save two files:

First here : Users/shekhar_icd/datasets/datasets_augmented/final_dataset_v40/icd_filtered/specialty_query_dict.json -> which is dictionary with specialty_subspecialty and query : ICD code mapping

Second here : Users/shekhar_icd/datasets/datasets_augmented/final_dataset_v40/icd_filtered/specialty_query_code_desciption_dict.json -> which is dictionary with specialty_subspecialty and query : ICD code description mapping

Once the results are saved, disable 'combine_results' flag and enable the 'create_splits' flag , i.e. set it to True. This will load the the 'specialty_query_dict.json' and 'specialty_query_code_desciption_dict.json' files i.e. output of the 'combine_results' flag and will create 4 chunks each of them.

The splits would be saved here : 

Splits for specialty_query_dict:
Users/shekhar_icd/datasets/datasets_augmented/final_dataset_v40/icd_filtered/specialty_query_splits/'


Splits for specialty_query_code_desciption_dict:
Users/shekhar_icd/datasets/datasets_augmented/final_dataset_v40/icd_filtered/specialty_query_code_desciption_splits/'

Once this step is complete, then disable the 'create_splits' flag, and enable the 'load_splits', which will load a specific chunk from the 4 chunks of specialty_query_splits and specialty_query_code_desciption_splits. The chunk to load is a paramter (file_index) which user will set at the beginning of the main function. Possible values of this file_index are 0,1,2,3

Once the chunks for specialty_query_dict and specialty_query_code_desciption_dict  are loaded from :
Users/shekhar_icd/datasets/datasets_augmented/final_dataset_v40/icd_filtered/specialty_query_splits/' and Users/shekhar_icd/datasets/datasets_augmented/final_dataset_v40/icd_filtered/specialty_query_code_desciption_splits/'

Then we use the get_icd_code_processor() function to filter out un-related and too-generic ICD code for each query. The output would be saved here : Users/shekhar_icd/datasets/datasets_augmented/final_dataset_v40/icd_filtered/filtered_icd_codes/










  




























