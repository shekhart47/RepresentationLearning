# RepresentationLearning


Stage 1 :  Generate Medical Specialties

Total Specialties : 590

Source Code : Users/shekhar_tanwar/ICD-ICD-Triplet/src/datasetbuilder/code-final-icd-cpt-dataset-builder/gpt_augmentation_pipeline/gpt_specialty_detection_pipeline_2.py


Stage 2 :  Generate Seed Queries Per Specialty

This is a 7 step pipeline.

Step 1 : First Use GPT 4.1 and create synthetic queries for each search category ( medical specialty_subspecialty ) <- completed

Source Code: Users/shekhar_icd/src/dataset/query_generator/Stage_2_step1_query_generator_07142025.py

NOTE : The prompt used here is aligned with the query length from production logs. So whatever production logs come in, measure the length of the input sequence ( at token level ) and draw a historgram. Find a range of query lengths where the density of queries are. In this case 2 to 8 tokens length.

NOTE : Total Queries Created  : 250

Step 2 : Then, for the synthetic queries generated, verify if the queries are actually diagnositic in nature. If they are label them as diagnostic, else Procedural or Exclude.
<- completed

Source Code : Users/shekhar_icd/src/dataset/stage2_step2_gpt41_query_classification.py

NOTE : Total Queries Filtered Percentage : 5.104923599320883


Step 3 : Then, for the production queries , classify the queries into different NUCC specialties_subspecialty labels. <- completed


Source Code : Users/shekhar_icd/src/dataset/stage2_step3_nucc_classifier_gpt41.py





NOTE : The production queries are classified into top 3 medical specialty_subspecialty by GPT.

NOTE : Each specialty ( not specilty_subspecialty) is then grouped on the number of production queries we have received ( 2 sets ) and the above plot shows 

Analysis Results: Internal medicine , Pediatrics and Registered Nurse among the top 3 specialties.


Step 4 : For the classified production queries, verify if they are diagnostic in nature, using the code for Step 2. <- completed

Data Path ( classified queries ) : ../../datasets/datasets_augmented/augmentation_set3/ues_keyword_nucc_classification/ues_nucc_dataset.csv

Step 5 : For the Filtered queries, filter out highly similar queries ( cosine similarity >= 0.95 ) by each medical specialty_subspecialty pair. <- completed

Step 6 : For the Filtered queries, create augmented versions of the query and add the augmented version to each medical specialty_subspecialty pair. <- completed

Source Code : Users/shekhar_icd/src/dataset/data_pipeline/stage2_step5_step6.ipynb

Step 7 : Keep the original production queries for evaluation <- completed

Stage 3 : Generate ICD Codes

Step 1 :  For the generated Queries ( N number of queries per specialty ) generate Two set of ICD codes, using GPT 4o and GPT 4.1 <- In progress

Source Code : Users/shekhar_icd/src/dataset/data_pipeline/stage3_step1_icd_generation_gpt4o_gpt41-v2.py

Step 2 :  Once the List of ICD Codes are produced, do the following:
	Step a : Refer the ICD reference lookup file to filter out hallucinated codes
	Step b : For the remaining codes, refer to the descriptions from the ICD reference file and or generate detailed description from GPT 4o ( Research this step )
	Step c :  Evaluation Loop : Feed the combined unique list of ICD codes along with text descriptions back to GPT 4.1 to filter out generic or unrelated codes

Source Code : Users/shekhar_icd/src/dataset/data_pipeline/stage3_step2_icd_verification_gpt4o_gpt41.py

Stage 4: Verify Medical Specialties

Step 1 :  Verify medical specialties using Query and ICD Codes as context 

For the queries, feed the queries and the generated ICD codes back to GPT verify if the original medical specialty_subspecialty can be associated with it or not.

Frame this as a binary classification problem. And report metrics.

Source Code : Users/shekhar_icd/src/dataset/data_pipeline/stage_4_step_1_specialty_verification.py

Stage 5: Attach hard Negatives and Flatenning

Step 1 :  For the specialty, ICD code pairs, attach negative samples

The negatives can be attached using the following approach:

Hard Negatives & Easy Negatives : The ICD codes associated with the query are the positives. Now excluding this list, the remaining ICD codes are candidates for negative sampling. So, now we compute the cosine similarity of the query with respect to the remaining ICD codes, these cosine similarity values are divided in percentile groups. The last group is where the ICD codes having the highest cosine similarity with respect to the query pair lies. From this percentile grouping, sample N/2 codes, and from all other percentile groups, sample the other N/2 codes thereby making a total of N negative samples per query.

Step 2 : Flatten the Dataset and Create Train , Test , Eval set by Specialty

Flatten the triplet dataset and then generate train, test, eval split by specialty.

Source Code : Users/shekhar_icd/src/dataset/data_pipeline/stage_5_step2.py
![Uploading image.png…]()
