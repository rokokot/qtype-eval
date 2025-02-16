# script to split tidy data into two separate datasets of polar and ~polar questions based on annotator agreement
import pandas as pd
from datasets import Dataset, load_dataset


"""

  This script processes the TiDy QA (primary task) dataset by looking at the annotations column and filtering items for their annotated `yes_no_answer` values. 

  The train split is annotated with a single value: `yes_no_answer: ['NONE']`
  The validation split is annotated with three values: `yes_no_answer: ['NONE', 'NONE', 'NONE']`

  The dataset is also cleaned of languages for which we do not have gold data from UD treebanks, either because UD data is sparce or not provided in the most recent release.
  
"""



dataset = load_dataset("google-research-datasets/tydiqa", "primary_task")

dataset.set_format("pandas")
df_valid_split = dataset['validation'][:].copy()
df_train_split = dataset['train'][:].copy()

languages = ['english', 'russian', 'japanese', 'arabic', 'finnish', 'korean', 'indonesian']

df_valid_split = df_valid_split[df_valid_split['language'].str.lower().isin(languages)]
df_train_split = df_train_split[df_train_split['language'].str.lower().isin(languages)]


polar_filter_valid = df_valid_split['annotations'].apply(lambda x: any(ans == 'YES' for ans in x['yes_no_answer']))
polar_filter_train = df_train_split['annotations'].apply(lambda x: any(ans == 'YES' for ans in x['yes_no_answer']))



polar_valid_split = df_valid_split[polar_filter_valid]
wh_valid_split = df_valid_split[~polar_filter_valid]
polar_train_split = df_train_split[polar_filter_train]
wh_train_split = df_train_split[~polar_filter_train]


polar_questions_valid_split = polar_valid_split[['question_text', 'language', 'annotations']]
wh_questions_valid_split = wh_valid_split[['question_text', 'language', 'annotations']]
polar_questions_train_split = polar_train_split[['question_text', 'language', 'annotations']]
wh_questions_train_split = wh_train_split[['question_text', 'language', 'annotations']]



ds_polar_valid_split = Dataset.from_pandas(polar_questions_valid_split)
ds_wh_valid_split = Dataset.from_pandas(wh_questions_valid_split)
ds_polar_train_split = Dataset.from_pandas(polar_questions_train_split)
ds_wh_train_split = Dataset.from_pandas(wh_questions_train_split)



ds_polar_train_split.save_to_disk("tydiq-train-polar")
ds_wh_train_split.save_to_disk("tydiqa-train-wh")

ds_polar_valid_split.save_to_disk("tydiqa-validation-polar")
ds_wh_valid_split.save_to_disk("tydiqa-validation-wh")
