import mteb
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv

load_dotenv()

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

model_list = ["iara-project/e5-large-matryoshka-sts-pt"]
dims_list = [None, 512, 256, 128, 64]
filepath = 'data/results_eval_mteb.csv'

tasks = [
    ("Assin2STS", None),
    ("SICK-BR-STS", None),
    ("STSBenchmarkMultilingualSTS", 'pt'),
    
    ('MassiveIntentClassification', 'pt'),
    ('MultiHateClassification', 'por'),
    ('BrazilianToxicTweetsClassification', None),
    ('HateSpeechPortugueseClassification', None),
    ('TweetSentimentClassification', 'portuguese'),

    ('MultiLongDocReranking', 'pt'),
    ('WikipediaRerankingMultilingual', 'pt'),
    ('XGlueWPRReranking', 'pt'),

    ('WebFAQRetrieval', 'por'),
    ('MultiLongDocRetrieval', 'pt'),
    ('WikipediaRetrievalMultilingual', 'pt')
]

for model_name in model_list:
    if "e5" in model_name:
        sentence_transformer_prompts = {"query": "query: ", "document": "passage: "}
    else:
        sentence_transformer_prompts = None
    model_meta = SentenceTransformer(
        model_name,
        device=DEVICE,
        prompts=sentence_transformer_prompts
    )
    for truncate_dim in dims_list:
        # select the desired tasks and evaluate
        task_name_list = []
        model_name_list = []
        main_score_list = []
        truncate_dims_list = []
        for task_info in tasks:
            print(f"""
#############################

[{model_name} - {truncate_dim} dims] Avaliando {task_info[0]} ({task_info[1]})...

#############################
            """)

            task = mteb.get_task(task_info[0], languages=['por'], hf_subsets=task_info[1])

            # with encode kwargs
            result = mteb.evaluate(model_meta, task, encode_kwargs={"batch_size": 256, "truncate_dim": truncate_dim}, cache=None)

            task_name = result.task_results[0].task_name
            model_name = result.model_name
            main_score = result.task_results[0].main_score

            task_name_list.append(task_name)
            model_name_list.append(model_name)
            main_score_list.append(main_score)
            truncate_dims_list.append(truncate_dim)

            print(f"Main Score: {main_score}")

            del task, result
            torch.cuda.empty_cache()
        
        df_results = pd.DataFrame({
            'model_name': model_name_list,
            'embedding_dim': truncate_dims_list,
            'task_name': task_name_list,
            'main_score': main_score_list
        })

        if os.path.exists(filepath):
            df_results_cache = pd.read_csv(filepath)
            df_results = pd.concat([df_results_cache, df_results], axis=0, ignore_index=True)

        df_results.to_csv(filepath, index=False)

        print(f"Avaliação concluída para {model_name} - {truncate_dim} dims!")