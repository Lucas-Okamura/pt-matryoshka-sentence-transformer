import logging
import os
from datetime import datetime
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator, SimilarityFunction
from dotenv import load_dotenv

load_dotenv()

# 1. Configurações e Modelo
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

model_name = "neuralmind/bert-large-portuguese-cased"
model_raw_name = model_name.split("/")[-1]
matryoshka_dims = [1024, 512, 256, 128, 64]
num_gpus = 1
batch_size = 64

model = SentenceTransformer(model_name)

# --- Funções de Normalização ---
def normalize_sts(example):
    # STSb e ASSIN usam escala 0-5. CoSENTLoss e Evaluators preferem 0-1 ou escala original bem definida.
    # Vamos normalizar para 0-1 para consistência.
    score_col = "similarity_score" if "similarity_score" in example else "relatedness_score"
    return {"label": float(example[score_col]) / 5.0}

# 2. Carga e Preparação dos Datasets
train_dataset = {}
eval_dataset = {}

# --- NLI: MultipleNegativesRankingLoss (Melhor para Pares Positivos/Inferência) ---
nli_splits = ["pt_anli", "pt_fever", "pt_ling", "pt_mnli", "pt_wanli"]
for split_name in nli_splits:
    ds = load_dataset("MoritzLaurer/multilingual-NLI-26lang-2mil7", split=split_name)
    # Filtramos apenas Entailment (label 0) para agir como pares positivos no MNRL
    train_dataset[f"nli_{split_name}"] = ds.filter(lambda x: x["label"] == 0).rename_columns({
        "premise": "anchor", "hypothesis": "positive"
    }).select_columns(["anchor", "positive"])

# --- STSb Multi MT (Subset 'pt'): CoSENTLoss (Melhor para Scores Contínuos) ---
stsb_pt = load_dataset("stsb_multi_mt", "pt")
train_dataset["stsb"] = stsb_pt["train"].map(normalize_sts).select_columns(["sentence1", "sentence2", "label"])
eval_dataset["stsb"] = stsb_pt["dev"].map(normalize_sts).select_columns(["sentence1", "sentence2", "label"])

# --- ASSIN 1: CoSENTLoss ---
assin1 = load_dataset("assin", split="train")
train_dataset["assin1"] = assin1.map(normalize_sts).rename_columns({
    "premise": "sentence1", "hypothesis": "sentence2"
}).select_columns(["sentence1", "sentence2", "label"])

# --- ASSIN 2: CoSENTLoss ---
assin2 = load_dataset("assin2")
train_dataset["assin2"] = assin2["train"].map(normalize_sts).rename_columns({
    "premise": "sentence1", "hypothesis": "sentence2"
}).select_columns(["sentence1", "sentence2", "label"])
eval_dataset["assin2"] = assin2["validation"].map(normalize_sts).rename_columns({
    "premise": "sentence1", "hypothesis": "sentence2"
}).select_columns(["sentence1", "sentence2", "label"])

# --- stjiris/IRIS_sts: CoSENTLoss ---
iris_sts = load_dataset("stjiris/IRIS_sts")
train_dataset["IRIS_sts"] = iris_sts["train"].map(normalize_sts).select_columns(["sentence1", "sentence2", "label"])
eval_dataset["IRIS_sts"] = iris_sts["validation"].map(normalize_sts).select_columns(["sentence1", "sentence2", "label"])

# --- eduagarcia/sick-br: CoSENTLoss ---
sick_br = load_dataset("eduagarcia/sick-br")
train_dataset["sick_br"] = sick_br["train"].map(normalize_sts).rename_columns({
    "sentence_A": "sentence1", "sentence_B": "sentence2"
}).select_columns(["sentence1", "sentence2", "label"])
eval_dataset["sick_br"] = sick_br["validation"].map(normalize_sts).rename_columns({
    "sentence_A": "sentence1", "sentence_B": "sentence2"
}).select_columns(["sentence1", "sentence2", "label"])

# --- sentence-transformers/mldr: MultipleNegativesRankingLoss ---
mldr = load_dataset("sentence-transformers/mldr", "pt-triplet")
train_dataset["mldr"] = mldr["train"]

# 3. Definição das Losses com Matryoshka
base_mnrl_loss = losses.MultipleNegativesRankingLoss(model)
matryoshka_mnrl_loss = losses.MatryoshkaLoss(model, base_mnrl_loss, matryoshka_dims=matryoshka_dims)

base_cosent_loss = losses.CoSENTLoss(model)
matryoshka_cosent_loss = losses.MatryoshkaLoss(model, base_cosent_loss, matryoshka_dims=matryoshka_dims)

# Mapeamento dinâmico corrigido
loss_map = {}

# Mapear datasets de TREINO
for name in train_dataset.keys():
    if "nli" in name or "mldr" in name:
        loss_map[name] = matryoshka_mnrl_loss
    else:
        loss_map[name] = matryoshka_cosent_loss

# 4. Evaluator (STSb e ASSIN2 Dev)
evaluators = []
for dim in matryoshka_dims:
    # STSb Dev
    evaluators.append(EmbeddingSimilarityEvaluator(
        sentences1=stsb_pt["dev"]["sentence1"],
        sentences2=stsb_pt["dev"]["sentence2"],
        scores=[s / 5.0 for s in stsb_pt["dev"]["similarity_score"]],
        name=f"stsb-pt-dev-{dim}",
        truncate_dim=dim,
    ))
    # ASSIN2 Dev
    evaluators.append(EmbeddingSimilarityEvaluator(
        sentences1=assin2["validation"]["premise"],
        sentences2=assin2["validation"]["hypothesis"],
        scores=[s / 5.0 for s in assin2["validation"]['relatedness_score']],
        name=f"assin2-dev-{dim}",
        truncate_dim=dim,
    ))
    # IRIS Dev
    evaluators.append(EmbeddingSimilarityEvaluator(
        sentences1=iris_sts["validation"]["sentence1"],
        sentences2=iris_sts["validation"]["sentence2"],
        scores=[s / 5.0 for s in iris_sts["validation"]['relatedness_score']],
        name=f"iris_sts-dev-{dim}",
        truncate_dim=dim,
    ))
    # sick_br Dev
    evaluators.append(EmbeddingSimilarityEvaluator(
        sentences1=sick_br["validation"]["sentence_A"],
        sentences2=sick_br["validation"]["sentence_B"],
        scores=[s / 5.0 for s in sick_br["validation"]['relatedness_score']],
        name=f"sick_br_dev-{dim}",
        truncate_dim=dim,
    ))

dev_evaluator = SequentialEvaluator(evaluators)

# 5. Argumentos de Treinamento
args = SentenceTransformerTrainingArguments(
    output_dir=f"output/{model_raw_name}-matryoshka-sts-pt-loss",
    num_train_epochs=20,
    per_device_train_batch_size=batch_size,
    warmup_steps=0.1,
    weight_decay=0.2,
    fp16=True,
    eval_strategy="steps",
    eval_steps=250,
    save_strategy="steps",
    logging_steps=100,
    learning_rate=5e-5,
    gradient_accumulation_steps=512//(batch_size * num_gpus),
    multi_dataset_batch_sampler="proportional",
    gradient_checkpointing=True, 
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # ddp_find_unused_parameters=False,
    save_total_limit=3
)

# 6. Trainer
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss_map,
    evaluator=dev_evaluator,
)

trainer.train()

# 7. Avaliação Final (Test Sets)
test_evaluators = []
for dim in matryoshka_dims:
    # STSb Test
    test_evaluators.append(EmbeddingSimilarityEvaluator(
        sentences1=stsb_pt["test"]["sentence1"],
        sentences2=stsb_pt["test"]["sentence2"],
        scores=[s / 5.0 for s in stsb_pt["test"]["similarity_score"]],
        name=f"stsb-test-{dim}",
        truncate_dim=dim,
    ))
    # ASSIN2 Test
    test_evaluators.append(EmbeddingSimilarityEvaluator(
        sentences1=assin2["test"]["premise"],
        sentences2=assin2["test"]["hypothesis"],
        scores=[s / 5.0 for s in assin2["test"]['relatedness_score']],
        name=f"assin2-test-{dim}",
        truncate_dim=dim,
    ))
    # IRIS Test
    test_evaluators.append(EmbeddingSimilarityEvaluator(
        sentences1=iris_sts["test"]["sentence1"],
        sentences2=iris_sts["test"]["sentence2"],
        scores=[s / 5.0 for s in iris_sts["test"]['relatedness_score']],
        name=f"iris_sts-test-{dim}",
        truncate_dim=dim,
    ))
    # sick_br Dev
    test_evaluators.append(EmbeddingSimilarityEvaluator(
        sentences1=sick_br["test"]["sentence_A"],
        sentences2=sick_br["test"]["sentence_B"],
        scores=[s / 5.0 for s in sick_br["test"]['relatedness_score']],
        name=f"sick_br_dev-{dim}",
        truncate_dim=dim,
    ))

final_test_evaluator = SequentialEvaluator(test_evaluators)
final_test_evaluator(model)

os.makedirs("models", exist_ok=True)
model.save_pretrained(f"models/{model_raw_name}-matryoshka-sts-pt")
model.push_to_hub(f"iara-project/{model_raw_name}-matryoshka-sts-pt")