import logging
import os
import traceback
from datetime import datetime

from datasets import Features, Value, interleave_datasets, load_dataset
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses, models
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

load_dotenv()

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

model_name = "lorenzocc/NeoBERTugues"
model_raw_name = model_name.split("/")[-1]

train_batch_size = 16
max_seq_length = 2048
random_seed = 42

max_steps = 500_000
save_steps = 1_000
logging_steps = 100

output_dir = f"output/{model_raw_name}-simcse-pt-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
final_model_dir = f"models/{model_raw_name}-simcse-pt"

simcse_features = Features(
    {
        "sentence1": Value("string"),
        "sentence2": Value("string"),
    }
)

text_features = Features({"text": Value("string")})

# =========================================================
# 1. Modelo
# =========================================================
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode="mean"
)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# =========================================================
# 2. Funções auxiliares
# =========================================================
def clean_text(text):
    if text is None:
        return ""
    return str(text).strip()

def to_text(example):
    return {"text": clean_text(example["text"])}

def is_non_empty(example):
    return bool(example["text"])

def to_simcse_pair(example):
    text = example["text"]
    return {
        "sentence1": text,
        "sentence2": text,
    }

# =========================================================
# 3. Carregamento em streaming
# =========================================================
logging.info("Carregando datasets em streaming...")

aroeira = load_dataset(
    "Itau-Unibanco/aroeira",
    split="train",
    streaming=True,
)

wikipedia_pt = load_dataset(
    "wikimedia/wikipedia",
    "20231101.pt",
    split="train",
    streaming=True,
)

# =========================================================
# 4. Preparação dos datasets
# =========================================================
logging.info("Preparando Aroeira...")
aroeira_remove_cols = [col for col in aroeira.column_names if col != "text"]
aroeira = (
    aroeira
    .map(to_text, remove_columns=aroeira_remove_cols)
    .filter(is_non_empty)
    .cast(text_features)
)

logging.info("Preparando Wikipedia PT...")
wiki_remove_cols = [col for col in wikipedia_pt.column_names if col != "text"]
wikipedia_pt = (
    wikipedia_pt
    .map(to_text, remove_columns=wiki_remove_cols)
    .filter(is_non_empty)
    .cast(text_features)
)

logging.info("Intercalando datasets...")
full_unsupervised_dataset = interleave_datasets(
    [wikipedia_pt, aroeira],
    seed=random_seed,
    stopping_strategy="all_exhausted",
)

train_dataset = full_unsupervised_dataset.map(
    to_simcse_pair,
    remove_columns=["text"],
).cast(simcse_features)

logging.info(f"Features finais do train_dataset: {train_dataset.features}")

# =========================================================
# 5. Loss SimCSE
# =========================================================
logging.info("Criando MultipleNegativesRankingLoss...")
train_loss = losses.MultipleNegativesRankingLoss(model)

# =========================================================
# 6. Argumentos de treinamento
# =========================================================
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    max_steps=max_steps,
    per_device_train_batch_size=train_batch_size,
    learning_rate=5e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,
    bf16=False,
    save_strategy="steps",
    save_steps=save_steps,
    save_total_limit=3,
    logging_steps=logging_steps,
    gradient_accumulation_steps=512 // (train_batch_size * 4),
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    run_name=f"{model_raw_name}-simcse-pt",
    remove_unused_columns=False,
    optim="adamw_torch",
    dataloader_num_workers=0,
    ddp_find_unused_parameters=False,
    data_seed=random_seed,
    accelerator_config={
        "dispatch_batches": False,
        "split_batches": False,
        "even_batches": False,
    },
)

# =========================================================
# 7. Trainer
# =========================================================
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
)

trainer.train()

# =========================================================
# 8. Salvamento local e push para o Hub
# =========================================================
os.makedirs("models", exist_ok=True)
model.save(final_model_dir)

try:
    model.push_to_hub(f"iara_project/{model_raw_name}-simcse-pt")
except Exception:
    logging.error(
        "Erro ao enviar modelo para o Hugging Face Hub:\n"
        f"{traceback.format_exc()}\n"
        f"Para subir manualmente, carregue o modelo salvo em {final_model_dir!r} "
        "e depois execute model.push_to_hub(...)."
    )