import gc
import os
from pathlib import Path
from dataclasses import replace
import json

import evaluate
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq

from datasets import load_dataset
import torch
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune import JupyterNotebookReporter
from transformers import TrainerCallback


class Vetorizer(IterableDataset):
    def __init__(self, tokenizer, dataset, seq_length, total_count):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.seq_length = seq_length
        self.total_count = total_count
    
    def __iter__(self):
        iterator = iter(self.dataset)
        while True:
            try:
                data = next(iterator)
                data['text'][0] = "summarize: " + data['text'][0]
                text_concatenated = " ".join(data['text'])
                label = data['label']
                text_tokenized = tokenizer(text_concatenated, padding='max_length', max_length=self.seq_length.encoder, truncation=True, return_tensors='pt')
                label_tokenized = tokenizer(label, padding='max_length', max_length=self.seq_length.decoder, truncation=True, return_tensors='pt')
                data = {
                    'input_ids': text_tokenized['input_ids'],
                    'attention_mask': text_tokenized['attention_mask'],
                    'labels': label_tokenized['input_ids'],
                }
                yield data
            except StopIteration:
                iterator = iter(self.dataset)

    def __len__(self):
        return self.total_count


def create_dataset(tokenizer, domain_data, args, seq_length):
    train_data = load_dataset('json', data_files=domain_data, split='train', streaming=True)
    no_iter_train_data = load_dataset('json', data_files=domain_data, split='train', streaming=False)
    total_train_data_cnt = len(no_iter_train_data)
    del no_iter_train_data
    gc.collect()

    eval_data = load_dataset('json', data_files=domain_data, split='valid', streaming=True)
    no_iter_eval_data = load_dataset('json', data_files=domain_data, split='valid', streaming=False)
    total_eval_data_cnt = len(no_iter_eval_data)
    del no_iter_eval_data
    gc.collect()
    
    train_dataset = Vetorizer(tokenizer, train_data, seq_length, total_train_data_cnt)
    eval_dataset = Vetorizer(eval_data, eval_data, seq_length, total_train_data_cnt)
    
    return train_dataset, eval_dataset

# 2) 사용자 정의 함수 (원본)
def compute_rouge_scores(references, candidate):
    # references: List[str], candidate: List[str]
    rouge = evaluate.load("rouge")
    scores = rouge.compute(
        predictions=candidate,
        references=references,
        rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_stemmer=True,
        # evaluate 라이브러리 최신 버전에서는 use_aggregator가 True가 기본입니다.
    )
    return scores  # {'rouge1':Score(...), ...}

# 3) Trainer용 compute_metrics 래퍼
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # 3‑1) 토큰 → 텍스트 복원
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 3‑2) 위 커스텀 함수 호출
    raw_scores = compute_rouge_scores(
        references=decoded_labels,
        candidate=decoded_preds
    )

    # 3‑3) Score 객체에서 fmeasure 만 빼와 %로 변환
    #    Score.mid.fmeasure 혹은 Score.fmeasure 속성을 사용
    flat_scores = {}
    for key, score in raw_scores.items():
        # evaluate 0.4.x 기준: score.mid.fmeasure
        # evaluate 0.3.x 기준: score.fmeasure
        f = getattr(score.mid, "fmeasure", None) or getattr(score, "fmeasure")
        flat_scores[key] = f * 100

    return flat_scores


class MyTuneReportCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return control
        from ray import tune
        # compute_metrics 가 리턴한 eval_* 키들 중 필요한 것만 보고
        tune.report(
            eval_loss=metrics.get("eval_loss"),
            eval_rouge1=metrics.get("eval_rouge1"),
            eval_rouge2=metrics.get("eval_rouge2"),
            eval_rougeL=metrics.get("eval_rougeL"),
            eval_rougeLsum=metrics.get("eval_rougeLsum"),
        )
        return control

def model_init():
    return T5ForConditionalGeneration.from_pretrained(model_ckpt)

def make_trainer(training_args, train_dataset, eval_dataset, tokenizer, compute_metrics):
    return Seq2SeqTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,   # 미리 로드된 Dataset 객체
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=None,  # 또는 이 인자를 생략
            label_pad_token_id=tokenizer.pad_token_id
        ),
        compute_metrics=compute_metrics,
        callbacks=[MyTuneReportCallback()]
    )

def backup_trainer(trainer, backup_dir: str = "./text_summarize_final_trainer_backup"):
    """
    Trainer와 연관된 모델, 토크나이저, 학습 인자, 상태, 로그 기록을 로컬에 백업합니다.

    Args:
        trainer: Seq2SeqTrainer 또는 Trainer 인스턴스
        backup_dir: 백업을 저장할 디렉토리 경로
    """
    # 1) 백업 디렉토리 생성
    os.makedirs(backup_dir, exist_ok=True)

    # 2) 모델, 토크나이저 저장
    trainer.save_model(backup_dir)               # pytorch_model.bin, config.json 등
    if trainer.tokenizer is not None:
        trainer.tokenizer.save_pretrained(backup_dir)

    # 3) Trainer state 저장 (trainer_state.json)
    trainer.save_state()

    # 4) TrainingArguments를 JSON으로 저장
    args_path = os.path.join(backup_dir, "text_summarize_final_training_args.json")
    trainer.args.to_json_file(args_path)

    # 5) 로그 히스토리(log_history)를 JSON으로 저장
    log_history = trainer.state.log_history
    logs_path = os.path.join(backup_dir, "text_summarize_final_log_history.json")
    with open(logs_path, "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2, ensure_ascii=False)

    print(f"백업 완료: {backup_dir}")

model_ckpt = 'paust/pko-t5-small'
tokenizer = T5TokenizerFast.from_pretrained(model_ckpt)
model = T5ForConditionalGeneration.from_pretrained(model_ckpt)

domain_name = 'law'

domain_data = {
    'train': f'{domain_name}/train.jsonl',
    'valid': f'{domain_name}/valid.jsonl',
    'test': f'{domain_name}/test.jsonl'
}

seq_length = {
    'encoder': 2048,
    'decoder': 512
}

args = Seq2SeqTrainingArguments(
    output_dir="./new_text_summarize_model",
    evaluation_strategy="steps",
    eval_steps=200,
    logging_steps=100,
    save_steps=500,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=3e-5,
    warmup_steps=500,
    fp16=torch.cuda.is_available(),
    predict_with_generate=True,
    generation_max_length=512,
    generation_num_beams=4
)

train_dataset, eval_dataset = create_dataset(tokenizer, domain_data, args, seq_length)
trainer = make_trainer(args, train_dataset, eval_dataset, tokenizer, compute_metrics)

ray_hp_space = {
    "learning_rate":        tune.loguniform(1e-6, 5e-5),
    "weight_decay":         tune.uniform(0.01, 0.1),
    "per_device_train_batch_size": tune.choice([4, 8, 16]),
    "num_train_epochs":     tune.choice([1, 2, 3]),
    "warmup_steps":         tune.randint(0, 500),
}

scheduler = ASHAScheduler(
    metric="eval_rougeLsum",
    mode="max",
    max_t=3,               # num_train_epochs 상한
    grace_period=1,        # 1 Epoch은 모두 돌려봐야
    reduction_factor=2
)

search_alg = BasicVariantGenerator()

abs_path = os.path.abspath("ray_text_summarize_results")
file_uri = Path(abs_path).as_uri()

reporter = JupyterNotebookReporter(
    overwrite=True,
    metric_columns=[
        "training_iteration",
        "eval_loss",
        "eval_rouge1",
        "eval_rouge2",
        "eval_rougeL",
        "eval_rougeLsum",
        "epoch",
    ]
)

ray.init(include_dashboard=False)

best_hp = trainer.hyperparameter_search(
    direction="maximize",
    backend="ray",
    hp_space=lambda _: ray_hp_space,
    n_trials=10,
    resources_per_trial={"cpu":64, "gpu":8},
    scheduler=scheduler,
    search_alg=search_alg,
    storage_path=file_uri,
    name="best_hp",
    progress_reporter=reporter
)

best_hp_dir_name = "best_hp_for_new_text_summarize_model"
os.makedirs(best_hp_dir_name, exist_ok=True)
with open(f"{best_hp_dir_name}/best_hp.json", "w") as f:
    json.dump(best_hp.hyperparameters, f, indent=2)

print("== Best Hyperparameters ==")
print(best_hp.hyperparameters, best_hp.objective)

best_config = best_hp.hyperparameters
final_args = replace(
    args,
    learning_rate=best_config['learning_rate'],
    weight_decay=best_config['weight_decay'],
    per_device_train_batch_size=best_config['per_device_train_batch_size'],
    num_train_epochs=best_config['num_train_epochs'],
    warmup_steps=best_config['warmup_steps']
)

final_model = T5ForConditionalGeneration.from_pretrained(model_ckpt)

final_trainer = Seq2SeqTrainer(
    model=final_model,              # ← 여기서 model_init이 아니라 model로
    args=final_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer,
        model=None,
        label_pad_token_id=tokenizer.pad_token_id
    ),
    compute_metrics=compute_rouge
)

final_trainer.train()
backup_trainer(final_trainer)
