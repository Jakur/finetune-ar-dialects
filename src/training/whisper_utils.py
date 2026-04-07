import evaluate

import torch
import time
from transformers import WhisperTokenizer, TrainerCallback

torch.multiprocessing.set_sharing_strategy("file_system")
from dataclasses import dataclass  # noqa: E402
from typing import Any, Dict, List, Union  # noqa: E402

tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-small", language="Arabic", task="transcribe"
)
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}


class TimingCallback(TrainerCallback):
    def __init__(self, dialect, type_, seed):
        self.start_time = time.time()
        self.epoch_start_time = None
        self.dialect = dialect
        self.type_ = type_
        self.seed = seed

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        print(f"Epoch {state.epoch} took {epoch_time:.2f} seconds")

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        print(f"Total training time: {total_time:.2f} seconds")
        with open(f"training_time_{self.dialect}_{self.type_}_{self.seed}.txt", "w") as f:
            f.write(
                f"Total training time: {total_time:.2f} seconds or {total_time / 3600:.2f} hours"
            )