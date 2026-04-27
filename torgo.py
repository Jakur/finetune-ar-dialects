# %%
import torch
from datasets import Audio, Dataset, DatasetDict, load_dataset, load_from_disk
import os
import evaluate

from dataclasses import dataclass
import torch
import torch.nn.functional as F
import time
from transformers import SeamlessM4TFeatureExtractor, WhisperTokenizer, TrainerCallback, WhisperPreTrainedModel, WhisperConfig, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutput
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, DataCollatorForSeq2Seq
from transformers.models.whisper.modeling_whisper import WhisperEncoder
from transformers import AutoProcessor, AutoModel
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel, Wav2Vec2BertProcessor, Wav2Vec2FeatureExtractor, Wav2Vec2BertForCTC
import torch.nn as nn 
import dill as pickle 
import safetensors 
from tokenizers import Tokenizer
from typing import Any, Dict, List, Union, Optional, Tuple
home = "/media/justin/SSD Ubuntu Stora/datasets/huggingface"
os.environ["TMPDIR"] = "/media/justin/SSD Ubuntu Stora/fat_temp"
os.environ["HF_HOME"] = home
os.environ["HF_HUB_CACHE"] = f"{home}/hub"
os.environ["HF_XET_CACHE"] = f"{home}/xet"
os.environ["HF_ASSETS_CACHE"] = f"{home}/assets"
torch.autograd.set_detect_anomaly(True)

# WAVEFORM = "input_values"
WAVEFORM = "input_features"


# %%
# def append_speaker(example: Dict[str, Dict[str, str]]) -> Dict[str, str]:
#     audio = example.get("audio", {})
#     path = audio.get("path")

#     # derive speaker from audio path
#     basename = os.path.basename(path)
#     match = re.match(r"^([A-Z]C?\d{2})_", basename)
#     speaker = match.group(1) if match else basename.split("_")[0]
#     if speaker not in TORGO_SPEAKERS:
#         raise ValueError(f"Unknown speaker derived from path: {speaker} ({basename})")

#     # append speaker column 
#     return {"speaker": speaker}

# %%
dataset = load_dataset("extraordinarylab/torgo")["test"]
dataset = dataset.map(lambda x: {"length": x["audio"].get_all_samples().duration_seconds }, num_proc=4)
dataset = dataset.filter(lambda x: x["length"] <= 30, num_proc=4) # Manually confirmed the 2 samples above this are garbage
dataset

class Wav2Vec2ForCTC2(Wav2Vec2ForCTC):
    def forward(
        self,
        input_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple | CausalLMOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
        self.eval()
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        # print(attention_mask)
        if labels is not None and labels.max() >= self.config.vocab_size:
            raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")
        # input_values = input_values.clamp(-1e-5, 1e-5)
        # print(input_values.min())
        # print(input_values.max())
        # print(torch.isnan(input_values).any())

        # print(input_values[0][:])

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        # print(outputs)


        # print(attention_mask.size())

        hidden_states = outputs[0]
        # print(torch.isnan(hidden_states).any())
        hidden_states = self.dropout(hidden_states)

        # print(hidden_states[0, 0, :])

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            # print(logits.transpose(0, 1)[0][0:10][:])
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            # print(log_probs.transpose(0, 1)[0][0:10][:])
            # print(input_lengths)
            # print(target_lengths)
            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                # print(loss)

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

# %%
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2ForCTC2.from_pretrained("facebook/wav2vec2-base-960h", ctc_loss_reduction="mean", mask_feature_length=5)
tokenizer = processor.tokenizer

feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
processor = Wav2Vec2BertProcessor(feature_extractor, tokenizer)
# model = Wav2Vec2BertForCTC.from_pretrained("facebook/w2v-bert-2.0", vocab_size=tokenizer.vocab_size, mask_feature_length=5)
model = Wav2Vec2BertForCTC.from_pretrained("/home/justin/Code/Oakland/finetune-ar-dialects/new_torgo2/checkpoint-35000", mask_feature_length=5)

# %%
#     parser.add_argument("--loso_test_speaker", type=str, default="M01", help="Speaker ID for test set.")
# parser.add_argument("--loso_val_speaker", type=str, default="M05", help="Speaker ID for validation set.")
val_speaker = "M05"
test_speaker = "M01"

train_ds = dataset.filter(lambda x: x["speaker"] != val_speaker and x["speaker"] != test_speaker, num_proc=4)
val_ds = dataset.filter(lambda x: x["speaker"] == val_speaker, num_proc=4)
# test_ds = dataset.filter(lambda x: x["speaker"] == test_speaker, num_proc=4)

# dataset.filter(lambda x: x["speaker"] == "FC02")

# %%
def convert(ds, num_shards=32, iterable=True):
    if iterable:
        ds = ds.to_iterable_dataset(num_shards=num_shards)
    return ds.rename_column("text", "labels").map(
        lambda x: {WAVEFORM: x["audio"].get_all_samples().data.squeeze(), 
            "labels": processor.tokenizer.encode(x["labels"].upper()) }).remove_columns(
            ["audio", "speaker", "speech_status", "microphone", "length"])
    # return ds.to_iterable_dataset(num_shards=num_shards).rename_column("text", "labels").map(
    #     lambda x: {"input_features": processor(x["audio"].get_all_samples().data, text=x["labels"].upper(), sampling_rate=16000)}
    #     ).remove_columns(["audio", "speaker", "speech_status", "microphone", "length", "labels"])
train = convert(train_ds, iterable=False)
val = convert(val_ds, num_shards=16, iterable=False)

# %%
import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for processing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2BertProcessor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        # print(features)
        input_features = [feature[WAVEFORM] for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        batch = self.processor(input_features, sampling_rate=16000)
        batch["input_features"] = torch.from_numpy(batch["input_features"])
        # print(batch["input_features"].shape)
        # batch = self.processor.pad(
        #     input_features,
        #     padding=self.padding,
        #     max_length=self.max_length,
        #     pad_to_multiple_of=self.pad_to_multiple_of,
        #     return_tensors="pt",
        # )
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        # print(batch["labels"])

        return batch

# %%
processor.pad(labels=[{"input_ids": [7, 7, 2]}, {"input_ids": [6, 6, 6, 6, 6]}])

# %%
# data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
from transformers import (
    Wav2Vec2ProcessorWithLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainingArguments
)
import regex as re
# from pyctcdecode import build_ctcdecoder
# old_processor = processor
# ctc_decode = build_ctcdecoder(list(processor.tokenizer.vocab), "text.arpa")
# processor = Wav2Vec2ProcessorWithLM(processor.feature_extractor, processor.tokenizer, ctc_decode)
# data_collator = DataCollatorForSeq2Seq(tokenizer=processor.tokenizer, model=model)
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
# data_collator = collator
bs = 2
# print(model)
for n, x in model.named_parameters():
    match = re.search(r"(?<=\.layers\.)\d+", n)
    if "lm_head" in n:
        x.requires_grad_(True)
        continue
    if match:
        layer_num = int(match.group())
        if layer_num > 20: 
            x.requires_grad_(True)
            continue
    x.requires_grad_(False)
    # if "feature_projection" in n:
    #     x.requires_grad_(False)
    # else:
    #     x.requires_grad_(True)

# Freeze
# for x in model.parameters():
#     x.requires_grad_(False)
# model.lm_head.requires_grad_(True)

print(model.lm_head.weight.size())
# model.freeze_base_model()

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
tokenizer = processor.tokenizer

def preprocess_logits_for_metrics(logits, labels):
    # logits = logits[0]
    # print(logits.size())
    # print(labels.size())
    # pred_ids = logits
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels

def compute_metrics(pred):
    pred_ids = pred.predictions[0] # logits (?) 
    label_ids = pred.label_ids
    # for p in pred_ids:
    #     print(p.shape)

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True, group_tokens=False)
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    # print("Prediction: " + pred_str[0])
    # print("Actual: " + label_str[0])

    with open("label.txt", "w") as f:
        for line in label_str:
            f.write(line + "\n")

    with open("pred.txt", "w") as f:
        for line in pred_str:
            f.write(line + "\n")
    

    print("Batch Decode Success")
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

optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if "lm_head" in n],
        "lr": 1e-3, # Higher learning rate for the new head
    },
    {
        "params": [p for n, p in model.named_parameters() if "lm_head" not in n],
        "lr": 5e-5, # Lower learning rate for the pre-trained base
    },
]

from torch.optim import AdamW
optimizer = AdamW(optimizer_grouped_parameters)

# %%
training_args = Seq2SeqTrainingArguments(
    output_dir=f"new_torgo3",
    per_device_train_batch_size=bs,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    warmup_steps=2500,
    num_train_epochs=20,
    # max_steps=15000,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    per_device_eval_batch_size=bs,
    predict_with_generate=False,
    generation_max_length=512,
    save_steps=5000,
    eval_steps=5000,
    logging_steps=100,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    save_total_limit=2,
    fp16=False,
    fp16_full_eval=False,
    batch_eval_metrics=False,
    eval_strategy="steps",
    save_strategy="steps",
    # train_sampling_strategy="sequential"
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train,
    eval_dataset=val,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
    optimizers=(optimizer, None),
    # tokenizer=processor.feature_extractor,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks=[TimingCallback("Torgo", "finetune", 0)],
)

trainer.train()

# from transformers import Trainer

# trainer = Trainer(
#     model=model,
#     data_collator=data_collator,
#     args=training_args,
#     compute_metrics=compute_metrics,
#     train_dataset=train,
#     eval_dataset=val,
# )

# %%
# trainer.train()
