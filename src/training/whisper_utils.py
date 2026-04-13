import evaluate

import torch
import time
from transformers import WhisperTokenizer, TrainerCallback, WhisperPreTrainedModel, WhisperConfig, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoder
import torch.nn as nn 
import dill as pickle 
import safetensors 
from tokenizers import Tokenizer

torch.multiprocessing.set_sharing_strategy("file_system")
from dataclasses import dataclass  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple, Union  # noqa: E402

tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-small", language="Arabic", task="transcribe"
)
custom_tokenizer = Tokenizer.from_file("exp_tokenizer")
custom_tokenizer.enable_padding()
# custom_tokenizer = WhisperTokenizer(vocab_file='vocab.json',
#                              merges_file="merges.txt",
#                              unk_token='<UNK>',
#                              bos_token= '<END>',
#                              pad_token= '<PAD>',
#                              eos_token= '<END>',
#                              add_prefix_space=True,
#                              model_max_length = 512,
#                             language='Arabic', task='transcribe')
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
with open('token_to_idx.bin', 'rb') as f:
    token_to_idx = pickle.load(f)
with open('idx_to_token.bin', 'rb') as f:
    idx_to_token = pickle.load(f)


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
    from safetensors.torch import save_file 
    print("Computing Metrics")
    pred_ids = pred.predictions[0]
    label_ids = pred.label_ids
    # print(label_ids[0])
    # print(pred_ids[0])

    label_ids[label_ids == -100] = tokenizer.pad_token_id
    # print(pred_ids.shape)
    # print(label_ids.shape)
    # pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    # label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    # print(label_str)
    # save_file({"pred_ids": torch.tensor(pred_ids)}, "tensor.data")
    # pred_str = label_str
    # pred_str = custom_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    pred_str = custom_tokenizer.decode_batch(pred_ids, skip_special_tokens=True)
    print("Prediction: " + pred_str[0])
    print("Actual: " + label_str[0])
    # try:
    #     pred_str = custom_tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    # except Exception as e:
    #     print(e)
    # print(pred_str)
    print("Batch Decode Success")
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}

# def compute_metrics(pred, compute_result=False):
#     if compute_result:
#         print("Computing Result")
#         wer = 100 * wer_metric.compute()
#         cer = 100 * cer_metric.compute()
#         print(wer)
#         print(cer)
#         return {"wer": wer, "cer": cer}
#     pred_ids = pred.predictions
#     label_ids = pred.label_ids

#     label_ids[label_ids == -100] = tokenizer.pad_token_id
#     print("Trying to do something")
#     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
#     print(pred_str)
#     assert(False)

#     wer_metric.add_batch(predictions=pred_str, reference=label_str)
#     cer_metric.add_batch(predictions=pred_str, references=label_str)
#     return None
    # wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    # cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
    # if compute_result:
    #     wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    #     return {"wer": wer, "cer": cer}
    # else:
    #     return None 


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


_HIDDEN_STATES_START_POSITION = 2

class ExtendedWhisperConfig(WhisperConfig):
    def __init__(
        self,
        ctc_loss_reduction: str = "mean",
        final_dropout: float = 0.0,
        ctc_zero_infinity: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ctc_loss_reduction = ctc_loss_reduction
        self.final_dropout = final_dropout
        self.ctc_zero_infinity = ctc_zero_infinity


class WhisperEncoderForCTC(WhisperPreTrainedModel):
    config_class = ExtendedWhisperConfig

    def __init__(self, config):
        super().__init__(config)

        self.encoder = WhisperEncoder(config)
        self.dropout = nn.Dropout(config.final_dropout)
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `WhisperEncoderForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        
        output_hidden_size = (
            config.output_hidden_size
            if hasattr(config, "add_adapter") and config.add_adapter
            else config.hidden_size
        )
        # vocab_size = len(idx_to_token)
        # vocab_size = custom_tokenizer.vocab_size
        vocab_size = custom_tokenizer.get_vocab_size()
        # vocab_size = config.vocab_size
        print(f"Creating Linear: {output_hidden_size} --> {vocab_size}")
        self.lm_head = nn.Linear(output_hidden_size, vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_features: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.encoder(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)
        l_size = logits.size()
        # print(f"Logits {l_size}")

        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(
                    f"Label values must be <= vocab_size: {self.config.vocab_size}"
                )

           
            attention_mask = (
                attention_mask
                if attention_mask is not None
                else torch.ones_like(input_features.transpose(1, 2), dtype=torch.long)
            )
            # TODO: check if this is correct
            input_lengths = torch.full([l_size[0]], l_size[1])
            # input_lengths = self._get_feat_extract_output_lengths(
            #     attention_mask.sum(-1)
            # ).to(torch.long)
            # print(input_lengths.size())
            # print(input_lengths[0, 0:25])
            # print("Batch size " + str(labels.size()[0]))
            # assuming that padded tokens are filled with -100
            # when not being attended to
            decoded = tokenizer.batch_decode(labels, skip_special_tokens=True)
            # print(decoded)
            # print(decoded)
            labels_list = custom_tokenizer.encode_batch(decoded)
            # print([x.ids for x in labels_list])
            # print(labels_list)
            # labels_list = custom_tokenizer.batch_encode_plus(decoded, padding=True)
            # labels_data = [torch.tensor(x) for x in labels_list["input_ids"]]
            labels_data = [torch.tensor(x.ids) for x in labels_list]
            target_lengths = torch.tensor([x.count_nonzero() for x in labels_data])
            # print(labels_data)
            # labels = torch.stack(labels_data, dim=0)
            # labels_mask = labels >= 1
            # target_lengths = labels_mask.sum(-1)
            # print(target_lengths.size())
            # print(target_lengths)
            labels = torch.cat(labels_data, dim=0)
            labels_mask = labels >= 1
            flattened_targets = labels.masked_select(labels_mask)
            # print(flattened_targets)
            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(
                logits, dim=-1, dtype=torch.float32
            ).transpose(0, 1)
            # print(log_probs.size())
            # print(flattened_targets.size())
            # print(input_lengths)
            # print(target_lengths)
            # assert(False)
            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=0,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )