from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_from_disk, concatenate_datasets, load_dataset
import argparse
import torch
import os
from transformers import EarlyStoppingCallback

torch.multiprocessing.set_sharing_strategy("file_system")
from whisper_utils import (  # noqa: E402
    DataCollatorSpeechSeq2SeqWithPadding,
    compute_metrics,
    TimingCallback,
    WhisperEncoderForCTC,
    ExtendedWhisperConfig,
    custom_tokenizer,
)

# home = "/media/justin/SSD Ubuntu Stora/datasets/huggingface"
# print(os.environ["HF_HOME"])
# print(os.environ["HF_HUB_CACHE"])
# print(os.environ["HF_XET_CACHE"])
# print(os.environ["HF_ASSETS_CACHE"])
# print(os.environ["TRANSFORMERS_CACHE"])
# os.environ["HF_HOME"] = home
# os.environ["HF_HUB_CACHE"] = f"{home}/hub"
# os.environ["HF_XET_CACHE"] = f"{home}/xet"
# os.environ["HF_ASSETS_CACHE"] = f"{home}/assets"
os.environ["TRANSFORMERS_CACHE"] = f"/media/justin/SSD Ubuntu Stora/fat_temp"

if __name__ == "__main__":
    root = ""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dialect",
        required=True,
        help="all, egyptian, gulf, iraqi, levantine, maghrebi",
    )
    args = parser.parse_args()
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=6, early_stopping_threshold=0.001
    )

    if args.dialect == "all":
        assert(False)
        # dialect_dataset = load_from_disk(os.path.join(root, "egyptian_train/"))
        # for d in ["gulf", "iraqi", "levantine", "maghrebi"]:
        #     train_d = load_from_disk(os.path.join(root, f"{d}_train/"))
        #     dialect_dataset = concatenate_datasets(
        #         [train_d, dialect_dataset]
        #     )
    else:
        dialect_dataset = load_dataset(f"otozz/{args.dialect}_train_set")["train"]
        # if args.dialect == "egyptian":
        #     dialect_dataset = load_from_disk("/media/justin/SSD Ubuntu Stora/datasets/egypt_train")["train"]
        # else:
        #     dialect_dataset = load_from_disk(os.path.join(root, f"{args.dialect}_train/"))
    # os.environ["TRANSFORMERS_CACHE"] = f"model_cache_{args.dialect}_finetune"
    # os.environ["HF_HOME"] = f"hf_cache_{args.dialect}_finetune"

    print(f"Training on {args.dialect} dialect, loaded from {dialect_dataset}")
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="Arabic", task="transcribe"
    )
    # processor = WhisperProcessor(WhisperFeatureExtractor.from_pretrained("openai/whisper-small"), tokenizer=custom_tokenizer)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # bs = 8
    bs = 4

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"whisper-small-feature_{args.dialect}",
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=1,
        learning_rate=1e-2,
        warmup_steps=2500,
        max_steps=15000,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        evaluation_strategy="steps",
        per_device_eval_batch_size=bs,
        predict_with_generate=False,
        generation_max_length=225,
        save_steps=5000,
        eval_steps=5000,
        logging_steps=100,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        fp16=True,
        fp16_full_eval=True,
        batch_eval_metrics=False
    )
    def preprocess_logits_for_metrics(logits, labels):
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids, labels
    
    for seed in [168]:
        print(f"Training with seed {seed}")       
        model = WhisperEncoderForCTC.from_pretrained("otozz/whisper-small-ar_tsize_1.0")
        # model = WhisperForConditionalGeneration.from_pretrained(
        #     "otozz/whisper-small-ar_tsize_1.0"
        # )
        print(f"Pad token id: {model.config.pad_token_id}")
        model.freeze_base_model()
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        # model.generation_config.language = "ar"
        model.config.max_length = 256 # 512
        train_test = dialect_dataset.train_test_split(test_size=0.05, seed=seed)
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=train_test["train"],
            eval_dataset=train_test["test"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            callbacks=[early_stopping_callback, TimingCallback(args.dialect, "finetune", seed)],
        )
        training_args.output_dir = f"whisper-small-finetune_{args.dialect}_seed{seed}"
        print(f"Output directory: {training_args.output_dir}")
        for i in range(10):
            try:
                trainer.train()
                break
            except Exception as e:
                print(f"Attempt {i + 1} failed with error: {e}")
            
            break
        print("----------------------------")
