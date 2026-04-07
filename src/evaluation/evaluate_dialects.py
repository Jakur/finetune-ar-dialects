from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_from_disk
import argparse
from whisper_utils import DataCollatorSpeechSeq2SeqWithPadding, compute_metrics
import json
import os
from transformers import EarlyStoppingCallback

if __name__ == "__main__":
    root = ""
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", required=True)
    args = parser.parse_args()

    model = WhisperForConditionalGeneration.from_pretrained(args.checkpoint)

    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="Arabic", task="transcribe"
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3, early_stopping_threshold=0.001
    )

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = "ar"
    model.config.max_length = 512

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir="evaluation",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=5000,
        gradient_checkpointing=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=250,
        eval_steps=250,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )
    all_results = {}
    model_name = str(args.checkpoint).split("/")[-1]
    for d in ["egyptian", "gulf", "iraqi", "levantine", "maghrebi"]:
        if d != "maghrebi":
            continue
        test_set = load_from_disk("/media/justin/SSD Ubuntu Stora/datasets/egypt_test")["train"]
        # test_set = load_from_disk(os.path.join(root, f"{d}_test/"))
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            eval_dataset=test_set,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
        )
        results = trainer.evaluate(language="ar")
        all_results[d] = results.copy()
        with open(f"results_{model_name}.json", "w") as f:
            json.dump(all_results, f)

    assert(False) 
    test_set = load_from_disk(os.path.join(root, "test"))
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        eval_dataset=test_set,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[early_stopping_callback],
    )
    results = trainer.evaluate(language="ar")
    all_results["MSA"] = results.copy()
    with open(f"results_{model_name}.json", "w") as f:
        json.dump(all_results, f)
