import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import argparse
from tag_loader import TagDataset
import sys
import yaml
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
from STEMTOX.model.model import MultiTaskingLlavaModel, MultiTaskingPaligemmaModel
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = "cuda" if torch.cuda.is_available() else "cpu"
from huggingface_hub import login
login(token="")


def main(**args):
    
    with open(args.dataset_config, 'r') as f:
        config = yaml.safe_load(f)
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    dataset = pd.read_csv(config['dataset_path'])
    num_classes = config['num_classes']
    class_counts = torch.tensor(list(config['class_counts'].values()), dtype=torch.float)
    assert num_classes == len(len(list(config['class_counts'].values()))), "Number of Classes and class counts both are different!!"
    dataset[config['label']] = dataset['label'].replace(config['label_dic'])
    finetuned_model_path = os.path.join(args.output_dir, f"./finetuned_{model}_{args.strategy}_{config['dataset_tag']}_{args.frame_wrok_type}")
    if not os.path.exists(finetuned_model_path):
        os.makedirs(finetuned_model_path)
    train_df, test_df = train_test_split(dataset, test_size=0.1, random_state=42, shuffle=True)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    train_data = TagDataset(train_df, dataset_tag=config['dataset_tag'])
    test_data = TagDataset(test_df, dataset_tag=config['dataset_tag'])
    
    if 'paligemma' in model_config['model']:
        model = MultiTaskingPaligemmaModel(strategy=args.strategy, class_counts=class_counts, num_classes=num_classes)
    elif 'llava' in model_config['model']:
        model = MultiTaskingLlavaModel(strategy=args.strategy, class_counts=class_counts, num_classes=num_classes)

    model.use_entropy_minimization = True
    if args.frame_wrok_type == "last_layer":
        model.use_entropy_minimization = False   

    training_args = TrainingArguments(
        num_train_epochs=model_config['epochs'],
        remove_unused_columns=False,
        per_device_train_batch_size=model_config['per_device_train_batch_size'],
        gradient_accumulation_steps=model_config['gradient_accumulation_steps'],
        gradient_checkpointing=False,
        warmup_steps=model_config['warmup_steps'],
        learning_rate=model_config['learning_rate'],
        weight_decay=model_config['weight_decay'],
        adam_beta2=model_config['adam_beta2'],
        logging_steps=model_config['logging_steps'],
        optim=model_config['optim'],
        resume_from_checkpoint=True,
        save_strategy="steps",
        save_steps=model_config['save_steps'],
        output_dir=finetuned_model_path,
        save_total_limit=model_config['save_total_limit'],
        fp16=model_config['fp16'],
        save_safetensors=model_config['save_safetensors'],
        report_to=["tensorboard"],
        dataloader_pin_memory=model_config['dataloader_pin_memory'],
    )


    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=model.collate_fn,
        args=training_args,
    )
    trainer.train()
    final_model_dir = os.path.join(finetuned_model_path, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    model.save_pretrained(final_model_dir)
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune STEMTOX")
    parser.add_argument(
        "--dataset_config", 
        type=str, 
        default="../config/hatexplain_config.yaml",
        help="Path to the dataset YAML configuration file"
    )
    parser.add_argument(
        "--model_config", 
        type=str, 
        default="../config/qwen_config.yaml",
        help="Path to the model YAML configuration file"
    )
    parser.add_argument(
        "--frame_wrok_type", 
        type=str, 
        default="entropy",
        help="Enter the framework type: "
    )
    parser.add_argument(
        "--strategy", 
        type=str, 
        default="finetuned",
        help="Enter the strategy type: "
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./",
        help="Enter the output directory: "
    )
    parser.add_argument(
        "--hf_token", 
        type=str, 
        default="#####",
        help="Enter the huggingface_token: "
    )
    args = parser.parse_args()
    login(token=args.hf_token)
    main(args=args)
    
    