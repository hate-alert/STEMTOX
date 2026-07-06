from huggingface_hub import login
import torch
import pandas as pd
import os
import argparse
import yaml
from huggingface_hub import snapshot_download, login
from STEMTOX.model.model import MultiTaskingLlavaModel, MultiTaskingPaligemmaModel
login(token="")

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
def main(args):
    with open(args.dataset_config, 'r') as f:
        config = yaml.safe_load(f)
    if config['dataset']['dataset_tag'] == 'ToxicTags':
        dataset_path = snapshot_download(
                        repo_id=config['dataset']['original_dataset_path'],
                        repo_type="dataset"
                    )
        dataset = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    elif config['dataset']['dataset_tag'] == 'fhm':
        dataset = pd.read_csv(config['dataset']['original_dataset_path'])
    label_map = config['dataset']['label_map']
    dataset["img"] = dataset["img"].apply(lambda x: os.path.join(config['dataset']['img_folder_path'],x))
    num_classes = config['dataset']['num_classes']
    if 'paligemma' in args.model:
        model = MultiTaskingPaligemmaModel.from_pretrained(load_directory=args.model, 
                                                           num_classes=num_classes)
        model_file_name = 'paligemma'
    elif 'llava' in args.model:
        model = MultiTaskingLlavaModel.from_pretrained(load_directory=args.model,
                                                       num_classes=num_classes)
        model_file_name = 'llava'

    model.use_entropy_minimization = True# Enable entropy loop
    # model.eval()
    results = model.inference(dataset=dataset, label_map=label_map)
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(args.output_dir, f'results_{config['dataset']['dataset_tag']}_{model_file_name}.csv'), index=False)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune STEMTOX")
    parser.add_argument(
        "--dataset_config", 
        type=str, 
        default="./config/toxictags.yaml",
        help="Path to the dataset YAML configuration file"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="huggingface_model_path",
        help="Path to the hugging face directory file"
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
        default="freeze",
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
    
    