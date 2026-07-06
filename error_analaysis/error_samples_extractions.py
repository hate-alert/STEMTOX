# import pandas as pd
# import os
# import ast
# dataset = pd.read_csv("/home/subhankar-am/ToxicTagsPhase2/dataset/ToxicTagsDataset - stage1_test.csv")

# path = input("Enter the path of the csv file:")
# df = pd.read_csv(path)

# dir_name = os.path.dirname(path)
# file_name = os.path.basename(path)

# output_path = os.path.join(dir_name, f"error_{file_name}")

# df = df[df['pred_label_name'] != dataset['stage2_label']]

# df.to_csv(output_path, index=False)
# print(f"File saved successfully at: {output_path}")



import pandas as pd
import ast
import re




class ErrorAnalysis:
    def __init__(self, predicted_dataset_path, dataset_path, true_label_name, pred_label_name, generated_tags, model="paligemma"):
        self.dataset = pd.read_csv(predicted_dataset_path)
        self.true_label_name = true_label_name
        self.pred_label_name = pred_label_name
        self.model = model
        self.generated_tags = generated_tags
        self.dataset[self.true_label_name] = pd.read_csv(dataset_path)[self.true_label_name]
        self.dataset = self.dataset[self.dataset[self.true_label_name]!=self.dataset[self.pred_label_name]].reset_index(drop=True)
        if model == "paligemma":
            # self.dataset[self.pred_label_name] = self.dataset[self.pred_label_name].apply(lambda x: ast.literal_eval(x)[0]).apply(ast.literal_eval)
            self.dataset[self.generated_tags] = self.dataset[self.generated_tags].apply(self.parse_tags)
        else:
            self.dataset[self.generated_tags] = self.dataset[self.generated_tags].apply(self.parse_tags)
            print(self.dataset.shape[0])
        self.dataset = self.dataset.dropna(subset=[self.generated_tags]).reset_index(drop=True)
        print(self.dataset.shape[0])
        
        self.freq_tags = {}
    
    def parse_tags(self, x):
        if self.model=="llava":
            try:
                val = ast.literal_eval(x)
                return  val
            except Exception:
                return None
        else:
            if pd.isna(x):
                return []

            x = str(x).strip()
            try:
                outer = ast.literal_eval(x)
                if isinstance(outer, list) and len(outer) == 1:
                    x = outer[0]
            except Exception:
                pass
            try:
                inner = ast.literal_eval(x)
                if isinstance(inner, list):
                    return [str(t).strip() for t in inner if str(t).strip()]
            except Exception:
                pass
            x = re.sub(r'^\[|\]$', '', x)
            tags = re.split(r',|\n', x)

            return [t.strip(" '\"\t") for t in tags if t.strip()]
        
            
            
    def frequency_error_calc(self):
        for index in range(self.dataset.shape[0]):
            for tag in self.dataset.at[index, self.generated_tags]:
                self.freq_tags[tag] = self.freq_tags.get(tag, 0) + 1
            
        return sorted(
                self.freq_tags.items(),
                key=lambda x: x[1],
                reverse=True
            )

model = 'llava'
# model = "paligemma"
error_data = ErrorAnalysis(
    predicted_dataset_path = f"/home/subhankar-am/ToxicTagsPhase2/frameworks/{model}/entropy_guided_framework/inference_results_stage_2.csv",
    dataset_path="/home/subhankar-am/ToxicTagsPhase2/dataset/test.csv",
    true_label_name='stage2_label',
    pred_label_name='pred_label_name',
    generated_tags="cleaned_text",
    # generated_tags="generated_tags",
    model=model
)
                        
df = pd.DataFrame(
    error_data.frequency_error_calc(),
    columns=["tag", "count"]
)
df.to_csv(f"./{model}_error.csv", index=False)