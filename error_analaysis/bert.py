import pandas as pd
from bertopic import BERTopic
import ast
from sklearn.feature_extraction.text import CountVectorizer

count = 0
def parse_tags(x):
    global count
    try:
        val = ast.literal_eval(x)
        return  val
    except Exception:
        count += 1
        return x

class BertTagErrorClusterer:
    def __init__(self, dataset_path, tag_column='generated_tags'):
        self.error_df = pd.read_csv(dataset_path)
        self.error_df[tag_column] = self.error_df[tag_column].apply(parse_tags)
        self.tag_column = tag_column
        print(self.error_df.shape[0])
        
        self.topic_model = None

    def run_clustering(self, min_topic_size=4):
       

        if self.error_df.empty:
            print("No valid samples left to cluster after filtering short tags.")
            return self.error_df

        docs = self.error_df[self.tag_column].apply(lambda x: " ".join(map(str, x))).tolist()

        self.topic_model = BERTopic(
            language="english",
            min_topic_size=min_topic_size,
            calculate_probabilities=True,
            nr_topics=4
        )
        topics, _ = self.topic_model.fit_transform(docs)
        
        topic_labels = self.topic_model.generate_topic_labels(nr_words=3, separator="_")
        
        unique_topics = sorted(list(set(topics)))
        topic_map = {tid: label for tid, label in zip(unique_topics, topic_labels)}
        
        self.error_df['topic_id'] = topics
        self.error_df['topic_name'] = [topic_map[t] for t in topics]

        return self.error_df

error_pd_path = input("Enter the path:\n")
model_name = input("Enter the model name:\n")
frame_work = input("Enter the framework name:\n")
approach = input("Enter the approach name:\n")

clusterer = BertTagErrorClusterer(error_pd_path)
df_results = clusterer.run_clustering()

output_path = f"./{model_name}_{frame_work}_{approach}_bert_clustered_errors.csv"
df_results.to_csv(output_path, index=False)
print(df_results.shape)