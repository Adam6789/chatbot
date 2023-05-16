import pandas as pd
import torchtext
from torchtext.datasets import SQuAD1

def squad1_to_csv():
    # download and extract the dataset
    train_dataset, dev_dataset = SQuAD1()
    # convert the dataset to a list of dictionaries
    train_data = []
    for example in train_dataset:
        data = {
            "context": example[0],
            "question": example[1],
            "answer": example[2],
            "answer_start": example[3],
        }
        train_data.append(data)
    # convert the list to a pandas dataframe
    df = pd.DataFrame(train_data)
    df.to_csv("/home/workspace/data/train_dataset_squad1.csv")

    test_data = []
    for example in dev_dataset:
        data = {
            "context": example[0],
            "question": example[1],
            "answer": example[2],
            "answer_start": example[3],
        }
        test_data.append(data)
    # convert the list to a pandas dataframe
    df = pd.DataFrame(test_data)
    df.to_csv("/home/workspace/data/dev_dataset_squad1.csv")