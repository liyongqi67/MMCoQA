# MMCoQA
MMCoQA aims to answer users’ questions with multimodal knowledge sources via multi-turn conversations.

For more details check out our ACL2022 paper "[MMCoQA: Conversational Question Answering over Text, Tables, and Images](https://aclanthology.org/2022.acl-long.290/)".


# Dataset download
The MMCoQA dataset can be downloaded via this [link](https://drive.google.com/drive/folders/1ErP9sjKYKxP76B18mjAyDnOTPn08emZD?usp=sharing).

# Dataset format
In the [dataset](https://drive.google.com/drive/folders/1ErP9sjKYKxP76B18mjAyDnOTPn08emZD?usp=sharing) folder you will find the following file question and contexts files:
1) `MMCoQA_train.text,MMCoQA_dev.text,MMCoQA_test.text` - contains questions, evidence and answers, for train, dev and test set respectively
Each line is a json format, that contains one question, the gold question, the conversation history, alongside its answers.
2) `qrels.txt` - contains the question and its relevant evidence (one document, table, or image).
3) `multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl` - contains metadata (id, title, path) for each image. You could load the image according to the `path` from the final_dataset_images.zip data.
4) `multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl` - Each line  represents a single table. table_rows is a list of rows, and each row contains is a list of cells. Each cell is provided with its text string and wikipedia entities. header provides for each column in the table: its name alongside parsing metadata computed such as NERs and item types. 
5) `multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl` - contains metadata (id, title, text) for each document.

# Environment Requirement
The code has been tested running under Python 3.8.8 The required packages are as follows:
- pytorch == 1.8.0
- pytrec-eval == 0.5
- faiss-gpu == 1.7.0
- transformers == 2.3.0
- tensorboard == 2.6.0
- tqdm == 4.59.0
- numpy == 1.19.2
- scipy == 1.6.2

# Contact
If there is any problem, please email liyongqi0@gmail.com
