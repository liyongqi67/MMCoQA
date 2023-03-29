# MMCoQA
MMCoQA aims to answer users’ questions with multimodal knowledge sources via multi-turn conversations.

For more details check out our ACL2022 paper "[MMCoQA: Conversational Question Answering over Text, Tables, and Images](https://aclanthology.org/2022.acl-long.290/)".


# Dataset download
The MMCoQA dataset can be downloaded via this [link](https://drive.google.com/drive/folders/1ErP9sjKYKxP76B18mjAyDnOTPn08emZD?usp=sharing).

# Dataset format
In the [dataset](https://drive.google.com/drive/folders/1ErP9sjKYKxP76B18mjAyDnOTPn08emZD?usp=sharing) folder you will find the following file question and contexts files:
1) `MMCoQA_train.text,MMCoQA_dev.text,MMCoQA_test.text` - contains questions, evidence and answers, for train, dev and test set respectively
Each line is a json format, that contains one question, the gold question, the conversation history, alongside its answers.
2) `qrels.txt` - contains the question and its relevant evidence (one document, table, or image) label.
3) `multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl` - contains metadata (id, title, path) for each image. You could load one image according to its `path` from the final_dataset_images.zip data.
4) `multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl` - Each line  represents a single table. `header` provides the column names in the table.  `table_rows` is a list of rows, and each row contains is a list of table cells. Each cell is provided with its text string and link (if the text could be clicked in Wikipedia page). 
5) `multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl` - contains metadata (id, title, text) for each document.

# Reporduce the MAE(pretrain) in our paper
1. Download the MMCoQA dataset and uncompress the zip file (including the final_dataset_images.zip).
2. MAE(pretrain) use the pretrained text based CoQA checkpoint from the work ORConvQA. Please download it from this link and move it into the retriever_checkpoint file.
3. Directly run train.retriever.py.



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
