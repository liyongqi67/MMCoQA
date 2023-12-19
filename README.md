# MMCoQA
MMCoQA aims to answer users’ questions with multimodal knowledge sources via multi-turn conversations.

For more details check out our ACL2022 paper "[MMCoQA: Conversational Question Answering over Text, Tables, and Images](https://aclanthology.org/2022.acl-long.290/)".


# Dataset download
The MMCoQA dataset can be downloaded via this [link](https://drive.google.com/drive/folders/1ErP9sjKYKxP76B18mjAyDnOTPn08emZD?usp=sharing).

# Dataset format
In the [dataset](https://drive.google.com/drive/folders/1ErP9sjKYKxP76B18mjAyDnOTPn08emZD?usp=sharing) folder you will find the following file question and contexts files:
1) `MMCoQA_train.text,MMCoQA_dev.text,MMCoQA_test.text` - contains questions, evidence and answers, for train, dev and test set respectively.
Each line is a json format, that contains one question, the gold question, the conversation history, alongside its answers.
2) `qrels.txt` - contains the question and its relevant evidence (one document, table, or image) label.
3) `multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl` - contains metadata (id, title, path) for each image. You could load one image according to its `path` from the final_dataset_images.zip data.
4) `multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl` - Each line  represents a single table. `header` provides the column names in the table.  `table_rows` is a list of rows, and each row contains is a list of table cells. Each cell is provided with its text string and link (if the text could be clicked in Wikipedia page). 
5) `multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl` - contains metadata (id, title, text) for each document.

# Reporduce the MAE(pretrain) in our paper
1) Download the MMCoQA dataset and uncompress the zip file (including the final_dataset_images.zip).

2) MAE(pretrain) use the pretrained text based CoQA checkpoint from the work ORConvQA to initialize the text encoder. Please download it (checkpoint-5917) from this [link](https://drive.google.com/file/d/15d7xPEZCIkN4m7Pov6ZPBVjWlEsy9Q8p/view?usp=sharing) and move it into the retriever_checkpoint folder.

3) Directly run train_retriever.py by setting the corresponding file paths.

```
python3 train_retriever.py 
--train_file MMCoQA_train.txt --dev_file MMCoQA_dev.txt --test_file MMCoQA_test.txt \
--passages_file multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl \
--multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl \
--images_file multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl \
--images_path final_dataset_images \
--retrieve_checkpoint ./retriever_checkpoint/checkpoint-5917
```
This script will store the checkpoints under the retriever_release_test folder. For your convenience, we upload our results (the checkpoint 'checkpoint-5061') in this [link](https://drive.google.com/file/d/1549wBJt8lgU19a_TM9K-GjkEL5GRbV38/view?usp=sharing).

4. Generate embedding of evidence via setting 'retrieve_checkpoint' as the corresponding checkpoint file under the retriever_release_test folder.
```
python3 train_retriever.py 
--gen_passage_rep True \
--retrieve_checkpoint ./retriever_release_test/checkpoint-5061 \
--train_file MMCoQA_train.txt --dev_file MMCoQA_dev.txt --test_file MMCoQA_test.txt \
--passages_file multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl \
--multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl \
--images_file multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl \
--images_path final_dataset_images \
```
  This script will store the embeddings of docs in the ./retriever_release_test/dev_blocks.txt file. For your convenience, we upload the dev_blocks.txt generated via the 'checkpoint-5061' in this [link](https://drive.google.com/file/d/1549wBJt8lgU19a_TM9K-GjkEL5GRbV38/view?usp=sharing).

5. Run the train_pipeline.py.
```
python3 train_pipeline.py 
--train_file MMCoQA_train.txt --dev_file MMCoQA_dev.txt --test_file MMCoQA_test.txt \
--passages_file multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl \
--multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl \
--images_file multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl \
--images_path final_dataset_images \
--gen_passage_rep_output ./retriever_release_test/dev_blocks.txt \
--retrieve_checkpoint ./retriever_release_test/checkpoint-5061 \
```
  This script will train the pipeline (retriever and answer extraction components) and store the checkpoints in release_test folder. For your convenience, we upload our results 'checkpoint-12000' in this [link](https://drive.google.com/file/d/1HW__WoZ13qqtPrw8t-bLb9eTEHzjsDJ0/view?usp=sharing). 
  
  You could load the checkpoint and test it via:
```
python3 train_pipeline.py 
--do_train False --do_eval False --do_test True --best_global_step 12000 \
--train_file MMCoQA_train.txt --dev_file MMCoQA_dev.txt --test_file MMCoQA_test.txt \
--passages_file multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl \
--multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl \
--images_file multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl \
--images_path final_dataset_images \
--gen_passage_rep_output ./retriever_release_test/dev_blocks.txt \
--retrieve_checkpoint ./retriever_release_test/checkpoint-5061 \
```
  The checkpoint 'checkpoint-12000' could achieve the results in the paper:
  
  Dev set: {"f1": 4.586325704965762, "EM": 0.020654044750430294, "retriever_ndcg": 0.07209415622067673, "retriever_recall": 0.42168674698795183}
  
  Test set: {"f1": 3.584818194987687, "EM": 0.0288135593220339, "retriever_ndcg": 0.07655641068040793, "retriever_recall": 0.4271186440677966}
  
  The file of the prints while running train_pipeline.py is in this [link](https://drive.google.com/file/d/1Zk3zAibxzfONZvD4bUL_cYoF30xs1Gbw/view?usp=sharing).
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
If there is any problem, please email liyongqi0@gmail.com. Please do not hesitate to email me directly as I do not frequently check GitHub issues.
