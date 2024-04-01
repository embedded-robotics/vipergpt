from main_simple_lib import *
import pickle
import csv

# Specifying the datapaths for PVQA dataset and loading the pickle file
img_train_path = "/data/mn27889/pvqa/images/train/"
qas_train_path = "/data/mn27889/pvqa/qas/train/train_qa.pkl"
with open(qas_train_path, 'rb') as file:
    pvqa_qas = pickle.load(file)

# Reading only yes/no questions
qas_yes_no = [qas for qas in pvqa_qas if qas['answer'] == 'yes' or qas['answer'] == 'no']
ques_yes_no = [qas['question'] for qas in qas_yes_no]
ans_yes_no = [qas['answer'] for qas in qas_yes_no]
img_yes_no_path = [img_train_path + qas['image'] + '.jpg' for qas in qas_yes_no]

# Reading a single image, question and answer line-by-line to get the answers from GPT
total_yes_no = len(qas_yes_no)

data_index = [idx for idx in range(0, total_yes_no)]

for idx in data_index[365:len(data_index)]:
    query = ques_yes_no[idx]
    query_ans = ans_yes_no[idx]
    query_img = load_image(img_yes_no_path[idx])
    
    code = get_code(query)
    predicted_ans = execute_code(code, query_img, show_intermediate_steps=False)
    
    with open('pvqa_yes_no.csv', 'a', newline='') as csvfile:
        pvqa_writer = csv.writer(csvfile)
        pvqa_writer.writerow([idx, query, query_ans, img_yes_no_path[idx], predicted_ans])