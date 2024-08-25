#!/usr/bin/env python

# coding: utf-8



# # compute the embeddings for Test data



# In[ ]:





import numpy as np

import csv

import pandas as pd



from simcse import SimCSE

import torch

from scipy.spatial.distance import cosine

from transformers import AutoModel, AutoTokenizer



tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

print('Model is loaded')

#********************

#READ the DATA

Test_data = pd.read_csv("/home/ubuntu/cluster/nbl-users/Parisa/mainDATA/Test_CLEAN2.csv", error_bad_lines=False, delimiter= '\t')

print('Data is loaded')

#********************

#Prepare the data for saving

data = Test_data.drop(['user1_MSG'], axis=1)

data = data.drop(['user2_MSG'], axis=1)

data = data.drop(['User1_ID'], axis=1)

data = data.drop(['User2_ID'], axis=1)



main_msg = data['conversations']
print('Data saving is ready')



#********************

#compute and save the features to a file

with open('/home/ubuntu/cluster/nbl-users/Parisa/parisacode/Test_Embeddings.csv', 'w', newline='') as f:

       

#len(main_msg)



    for i in range(len(main_msg)):
        print(i)
        inputs = tokenizer(main_msg[i], padding=True, truncation=True, return_tensors="pt")

    

        with torch.no_grad():

            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output

            writer = csv.writer(f, delimiter=',')

            writer.writerows(embeddings.numpy())

f.close()

#*********************

#Merge the features with their labels

features = pd.read_csv('/home/ubuntu/cluster/nbl-users/Parisa/parisacode/Test_Embeddings.csv', header=None,delimiter= ',')

data.drop("conversations", axis = 1, inplace = True)

result_features= pd.concat([data,features], axis=1)



        

#save results in a file



result_features.to_csv("/home/ubuntu/cluster/nbl-users/Parisa/parisacode/SIMCe_TEST_Embeddings.csv")



    



