<h1 align="center">The Problem: "Detecting Online Grooming By Simple Contrastive Chat Embeddings"</h1>






This code proposes a contrastive learning framework for feature extraction in a sentence-based manner, where it can assign a feature vector to the conversation with misspellings using subword information.
It proposes a configuration of RoBERTa encoders and supervised SimCSE for training the SVM model, leading to a high rate of detecting relevant samples (predatory samples).
https://dl.acm.org/doi/abs/10.1145/3579987.3586564

<h3>The overall pipeline </h3>

- My approach comprises several components, starting with the **data preprocessing,** followed by a **feature extractor** based on
**pre-trained network of simple contrastive learning that extracts sentence embeddings (SimCSE)**, and a **classification model**. I use Simple Contrastive Sentence Embedding framework (SimCSE) based on a contrastive objective with pre-trained language models to extract the features, and an SVM for classifying the chat conversations
The overall pipeline of the proposed system is presented in Figure 1 below:
<img src="https://github.com/prezaeeb/SimCSE_Embeddings/blob/800af5fa6213ce47f68426c56c414132aa90b614/ProposedModel.png" alt="Sample Image" width="800" height="700">



<h3>The Data </h3>
The applied data has various conversations from online platforms. Mainly, three different conversations are
</h2> 

- a typical conversation that does not have any sexual topic,
  
- a conversation on sexual topics between adults, and
  
- a conversation between a child molester and a minor victim.

<h3>The Pre-Processing </h3>  
We performed labeling based on if a predator id is seen in a conversation as an author, that conversation will be tagged as a predatory sample and vice versa (see Figure 4). Also, considering that a predatory conversation always has two authors, we removed the samples with more than two users or only one user. Further, all conversations with less than seven messages were eliminated since they did not provide enough information to be classified. As another refinement, we removed non-English words with no special meanings. No stemming or lemmatization in the pre-processing of the data was performed to keep as much information as possible.
<img src="https://github.com/prezaeeb/SimCSE_Embeddings/blob/994575e3130a8ded09fdc875aea1f396fe5d1566/PreProcessing.png" alt="Sample Image" width="700" height="500">


<h3>The Feature Extraction </h3>  
</h2>  

- The Sentence embeddings (Features) in SimCSE are extracted using both supervised and unsupervised approaches.https://github.com/princeton-nlp/SimCSE 
  
- In an unsupervised approach, the same sentence is passed twice to the pre-trained network. The standard dropout is used twice for each sentence to gain two different embeddings as positive pairs. Then, the other sentences in the same mini-batch are considered negatives to make the model predict the positive sentence among the negative ones.

- In the supervised approach, two sentences are entailment pairs if they are related semantically. For instance, consider the "Two dogs are running" as the main sentence; its entailment can be "There are animals outdoors. ", and its contradiction sentence can be "The pets are sitting on a couch.". The sentence and its entailment pairs are considered positive samples. To improve the performance, a contradiction sentence is also given to the pre-trained encoder as the negative sample.


  <h3>The classifier </h3>
  The SVM model takes the embeddings as feature sets and train and test datasets, and predict the grooming conversations. 

  <h3>Grooming Conversation Detection:</h3>
    
  - In this work, we focus on **semantic analysis** of grooming chatlogs where the proper feature space covers the meanings behind the sentences and phrases in chat conversations. As such, we produce the feature sets based on a simple contrastive sentence embedding framework (SimCSE). In other words, **we use a SimCSE pretrained network to extract the embeddings for each conversation in a sentence-based manner rather than one entity, such as a word or a token**.


  <h3>Performance Metrics</h3>

  - The grooming detection is considered a **two-class classification** problem where we aim to distinguish predatory conversations from non-predatory ones. This research work applies the standard performance metrics containing **Accuracy(Acc)**, **Precision(Pr)**, **Recall(Re)**, and **F-score** (a weighted harmonic mean between precision and recall).
 
  <h3>Requirements: CPU & GPU versions</h3>

  -Given the scale of the dataset, this codebase is designed for parallel processing. For efficient training and inference, it is highly recommended to utilize hardware acceleration, such as a **GPU**, or a **scalable cloud service** like **Azure**.

  -To accommodate different hardware, both a **CPU-optimized** and a **GPU-optimized** version of the code have been provided. The GPU version is ideal for running on **cloud environments** like **Azure** for faster processing.


