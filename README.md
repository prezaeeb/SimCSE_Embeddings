<h1 align="center">The Problem: "Detecting Online Grooming By Simple Contrastive Chat Embeddings"</h1>






This code proposes a contrastive learning framework for feature extraction in a sentence-based manner, where it can assign a feature vector to the conversation with misspellings using subword information.
It proposes a configuration of RoBERTa encoders and supervised SimCSE for training the SVM model, leading to a high rate of detecting relevant samples (predatory samples).
https://dl.acm.org/doi/abs/10.1145/3579987.3586564

<h3>The overall pipeline </h3>

- My approach comprises several components, starting with the **data preprocessing,** followed by a **feature extractor** based on
**pre-trained network of simple contrastive learning that extracts sentence embeddings (SimCSE)**, and a **classification model**. I use Simple Contrastive Sentence Embedding framework (SimCSE) based on a contrastive objective with pre-trained language models to extract the features, and an SVM for classifying the chat conversations
The overall pipeline of the proposed system is presented in Figure 1 below:
![Image Alt](https://github.com/prezaeeb/SimCSE_Embeddings/blob/800af5fa6213ce47f68426c56c414132aa90b614/ProposedModel.png)


<h3>The Data </h3>
The applied data has various conversations from online platforms. Mainly, three different conversations are
</h2> 

- a typical conversation that does not have any sexual topic,
  
- a conversation on sexual topics between adults, and
  
- a conversation between a child molester and a minor victim.

<h3>The Pre-Processing </h3>  
We performed labeling based on if a predator id is seen in a conversation as an author, that conversation will be tagged as a predatory sample and vice versa (see Figure 4). Also, considering that a predatory conversation always has two authors, we removed the samples with more than two users or only one user. Further, all conversations with less than seven messages were eliminated since they did not provide enough information to be classified. As another refinement, we removed non-English words with no special meanings. No stemming or lemmatization in the pre-processing of the data was performed to keep as much information as possible.
