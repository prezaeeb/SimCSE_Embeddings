



The Problem: "Detecting Online Grooming By Simple Contrastive Chat Embeddings"


This code proposes a contrastive learning framework for feature extraction in a sentence-based manner, where it can assign a feature vector to the conversation with misspellings using subword information.
It proposes a configuration of RoBERTa encoders and supervised SimCSE for training the SVM model, leading to a high rate of detecting relevant samples (predatory samples).
https://dl.acm.org/doi/abs/10.1145/3579987.3586564

The overall pipeline of the proposed system is presented in Figure 1 below:
![Image Alt](https://github.com/prezaeeb/SimCSE_Embeddings/blob/800af5fa6213ce47f68426c56c414132aa90b614/ProposedModel.png)


3 DATA
The applied data has various conversations from online platforms.
Mainly, three different conversations are
• a typical conversation that does not have any sexual topic,
• a conversation on sexual topics between adults, and
• a conversation between a child molester and a minor victim.
