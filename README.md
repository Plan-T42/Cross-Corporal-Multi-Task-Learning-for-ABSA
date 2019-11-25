Cross-Corporal Multi-Task Learning for Aspect-Based Sentiment Analysis

Literature:

[BERT](https://arxiv.org/abs/1810.04805)

[Joint Apsect and Polarity Classification](https://arxiv.org/abs/1808.09238)

[Transformer](https://arxiv.org/abs/1706.03762)/[annotated version with Python code](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

Andreas Pirchner | Farrukh Mushtaq | F
ranziska Sperl | Xinyi Tu

Sentiment Analysis refers to the automatic detection of the sentiment expressed in a piece of text, while Aspect-based Sentiment Analysis (ABSA) aims at a finer analysis i.e. it requires that certain aspects of an entity in question be distinguished in a given piece of text and the sentiment be classified with regard to each of them. Furthermore, we want to have a model that can capture the variation across multiple corpora for ABSA and make the model more adaptable to new data.

So in this work, we propose a new model to jointly train and test multiple corpora for the task of aspect-based sentiment analysis, in an effort to have data from other domains act as a kind of a regularizer for the domain data in focus. In contrast to other ABSA approaches, we jointly model the detection of aspects along with the classification of their polarity for all the given domains, in an end-to-end trainable neural network. We use BERT for text embeddings and conduct experiments with different neural architectures on multiple datasets. We weren’t able to see the regularization effect on the proposed model, which in all cases, had lower performance results than the simple baselines for ABSA on a single domain. Still, among the neural architectures tried, BiLSTM model on top of BERT text embeddings gave the best results.
