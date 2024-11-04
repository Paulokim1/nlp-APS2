# Embedding-Based Retrieval System for SMS Dataset

This project implements a document retrieval system using embeddings, applied to an SMS spam classification dataset. It builds on by generating semantic embeddings with a pre-trained model, fine-tuning them, and performing similarity-based document retrieval. 

The goal is to improve search relevance by using embeddings that capture deeper semantic patterns in the data.

## Steps

### Step 1: Embeddings Generation

We generated embeddings for each message using the `sentence-transformers` model. The pre-trained model (sBERT) encodes semantic meaning, producing a vector representation of each SMS message. We fine-tuned these embeddings using a denoising autoencoder to adapt them to our dataset. Fine-tuning helped align the embedding space with the spam/ham classification task, making it more specific to our dataset.

### Step 2: Dimensionality Reduction with Autoencoder

The embeddings from the pre-trained model are high-dimensional. To reduce computational complexity, we used an autoencoder to lower the dimensionality. The autoencoder helped to remove noise while preserving semantic information. We trained it to minimize reconstruction loss, ensuring the embeddings retained meaningful structure for similarity search.

### Step 3: Visualizing Embeddings with t-SNE

We projected the high-dimensional embeddings into 2D space using t-SNE for visualization. This step helped assess how well the embeddings capture semantic relationships. We compared the pre-trained embeddings with the fine-tuned embeddings. The fine-tuned embeddings showed clearer clustering by spam and ham categories, indicating that the fine-tuning was effective.

### Step 4: Implementing the Retrieval System

The retrieval system calculates similarity between a query embedding and dataset embeddings. We used cosine similarity to measure the closeness between embeddings. For a given query, the system returns messages with the highest similarity scores, ranked in descending order. This method allows for efficient retrieval based on semantic similarity.

### Step 5: Testing the Retrieval System

We tested the system with three types of queries:
1. A query expected to yield 10 results (e.g., "You have won a free ticket to our prize draw!").
2. A query expected to yield fewer than 10 results (e.g., "Congratulations, you've been selected for a prize").
3. A non-obvious query to test broader retrieval (e.g., "Coffee is good for sleep").
