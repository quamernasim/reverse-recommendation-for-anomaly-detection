# Reverse Recommendations for Anomaly Detection: Fraud Detection System with Qdrant and RAG
<img src="assets/Fraud Detection.png" alt="Fraud Detection"/>

This repository demonstrates how to build an anomaly detection system using reverse recommendations. Specifically, we use Qdrant to create a vector store of normal transactions and apply a RAG (Retrieval-Augmented Generation) based approach to explain anomalous transactions, such as fraudulent activity. The system compares new transactions to a baseline of normal behavior and flags transactions that deviate from expected patterns.

Key Features:
- Fraud Detection: Detect fraudulent transactions by comparing them against typical user behavior using vector search.
- Similarity Search: Use Qdrant for nearest-neighbor search to find the most dissimilar (anomalous) transactions.
- Explainability with RAG: Combine reverse recommendation with a RAG model to not only detect fraud but also provide reasons for flagging a transaction as fraudulent.

## Dataset
We use a credit card transaction dataset that includes various features such as:

- Customer Basic Information
- Customer Residence Information
- Merchant Information
- Transaction Information
- Transaction Location Information

The dataset is highly imbalanced, with only 0.58% of transactions labeled as fraud. This imbalance presents a challenge for traditional supervised learning techniques, making this anomaly detection system more effective.

You can download the dataset from this [link](https://www.kaggle.com/datasets/kartik2112/fraud-detection).

## Approach
### Reverse Recommendations
Reverse recommendations are utilized to detect outliers or anomalies in the dataset. The core idea is to compare new transactions to a baseline of normal transactions for each user and flag transactions that deviate significantly from normal behavior.

### Vector Search with Qdrant
We use Qdrant, a vector search engine, to store vector embeddings of genuine transactions. These embeddings are derived from transaction features such as amount, merchant details, and location data, excluding customer-specific information to avoid inflating similarity scores.

The new transactions are also converted into embeddings, and Qdrant searches for the most similar transactions. If the new transaction is too dissimilar (based on a cosine similarity threshold), it is flagged as fraudulent.

### Anomaly Detection
The system flags transactions as anomalies if their similarity score is below a predefined threshold (e.g., 95%). The lower the similarity score, the further away the transaction is from normal behavior in the vector space, which increases the likelihood of fraud.

### RAG for Explainability
To explain why a transaction is flagged as fraudulent, we use a RAG model. The RAG model combines the vector embeddings from Qdrant with a language model that provides explanations based on transaction descriptions and user information. This helps customer support agents take appropriate actions when suspicious transactions are detected.

- Context Generation: The nearest transactions from the Qdrant database are retrieved and used to provide context for the RAG model.
- LLM Integration: The system uses prompts to guide the model in explaining why an anomaly was detected.

## Running the Code
### Requirements
- Python 3.10.14

### Installation of Dependencies
```bash
pip install -r requirements.txt
```

### Dis-Similarity Based Anomaly Detection
```bash
python src/dissimilarity_based_fraud_detection.py --root {path_to_root} --data_path {path_to_data} --embedding_model_id {model_id} --max_transactions {max_transactions} --new_transaction_info {new_transaction_info} --user {user_id}
```

### RAG Based Explainable Anomaly Detection
```bash
python src/rag_based_fraud_detection.py --root {root} --data_path {data_path} --embedding_model_id {embedding_model_id} --max_transactions {max_transactions} --new_transaction_info {new_transaction_info} --user {user} --model_id {model_id}
```

## Conclusion
In this repository, we've combined reverse recommendations with vector search and RAG to detect and explain fraudulent transactions. This approach can be extended to other anomaly detection scenarios, such as insurance claims, customer interactions, or network traffic monitoring.

Key takeaways:
- Reverse recommendation helps spot outliers efficiently.
- Vector search simplifies anomaly detection.
- The RAG model adds transparency by explaining flagged anomalies.