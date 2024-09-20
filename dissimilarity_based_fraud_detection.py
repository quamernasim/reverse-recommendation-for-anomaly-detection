import time
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient, models

from utils import (
    convert_transaction_data_to_str,
    get_transactional_data,
    embed_transaction,
    insert_transaction,
    get_as_close_transactions,
    load_data,
    detect_anomalies
)

root = '/home/quamer23nasim38/reverse-recommendation-for-anomaly-detection/'
data_path = 'data/fraudTrain.csv'

data, random_cc_num = load_data(root, data_path)




# Load the pre-trained model
embedding_model_id = "BAAI/bge-small-en"
tokenizer = AutoTokenizer.from_pretrained(embedding_model_id)
model = AutoModel.from_pretrained(embedding_model_id)
model.eval()
print("Model loaded successfully")

for user in random_cc_num:
    # Get the data for the user
    user_data = data[data['cc_num'] == user]
    # Filter out the fraud transactions
    user_data = user_data[user_data['is_fraud'] == 0]
    if user_data.shape[0]>1500:
        # get the user data which has at most 1500 transactions.
        # This can be changed to any number as per the requirement
        break


# Initialize in-memory Qdrant client
client = QdrantClient(":memory:")

# Create a collection in Qdrant for storing transaction embeddings
client.create_collection(
    collection_name="transactions",
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
)

for idx, (_, transaction) in tqdm(enumerate(user_data.iterrows()), total=len(user_data)):
    # get the transactional information for a particular transaction
    transaction_information, merchant_information, payment_address, merchant_address = get_transactional_data(transaction, convert_coordinates_to_address=True)
    # convert the transaction information to string
    transaction_description = convert_transaction_data_to_str(transaction_information, merchant_information, payment_address, merchant_address)
    # embed the transaction description
    embedding = embed_transaction(transaction_description, model, tokenizer)
    embedding = embedding[0].tolist()
    # upload the transaction embedding and data to the qdrant client
    insert_transaction(embedding, transaction_description, idx, client)
    time.sleep(1)

    if idx == 200:
        break

new_transaction_info = '''
420000.54
-----------------------
Rajesh, Kumar; savings_account
-----------------------
Chandini Chowk; Delhi; India; 20.0583; 16.008
-----------------------
Vietnaam; 20.152538; 16.227746
'''

# Embed the new transaction information
new_embedding = embed_transaction(new_transaction_info, model, tokenizer)
results = get_as_close_transactions(new_embedding, client)

if detect_anomalies(results):
    print("The new transaction is fraudulent")
else:
    print("The new transaction is genuine")