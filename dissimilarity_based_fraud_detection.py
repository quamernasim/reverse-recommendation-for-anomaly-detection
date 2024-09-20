import time
import argparse
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

def main(args):
    root = args.root
    data_path = args.data_path
    embedding_model_id = args.embedding_model_id
    user = args.user
    max_transactions = args.max_transactions
    new_transaction_info = args.new_transaction_info

    # Load the data
    data, _ = load_data(root, data_path, random=False)

    # Load the pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_id)
    model = AutoModel.from_pretrained(embedding_model_id)
    model.eval()

    user_data = data[data['cc_num'] == user]
    # Filter out the fraud transactions
    user_data = user_data[user_data['is_fraud'] == 0]

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

        if idx == max_transactions:
            break

    # Embed the new transaction information
    new_embedding = embed_transaction(new_transaction_info, model, tokenizer)
    results = get_as_close_transactions(new_embedding, client)

    if detect_anomalies(results):
        print("The new transaction is fraudulent")
    else:
        print("The new transaction is genuine")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--root", type=str, default='/home/quamer23nasim38/reverse-recommendation-for-anomaly-detection/')
    argparser.add_argument("--data_path", type=str, default='data/fraudTrain.csv')
    argparser.add_argument("--embedding_model_id", type=str, default="BAAI/bge-small-en")
    argparser.add_argument("--max_transactions", type=int, default=100)
    argparser.add_argument("--new_transaction_info", type=str, default='420000.54\n-----------------------\nRajesh, Kumar; savings_account\n-----------------------\nChandini Chowk; Delhi; India; 20.0583; 16.008\n-----------------------\nVietnaam; 20.152538; 16.227746')
    argparser.add_argument("--user", type=int, default=213161869125933)
    args = argparser.parse_args()

    main(args)