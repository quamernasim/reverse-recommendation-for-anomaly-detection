import time
from tqdm import tqdm

from utils import (
    convert_transaction_data_to_str,
    get_user_basic_info,
    get_transactional_data,
    embed_transaction,
    insert_transaction,
    get_context_for_anomaly_detection,
    load_data,
)

root = '/home/quamer23nasim38/reverse-recommendation-for-anomaly-detection/'
data_path = 'data/fraudTrain.csv'

data, random_cc_num = load_data(root, data_path)

from transformers import AutoTokenizer, AutoModel

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
        # Get the basic information for the user
        customer_information, registered_address = get_user_basic_info(user_data.iloc[0])
        print(f"Customer Information Loaded Successfully for {user}")
        break

from qdrant_client import QdrantClient, models

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

context = get_context_for_anomaly_detection(new_transaction_info, client, model, tokenizer, k=10)


system_prompt = '''
You're an intelligent AI assistant that helps in detecting fraudulent transactions. 

You're provided with the three key information:
    1. CUSTOMER INFORMATION: This has all the basic information about the customer which should give some idea about customer behaviour. The template is provided below.
    2. CONTEXT: This has several  examples of a normal and non-fraudulent transactional information for the user. The template for each transaction is provided below.
    3. NEW TRANSACTIONAL INFORMATION: This is the new transactional information that you need to classify as fraudulent or not. The template is same as normal transactional information 

Template for CUSTOMER INFORMATION and TRANSACTIONAL INFORMATION are provided below:
    1. CUSTOMER INFORMATION TEMPLATE
        {NAME}; {GENDER}; {AGE}; {JOB}
        -----------------------
        {REGISTERED ADDRESS}

    2. TRANSACTIONAL INFORMATION TEMPLATE: 
        {AMOUNT}
        -----------------------
        {MERCHANT NAME}; {CATEGORY}
        -----------------------
        {PAYMENT ADDRESS}
        -----------------------
        {MERCHANT ADDRESS} 

Your task is to uderstand USER's personal information, registered address, and examples of normal transactional information based on template provided and classify the new transactional information as fraudulent or not based on the context provided and also provide the reason for your classification.

You're only allowed to provide response in a json format with the following keys:
    1. classification: This should be either of the following:
        a. Fraudulent
        b. Non-Fraudulent
    2. reason: This should be a string explaining the reason for your classification.

Example of the response:
{
    "classification": "Fraudulent",
    "reason": "The transaction amount is significantly higher than the average transaction amount."
}
    
You can not provide any other response apart from the above mentioned json format with the keys mentioned above. In the classification key, you can only provide either "Fraudulent" or "Non-Fraudulent" as the value.
'''

prompt_template = f'''
1. CUSTOMER INFORMATION:
    {customer_information['name']}; {customer_information['gender']}; {customer_information['age']}; {customer_information['job']}
    -----------------------
    {registered_address['street']}; {registered_address['city']}; {registered_address['state']}; {registered_address['zip']}

2. CONTEXT:
    {context}

3. NEW TRANSACTIONAL INFORMATION:
    {new_transaction_info}

RESPONSE:
'''


from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16}
)


messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt_template},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)

eval(outputs[0]["generated_text"][-1]['content'])