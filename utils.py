import random
import torch
import numpy as np
import pandas as pd
from os.path import join as pjoin
from datetime import datetime
from geopy.geocoders import Photon, Nominatim
from qdrant_client import models

def dob_to_age(dob_str):
    """
    Converts date of birth (dob) in 'YYYY-MM-DD' format to age in years.
    
    Args:
        dob_str (str): Date of birth in 'YYYY-MM-DD' format.
    
    Returns:
        int: Age in years.
    """
    # Convert the dob_str into a datetime object
    dob = datetime.strptime(dob_str, '%Y-%m-%d')
    
    # Get the current date
    today = datetime.today()
    
    # Calculate age
    age = today.year - dob.year
    
    # Adjust age if birthday hasn't occurred yet this year
    if (today.month, today.day) < (dob.month, dob.day):
        age -= 1
    
    return age

def lat_long_to_address(lat, long):
    """
    Converts latitude and longitude to an address using Geopy.
    
    Args:
        lat (float): Latitude value.
        long (float): Longitude value.
    
    Returns:
        str: Corresponding address.
    """
    photon, nominatim = False, False

    # Try to use Photon first, then Nominatim
    try:
        GEOLOCATOR = Photon(user_agent="measurements")
        # Reverse geocode the coordinates
        location = GEOLOCATOR.reverse((lat, long), language='en')
        photon = True
    except:
        try:
            GEOLOCATOR = Nominatim(user_agent="measurements")
            # Reverse geocode the coordinates
            location = GEOLOCATOR.reverse((lat, long), language='en')
            nominatim = True
        # If both fail, return None
        except:
            return None
        
    # Extract the address from the location
    if location:
        if photon:
            properties = location.raw['properties']
        elif nominatim:
            properties = location.raw['address']

        # Remove the extent and osm_id keys
        if properties.get('extent'):
            del properties['extent']
        if properties.get('osm_id'):
            del properties['osm_id']
        return properties
    else:
        return None
    
def convert_transaction_data_to_str(transaction_information, merchant_information, payment_address, merchant_address):
    '''
    Convert transaction data to string format
    It takes transaction_information, merchant_information, payment_address, merchant_address as input
    and returns a formatted string

    Args:
        transaction_information : DataFrame row containing transaction information
        merchant_information : DataFrame row containing merchant information
        payment_address : Address of the payment location
        merchant_address : Address of the merchant location

    Returns:
        str: Formatted string containing transaction information
    '''
    template = f'''
{transaction_information['amt']}
-----------------------
{merchant_information['merchant']}; {merchant_information['category']}
-----------------------
{payment_address}
-----------------------
{merchant_address}
'''
    return template

def get_user_basic_info(transaction_detail):
    '''
    Get user basic information from transaction details
    It takes transaction_detail as input and returns customer_information and registered_address
    Basic details include basic information of the customer and registered address

    Args:
        transaction_detail : DataFrame containing transaction details

    Returns:
        customer_information : DataFrame containing customer information
        registered_address : DataFrame containing registered address
    '''
    customer_information = transaction_detail[['name', 'gender', 'job', 'age']]
    registered_address = transaction_detail[['street', 'city', 'state', 'zip']]
    return customer_information, registered_address

def get_transactional_data(transaction_detail, convert_coordinates_to_address=True):
    '''
    Get transactional data from transaction details
    It takes transaction_detail as input and returns transaction_information, merchant_information, payment_address, merchant_address
    Transactional data includes transaction information, merchant information, payment address and merchant address

    Args:
        transaction_detail : DataFrame containing transaction details
        convert_coordinates_to_address : Boolean value to convert coordinates to address

    Returns:
        transaction_information : DataFrame containing transaction information
        merchant_information : DataFrame containing merchant information
        payment_address : Address of the payment location
        merchant_address : Address of the merchant location
    '''
    transaction_information = transaction_detail[['trans_date_trans_time', 'amt']]
    merchant_information = transaction_detail[['merchant', 'category']]

    payment_lat, payment_long = transaction_detail[['lat', 'long']].values
    # Convert the coordinates to an address
    if convert_coordinates_to_address:
        payment_address = lat_long_to_address(payment_lat, payment_long)
    else:
        payment_address = None
    # If the address is found, add the coordinates to the address
    if payment_address:
        payment_address.update({'lat': payment_lat, 'long': payment_long})
    else:
        payment_address = {'lat': payment_lat, 'long': payment_long}
    payment_address = '; '.join([str(v) for _, v in payment_address.items()])

    # Get the merchant's latitude and longitude and convert them to an address, same as above
    merchant_lat, merchant_long = transaction_detail[['merch_lat', 'merch_long']].values
    if convert_coordinates_to_address:
        merchant_address = lat_long_to_address(merchant_lat, merchant_long)
    else:
        merchant_address = None
    if merchant_address:
        merchant_address.update({'lat': merchant_lat, 'long': merchant_long})
    else:
        merchant_address = {'lat': merchant_lat, 'long': merchant_long}
    merchant_address = '; '.join([str(v) for _, v in merchant_address.items()])
    
    return transaction_information, merchant_information, payment_address, merchant_address

def embed_transaction(description, model, tokenizer):
    '''
    This function takes a description as input and returns the embeddings of the description
    It uses the pre-trained model and tokenizer to get the embeddings of the description

    Args:
        description : Description of the transaction
        model : Pre-trained model for embedding
        tokenizer : Tokenizer for the

    Returns:
        embeddings : Embeddings of the description
    '''
    inputs = tokenizer(description, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs)
        embeddings = embeddings[0][:, 0]
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings

def insert_transaction(transaction_embedding, payload, idx, client):
    '''
    This function inserts transaction data into the Qdrant collection
    It takes transaction_embedding, payload and idx as input and inserts the data into the collection

    Args:
        transaction_embedding : Embeddings of the transaction
        payload : Description of the transaction
        idx : Index of the transaction
        client : Qdrant client

    Returns:
        client : Qdrant client
    '''
    client.upsert(
        collection_name="transactions",
        points=[
            models.PointStruct(
                id=idx,
                payload={
                    "transaction_data": payload,
                },
                vector=transaction_embedding,
            ),
        ],
    )

    return client

def get_as_close_transactions(embedding, client):
    '''
    This function takes the embedding and client as input and returns the closest transactions
    It queries the Qdrant client to get the closest transactions based on the embedding

    Args:
        embedding : Embeddings of the transaction
        client : Qdrant client

    Returns:
        results : Closest transactions based on the embedding
    '''
    results = client.query_points(
        collection_name="transactions",
        query=embedding[0].tolist(),
        limit=10,
    ).points
    return results

def filter_data(df):
    '''
    This function filters the data based on random cc_num
    It extracts 5 random cc_num (user) and filters the data based on those cc_num

    Args:
        df: pandas dataframe

    Returns:
        filtered_data: pandas dataframe
    '''
    # get random cc_num
    random_cc_num = random.sample(list(df['cc_num'].unique()), 5)

    # filter data based on random cc_num
    filtered_data = df[df['cc_num'].isin(random_cc_num)]
    return filtered_data, random_cc_num

def modify_column(data):
    '''
    This function modifies the columns of the data
    It converts the dob to age, converts the gender codes to strings, 
    combines the first and last names, and replaces the 'fraud_' prefix in the merchant column

    Args:
        data: pandas dataframe

    Returns:
        data: pandas dataframe
    '''
    # convert dob to age
    data['age'] = data['dob'].map(dob_to_age)

    # Explicitly convert the gender codes to strings
    data.gender = data.gender.replace({
        'F': 'Female',
        'M': 'Male'
    })

    # Combine the first and last names
    data['name'] = data['first'] + ' ' + data['last']

    # Replace the 'fraud_' prefix in the merchant column
    data['merchant'] = data.merchant.str.replace('fraud_', '')

    return data

def load_data(root, data_path):
    '''
    This function loads the data from the given path
    It filters the data and modifies the columns

    Args:
        root: str
        data_path: str

    Returns:
        data: pandas dataframe
        random_cc_num: list
    '''
    df = pd.read_csv(pjoin(root, data_path))
    data, random_cc_num = filter_data(df)
    data = modify_column(data)
    return data, random_cc_num


def detect_anomalies(query_response, threshold=0.95):
    '''
    This function detects whether the new transation is fraudulent or not.
    It does this by checking the scores of all the similar retrieved transactions.
    The idea is that if the new transaction will be genuine transaction then the mean similarity score of the retrieved transaction with be very high, 
    meaning new transaction is very similar to genuine normal transaction
    But if the new transaction is fraudulent then most likely the similarity score will not be as high and based on a threhsold we can say that since 
    this new transaction is not as similar to genuine transaction hence this is fraudulent transaction

    Args:
        query_response : Response from the Qdrant client
        threshold : Threshold value to determine whether the new transaction is fraudulent or not

    Returns:
        bool: True if the new transaction is fraudulent, False otherwise
    '''
    similarity_scores = []
    for result in query_response:
        similarity_scores.append(result.score)

    # If the mean similarity score is less than the threshold, return True
    if np.mean(similarity_scores) < threshold:
        return True
    else:
        return False