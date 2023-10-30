import flwr as fl
import logging
import io
import torch
import requests
from web3 import Web3
from flwr.common import parameter
import numpy as np

logging.getLogger("flwr").setLevel(logging.DEBUG)

PINATA_BASE_URL = "https://api.pinata.cloud/pinning/pinFileToIPFS"
PINATA_FETCH_BASE_URL = "https://gateway.pinata.cloud/ipfs/"
PINATA_API_KEY = "3bc0a32cc1e8c321551b"
PINATA_SECRET_API_KEY = "f0c1bd57fdbbc87526688e4f16d08b34c2dccbdfda9e50fc23cd60cdefffaf1a"
PINATA_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySW5mb3JtYXRpb24iOnsiaWQiOiJjY2VlYTdjZC1lZDUwLTQxMWQtYmZiYi1jMDgwZDFhYjhkMmUiLCJlbWFpbCI6InBhdWxvYW1zMTZAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsInBpbl9wb2xpY3kiOnsicmVnaW9ucyI6W3siaWQiOiJGUkExIiwiZGVzaXJlZFJlcGxpY2F0aW9uQ291bnQiOjF9XSwidmVyc2lvbiI6MX0sIm1mYV9lbmFibGVkIjpmYWxzZSwic3RhdHVzIjoiQUNUSVZFIn0sImF1dGhlbnRpY2F0aW9uVHlwZSI6InNjb3BlZEtleSIsInNjb3BlZEtleUtleSI6IjNiYzBhMzJjYzFlOGMzMjE1NTFiIiwic2NvcGVkS2V5U2VjcmV0IjoiZjBjMWJkNTdmZGJiYzg3NTI2Njg4ZTRmMTZkMDhiMzRjMmRjY2JkZmRhOWU1MGZjMjNjZDYwY2RlZmZmYWYxYSIsImlhdCI6MTY5NzIyNjYzNX0.lYHfjeLg5fJYwY97yqaEon8A56y9h-o-TWeYs9E_tXw"
w3 = Web3(Web3.HTTPProvider(
    'https://eth-sepolia.g.alchemy.com/v2/T-sxo_sufcaSBdl7ww2U6FoOZwnpyNnU'))

# Contract details
CONTRACT_ADDRESS = '0x7b38788aF425728f8295642cC211E17dE13cd2A3'
SERVER_ADDRESS = "127.0.0.1:8087"
ABI = [
    {
        "inputs": [
            {
                "internalType": "string",
                "name": "ipfhash",
                                "type": "string"
            }
        ],
        "name": "storeWeight",
        "outputs": [],
        "stateMutability": "nonpayable",
                "type": "function"
    },
    {
        "anonymous": False,
        "inputs": [
            {
                "indexed": False,
                "internalType": "string",
                "name": "currentWeight",
                                "type": "string"
            },
            {
                "indexed": False,
                "internalType": "string",
                "name": "newWeight",
                                "type": "string"
            },
            {
                "indexed": False,
                "internalType": "uint256",
                "name": "blockNumber",
                                "type": "uint256"
            },
            {
                "indexed": False,
                "internalType": "uint256",
                "name": "timeStamp",
                                "type": "uint256"
            }
        ],
        "name": "weightUpdated",
        "type": "event"
    },
    {
        "inputs": [],
        "name": "retrieveLatestWeight",
                "outputs": [
            {
                "internalType": "string",
                "name": "",
                "type": "string"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    }
]  # ABI for your contract

# Initialize contract object
contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=ABI)

# Account details (the account sending the transaction)
SENDER_ADDRESS = '0xe459c175e06FFcE758CE14Ed7C4d016a6d9a858F'
# BE VERY CAREFUL WITH THIS!
PRIVATE_KEY = '5913466bb0971661e17b14fa96b75fe9917bf94bb2cc983eeb35df256005e321'

# Helper functions


def get_ipfs_hash_from_contract():
    return contract.functions.retrieveLatestWeight().call()


def deserialize_weights(buffer):
    buffer.seek(0)  # Reset buffer position to the beginning, just in case
    model_weights = torch.load(buffer)
    return model_weights


def retrieve_from_nft_storage(cid):
    # Using the IPFS gateway provided by NFT.Storage
    url = f"https://dweb.link/ipfs/{cid}"
    response = requests.get(url)
    response.raise_for_status()  # Raise exception if any HTTP error occurs
    buffer = io.BytesIO(response.content)
    return buffer


def load_weights_from_nft_storage(cid):
    buffer = retrieve_from_nft_storage(cid)
    return deserialize_weights(buffer)


# def load_weights_from_ipfs(ipfs_hash):
#     weights_url = f"{PINATA_FETCH_BASE_URL}{ipfs_hash}"
#     weights_response = requests.get(weights_url)
#     if weights_response.status_code == 200:
#         buffer = io.BytesIO(weights_response.content)
#         return torch.load(buffer)
#     return None


def get_old_weights_from_contract():
    ipfs_hash = get_ipfs_hash_from_contract()
    print("oldhash", ipfs_hash)
    if ipfs_hash:
        weights = load_weights_from_nft_storage(ipfs_hash)
        print("got old weights")
        return weights
    else:
        return None


def store_ipfs_hash_to_contract(ipfs_hash):
    # Prepare the transaction
    txn = contract.functions.storeWeight(ipfs_hash).buildTransaction({
        'chainId': 11155111,   # Chain ID for Rinkeby testnet. Change if using another network
        'gas': 200000,
        'gasPrice': w3.toWei('5', 'gwei'),
        'nonce': w3.eth.getTransactionCount(SENDER_ADDRESS),
    })

    # Sign the transaction
    signed_txn = w3.eth.account.signTransaction(txn, PRIVATE_KEY)

    # Send the transaction
    txn_hash = w3.eth.sendRawTransaction(signed_txn.rawTransaction)

    # Wait for the transaction to be mined
    txn_receipt = w3.eth.waitForTransactionReceipt(txn_hash)

    print(txn_receipt)

    return txn_receipt


def serialize_weights(model_weights):
    buffer = io.BytesIO()
    torch.save(model_weights, buffer)
    buffer.seek(0)  # Important: Reset buffer position to the beginning
    return buffer


def pin_to_nft_storage(buffer, api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJkaWQ6ZXRocjoweEQwOTM1MmIyMDgzMzAyNjRBNmJlMjg3NzA2RjdiNzVGZkE3MTdlN2IiLCJpc3MiOiJuZnQtc3RvcmFnZSIsImlhdCI6MTY5NzI5NjU0NTYwNiwibmFtZSI6IkZlZGEifQ.4Urvc0G5DZXeZ-JZPRejg_ltZrdGhgQ0Qc-3f3k8eZ0"):
    url = "https://api.nft.storage/upload"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/octet-stream"
    }

    response = requests.post(url, headers=headers, data=buffer)
    response.raise_for_status()  # Raise exception if any HTTP error occurs

    # Extract the CID from the response (assuming the response is JSON)
    cid = response.json()["value"]["cid"]

    print("cid", cid)

    return cid

def pin_to_pinata(serialized_weights):
    headers = {
        "Authorization": f"Bearer {PINATA_JWT}"
    }
    response = requests.post(
        PINATA_BASE_URL,
        files={"file": serialized_weights},
        headers=headers
    )
    if response.status_code != 200:
        raise Exception("Failed to pin data to Pinata")
    return response.json().get("IpfsHash")


def combine_weights(old_weights, new_weights, learning_rate):
    updated_weights = []
    for old_weight, new_weight in zip(old_weights, new_weights):
        # Ensure both old_weight and new_weight are NumPy arrays
        if isinstance(old_weight, np.ndarray) and isinstance(new_weight, np.ndarray):
            # Apply the combination logic to the NumPy arrays
            combined_array = (1.0 - learning_rate) * \
                old_weight + learning_rate * new_weight
            updated_weights.append(combined_array)
        else:
            # If the weights are not NumPy arrays, take the new weights as is
            updated_weights.append(new_weight)
    return updated_weights


# Custom strategy


class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, min_available_clients, min_fit_clients, learning_rate):
        super().__init__(min_available_clients=min_available_clients,
                         min_fit_clients=min_fit_clients)
        self.learning_rate = learning_rate
        self.aggregated_weights = None

    def aggregate_fit(self, rnd, results, failures):
        # Ensure there are available clients for aggregation

        if len(results) < self.min_available_clients:
            return self.aggregated_weights, None

        new_weights, metrics = super().aggregate_fit(rnd, results, failures)
        self.aggregated_weights = new_weights

        # # If it's the first round, initialize aggregated_weights
        # if rnd == 1:
        #     self.aggregated_weights = new_weights
        # else:
        #     # Incrementally update the aggregated weights
        #     self.aggregated_weights = combine_weights(
        #         self.aggregated_weights, new_weights, self.learning_rate
        #     )

        # Check if it's the last round
        if rnd == 2: 
                new_compute = parameter.parameters_to_ndarrays(new_weights)
        
                serialized_weights = serialize_weights(new_compute)

                # Save the serialized weights to Pinata
                ipfs_hash = pin_to_pinata(serialized_weights)

                store_ipfs_hash_to_contract(ipfs_hash=ipfs_hash)

                print(f"IPFS hash (via Pinata): {ipfs_hash}")

        return self.aggregated_weights, metrics


if __name__ == "__main__":
    # Use the custom strategy with a learning rate of 0.1 (adjust as needed)
    learning_rate = 0.1
    strategy = CustomFedAvg(min_available_clients=2,
                            min_fit_clients=2, learning_rate=learning_rate)
    fl.server.start_server(server_address=SERVER_ADDRESS,
                           config=fl.server.ServerConfig(num_rounds=2), strategy=strategy)
