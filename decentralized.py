from web3 import Web3
import requests
import torch
import io

PINATA_BASE_URL = "https://gateway.pinata.cloud/ipfs/"
PINATA_API_KEY = "3bc0a32cc1e8c321551b"
PINATA_SECRET_API_KEY = "f0c1bd57fdbbc87526688e4f16d08b34c2dccbdfda9e50fc23cd60cdefffaf1a"
PINATA_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySW5mb3JtYXRpb24iOnsiaWQiOiJjY2VlYTdjZC1lZDUwLTQxMWQtYmZiYi1jMDgwZDFhYjhkMmUiLCJlbWFpbCI6InBhdWxvYW1zMTZAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsInBpbl9wb2xpY3kiOnsicmVnaW9ucyI6W3siaWQiOiJGUkExIiwiZGVzaXJlZFJlcGxpY2F0aW9uQ291bnQiOjF9XSwidmVyc2lvbiI6MX0sIm1mYV9lbmFibGVkIjpmYWxzZSwic3RhdHVzIjoiQUNUSVZFIn0sImF1dGhlbnRpY2F0aW9uVHlwZSI6InNjb3BlZEtleSIsInNjb3BlZEtleUtleSI6IjNiYzBhMzJjYzFlOGMzMjE1NTFiIiwic2NvcGVkS2V5U2VjcmV0IjoiZjBjMWJkNTdmZGJiYzg3NTI2Njg4ZTRmMTZkMDhiMzRjMmRjY2JkZmRhOWU1MGZjMjNjZDYwY2RlZmZmYWYxYSIsImlhdCI6MTY5NzIyNjYzNX0.lYHfjeLg5fJYwY97yqaEon8A56y9h-o-TWeYs9E_tXw"
w3 = Web3(Web3.HTTPProvider(
    'https://eth-sepolia.g.alchemy.com/v2/T-sxo_sufcaSBdl7ww2U6FoOZwnpyNnU'))

# Contract details
CONTRACT_ADDRESS = '0x7b38788aF425728f8295642cC211E17dE13cd2A3'
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
PRIVATE_KEY = ''  # BE VERY CAREFUL WITH THIS!


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


def load_weights_from_ipfs(ipfs_hash):
    weights_url = f"{PINATA_BASE_URL}{ipfs_hash}"
    weights_response = requests.get(weights_url)
    print(weights_response)
    if weights_response.status_code == 200:
        buffer = io.BytesIO(weights_response.content)
        return torch.load(buffer)
    return None


def update_model_weights_from_contract(model):
    ipfs_hash = get_ipfs_hash_from_contract()
    print(ipfs_hash)
    if ipfs_hash:
        weights = load_weights_from_ipfs(ipfs_hash)
        print(weights)

        if weights:
            model_weights_shapes = [
                w.shape for w in model.get_weights()]
            loaded_weights_shapes = [w.shape for w in weights]

            if model_weights_shapes != loaded_weights_shapes:
                print("imbalanced weight shapes")

            else:
                try:
                    model.set_weights(weights)
                    return model
                except Exception as e:

                    print(f"An error occurred while setting weights: {e}")
                    return None
                print("weights found")
        else:
            print("weights not found")
            return None

# def update_model_weights_from_contract():
#     ipfs_hash = get_ipfs_hash_from_contract()
#     print(ipfs_hash)
#     if ipfs_hash:
#         weights = load_weights_from_nft_storage(ipfs_hash)
#         print(weights)

#         if weights:
#             model_weights_shapes = [
#                 w.shape for w in st.session_state['model'].get_weights()]
#             loaded_weights_shapes = [w.shape for w in weights]

#             if model_weights_shapes != loaded_weights_shapes:
#                 st.error(
#                     f"Model weight shapes: {model_weights_shapes} don't match loaded weight shapes: {loaded_weights_shapes}")

#             else:
#                 try:
#                     st.session_state['model'].set_weights(weights)
#                 except Exception as e:
#                     st.error(f"An error occurred while setting weights: {e}")
#                 print("weights found")
#                 st.info("Model weights updated from the blockchain!")
#         else:
#             print("weights not found")