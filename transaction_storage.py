import json
import os

TRANSACTION_FILE = "transactions.json"
MIN_TRANSACTIONS = 5

def load_transactions():
    if not os.path.exists(TRANSACTION_FILE):
        return []
    with open(TRANSACTION_FILE, "r") as f:
        return json.load(f)

def save_transactions(transactions):
    with open(TRANSACTION_FILE, "w") as f:
        json.dump(transactions, f)

def add_transaction(transaction):
    transactions = load_transactions()
    transactions.append(transaction)
    save_transactions(transactions)