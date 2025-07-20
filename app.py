# app.py
import streamlit as st
from transaction_storage import load_transactions, add_transaction, MIN_TRANSACTIONS
from kmeans_module import run_kmeans
from apriori_module import run_apriori
from collections import Counter
import pandas as pd

# List of items in the supermarket
ITEMS = ["milk", "bread", "eggs", "soda", "chips", "cereal", "juice", "butter", "cheese", "yogurt"]

# App-wide style
st.set_page_config(page_title="Supermarket Simulator", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #f63366;
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }
    .stTabs [role="tab"] {
        background-color: #e0e0e0;
        color: black;
        font-weight: 600;
        border-radius: 10px 10px 0 0;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🛒 Interactive Supermarket Simulation")

# Tabs for clean layout
tab1, tab2, tab3 = st.tabs(["🛍️ Shop", "📈 Data Mining", "📊 Insights"])

# --- TAB 1: SHOP ---
with tab1:
    st.markdown("### 🛒 <span style='color:green'>Step 1: Create a Transaction</span>", unsafe_allow_html=True)
    selected_items = st.multiselect("Select items to add to your cart:", ITEMS)

    if st.button("Submit Transaction"):
        if selected_items:
            add_transaction(selected_items)
            st.success("Transaction added!")
        else:
            st.warning("Please select at least one item.")

    # Show receipts
    st.subheader("🧾 Store Receipt - Transactions")
    transactions = load_transactions()
    if transactions:
        for i, txn in enumerate(transactions, 1):
            items_list = "\n".join(f"- {item}" for item in txn)
            receipt_text = (
                f"RECEIPT #{i:03d}\n"
                f"------------------------\n"
                f"Items:\n"
                f"{items_list}\n"
                f"------------------------\n"
                f"Total items: {len(txn)}"
            )
            st.code(receipt_text)
    else:
        st.info("No transactions recorded yet.")

# --- TAB 2: DATA MINING ---
with tab2:
    st.markdown("### 📈 <span style='color:blue'>Step 2: Run Data Mining</span>", unsafe_allow_html=True)
    transactions = load_transactions()

    if len(transactions) >= MIN_TRANSACTIONS:
        if st.button("Run K-Means Clustering"):
            result = run_kmeans(transactions, k=2)
            st.subheader("K-Means Clustering Results")
            st.dataframe(result)

            max_cluster = result.loc[result["Number of Transactions"].idxmax()]
            min_cluster = result.loc[result["Number of Transactions"].idxmin()]

            st.markdown(f"""
            **Insights:**
            - 🟢 Most customers fall into **{max_cluster['Cluster']}** with **{max_cluster['Number of Transactions']}** transactions.
            - 🔵 Least activity observed in **{min_cluster['Cluster']}** with **{min_cluster['Number of Transactions']}** transactions.
            """)

        if st.button("Run Apriori Association Rules"):
            rules_df = run_apriori(transactions)
            st.subheader("Association Rules")
            st.dataframe(rules_df)

            if not rules_df.empty:
                top_rule = rules_df.sort_values("confidence", ascending=False).iloc[0]
                st.markdown(f"""
                **Insights:**
                - 📈 Strongest rule: **{set(top_rule['antecedents'])} → {set(top_rule['consequents'])}**
                - 💡 Confidence: **{top_rule['confidence']:.2f}**, Lift: **{top_rule['lift']:.2f}**
                """)
            else:
                st.info("No strong association rules found. Try adding more diverse transactions.")
    else:
        st.warning(f"Create at least {MIN_TRANSACTIONS} transactions to enable data mining.")

# --- TAB 3: INSIGHTS ---
with tab3:
    st.markdown("### 📊 <span style='color:orange'>Item Popularity</span>", unsafe_allow_html=True)
    transactions = load_transactions()
    if transactions:
        all_items = [item for txn in transactions for item in txn]
        item_counts = Counter(all_items)
        item_df = pd.DataFrame(item_counts.items(), columns=["Item", "Count"]).sort_values(by="Count", ascending=False)
        st.bar_chart(item_df.set_index("Item"))

        max_count = max(item_counts.values())
        most_common_items = [item for item, cnt in item_counts.items() if cnt == max_count]
        items_str = ", ".join(f"`{item}`" for item in most_common_items)
        st.markdown(f"**Most popular item(s):** 🛒 {items_str} with **{max_count}** selections each.")
    else:
        st.info("No transactions yet to analyze item popularity.")
