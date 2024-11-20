import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def main():
    transactions = [
        ["milk", "bread", "eggs"],
        ["bread", "butter"],
        ["milk", "bread", "butter", "eggs"],
        ["bread", "eggs"],
        ["milk", "eggs"],
    ]

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
    # Get the number of itemsets
    num_itemsets = len(frequent_itemsets)  
    # Pass num_itemsets to association_rules
    rules = association_rules(frequent_itemsets, num_itemsets=num_itemsets, metric="confidence", min_threshold=0.7)  

    print("Frequent Itemsets:\n", frequent_itemsets)
    print("\nAssociation Rules:\n", rules)


if __name__ == "__main__":
    main()
