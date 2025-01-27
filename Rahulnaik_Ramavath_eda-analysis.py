import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

customers_df = pd.read_csv('Customers.csv')
products_df = pd.read_csv('Products.csv')
transactions_df = pd.read_csv('Transactions.csv')

customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])

def print_basic_info(df, name):
    print(f"\n=== {name} Dataset Analysis ===")
    print("\nShape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nData types:\n", df.dtypes)
    print("\nSample data:\n", df.head())

print_basic_info(customers_df, "Customers")
print_basic_info(products_df, "Products")
print_basic_info(transactions_df, "Transactions")


customer_purchase_freq = transactions_df.groupby('CustomerID').size()
customer_total_spend = transactions_df.groupby('CustomerID')['TotalValue'].sum()
customer_avg_basket = transactions_df.groupby('CustomerID')['TotalValue'].mean()

product_sales = transactions_df.groupby('ProductID').agg({
    'Quantity': 'sum',
    'TotalValue': 'sum'
}).sort_values('TotalValue', ascending=False)

customer_region_sales = pd.merge(
    transactions_df, 
    customers_df[['CustomerID', 'Region']], 
    on='CustomerID'
).groupby('Region').agg({
    'TransactionID': 'count',
    'TotalValue': 'sum'
})

transactions_df['Month'] = transactions_df['TransactionDate'].dt.to_period('M')
monthly_sales = transactions_df.groupby('Month')['TotalValue'].sum()

product_category_sales = pd.merge(
    transactions_df,
    products_df[['ProductID', 'Category']],
    on='ProductID'
).groupby('Category').agg({
    'TransactionID': 'count',
    'TotalValue': 'sum'
})

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
monthly_sales.plot(kind='line')
plt.title('Monthly Sales Trend')
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
product_category_sales['TotalValue'].plot(kind='pie', autopct='%1.1f%%')
plt.title('Sales Distribution by Category')

plt.subplot(2, 2, 3)
customer_region_sales['TotalValue'].plot(kind='bar')
plt.title('Sales by Region')
plt.xticks(rotation=45)

plt.subplot(2, 2, 4)
plt.hist(customer_purchase_freq, bins=30)
plt.title('Customer Purchase Frequency Distribution')

plt.tight_layout()
plt.show()
