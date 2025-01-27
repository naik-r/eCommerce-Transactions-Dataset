import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

customers_df = pd.read_csv('Rahulnaik_Ramavath_Customers.csv')
products_df = pd.read_csv('Rahulnaik_Ramavath_Products.csv')
transactions_df = pd.read_csv('Rahulnaik_Ramavath_Transactions.csv')

customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])

class LookalikeModel:
    def __init__(self, customers_df, transactions_df, products_df):
        self.customers_df = customers_df
        self.transactions_df = transactions_df
        self.products_df = products_df
        self.feature_matrix = None
        
    def create_customer_features(self):
        customer_features = self.transactions_df.groupby('CustomerID').agg({
            'TransactionID': 'count',
            'TotalValue': ['sum', 'mean', 'std'],
            'Quantity': ['sum', 'mean']
        }).fillna(0)
        
        customer_features.columns = ['_'.join(col).strip() for col in customer_features.columns.values]
        
        trans_with_categories = pd.merge(
            self.transactions_df,
            self.products_df[['ProductID', 'Category']],
            on='ProductID'
        )
        
        category_preferences = pd.crosstab(
            trans_with_categories['CustomerID'],
            trans_with_categories['Category'],
            values=trans_with_categories['TotalValue'],
            aggfunc='sum'
        ).fillna(0)
        
        customer_features = pd.merge(
            customer_features,
            category_preferences,
            left_index=True,
            right_index=True,
            how='left'
        ).fillna(0)
        
        latest_date = self.customers_df['SignupDate'].max()
        customer_duration = pd.DataFrame(
            self.customers_df.set_index('CustomerID')['SignupDate'].apply(
                lambda x: (latest_date - x).days
            )
        ).rename(columns={'SignupDate': 'account_age'})
        
        customer_features = pd.merge(
            customer_features,
            customer_duration,
            left_index=True,
            right_index=True,
            how='left'
        )
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(customer_features)
        
        self.feature_matrix = pd.DataFrame(
            scaled_features,
            index=customer_features.index,
            columns=customer_features.columns
        )
        
    def find_lookalikes(self, customer_id, n_recommendations=3):
        if self.feature_matrix is None:
            self.create_customer_features()
            
        if customer_id not in self.feature_matrix.index:
            print(f"Customer {customer_id} not found in the dataset")
            return pd.DataFrame()
            
        customer_vector = self.feature_matrix.loc[customer_id].values.reshape(1, -1)
        similarities = cosine_similarity(customer_vector, self.feature_matrix)
        
        similar_indices = similarities[0].argsort()[::-1][1:n_recommendations+1]
        similar_scores = similarities[0][similar_indices]
        
        similar_customers = pd.DataFrame({
            'CustomerID': self.feature_matrix.index[similar_indices],
            'SimilarityScore': similar_scores
        })
        
        return similar_customers

print("Initializing Lookalike Model...")
model = LookalikeModel(customers_df, transactions_df, products_df)

print("Generating lookalikes for first 20 customers...")
results = []
first_20_customers = customers_df['CustomerID'].iloc[:20]

for cust_id in first_20_customers:
    print(f"Processing customer {cust_id}...")
    lookalikes = model.find_lookalikes(cust_id)
    results.append({
        'CustomerID': cust_id,
        'Lookalikes': lookalikes.to_dict('records')
    })

print("Preparing results for CSV...")
formatted_results = []
for result in results:
    customer_id = result['CustomerID']
    for idx, lookalike in enumerate(result['Lookalikes'], 1):
        formatted_results.append({
            'SourceCustomerID': customer_id,
            f'LookalikeCustomer_{idx}': lookalike['CustomerID'],
            f'SimilarityScore_{idx}': round(lookalike['SimilarityScore'], 4)
        })

output_df = pd.DataFrame(formatted_results)
output_df.to_csv('Rahulnaik_Ramavath_Lookalike.csv', index=False)
print("Results saved to Rahulnaik_Ramavath_Lookalike.csv")

print("\nFirst few results:")
print(output_df.head())
