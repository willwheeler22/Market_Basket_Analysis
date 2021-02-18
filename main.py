import pandas as pd
import os
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import psycopg2


# Use psycopg2 to connect to database and run query
connection = psycopg2.connect(user="user",
                              password="pw",
                              host="server",
                              port="port",
                              database="db")
cursor = connection.cursor()
select_statement = "query"

# Query pulls order number, SKU, quantity from ERP database

sql_query = pd.read_sql_query(select_statement, connection)
query = pd.DataFrame(sql_query)

# sum of billed qty by order number
grouped_orders = query.groupby(['ordernbr', 'groupedsku'])["billedqty"].sum().reset_index(name='count')
basket = (grouped_orders.groupby(['ordernbr', 'groupedsku'])['count']
          .sum().unstack().reset_index().fillna(0)
          .set_index('ordernbr'))


# The encoding function
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

# turns the count into either 1 or 0
basket_sets = basket.applymap(encode_units)

#use apriori algorithm to perform the market basket analysis
frequent_itemsets = apriori(basket_sets, min_support=0.0001, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.sort_values('lift', ascending=False, inplace=True)

# update MBA csv file to folder
os.remove('MBA.csv')
rules.to_csv(r'MBA.csv')

