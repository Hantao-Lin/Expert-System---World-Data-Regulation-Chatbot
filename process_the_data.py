# -*- coding: utf-8 -*-
"""Process the Data.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1bePUHNwpg8Lio1I9Tu7mie1ICIVLkHRt
"""

from google.colab import drive
drive.mount('/content/drive')

"""#GraphQL"""

!pip install neo4j graphene flask flask-graphql pyngrok pandas

from neo4j import GraphDatabase
import pandas as pd

# Assume cleaned_df is already loaded in your environment.
print("DataFrame shape:", cleaned_df.shape)

# Set up the Neo4j driver.
uri = "neo4j+s://66f142db.databases.neo4j.io"  # Your Neo4j instance URI
user = "neo4j"
password = "QMSrMna3lB27p-KBYwiGNL6lhhC_TJ8sHIk8eZ9Hmc0"  # Replace with actual password
driver = GraphDatabase.driver(uri, auth=(user, password))

def create_legal_provision(tx, country, section, text):
    """
    Inserts unique nodes for Country, Section, and Text in Neo4j,
    ensuring that each country-section combination has a unique provision.
    """
    section_id = f"{country}_{section}".replace(" ", "_")  # Unique Section ID
    text_id = f"{country}_{section}_Provision".replace(" ", "_")  # Unique Text ID

    query = """
    MERGE (c:Country {name: $country})
    MERGE (s:Section {name: $section, section_id: $section_id})
    MERGE (t:Text {content: $text, text_id: $text_id})
    MERGE (c)-[:HAS_SECTION]->(s)
    MERGE (s)-[:HAS_PROVISION]->(t)
    RETURN c, s, t
    """
    tx.run(query, country=country, section=section, section_id=section_id, text=text, text_id=text_id)

# Load the cleaned_df data into Neo4j
with driver.session() as session:
    for idx, row in cleaned_df.iterrows():
        country = row["Country"]
        for col in cleaned_df.columns[1:]:  # Skip "Country" column
            text = row[col]
            if pd.isna(text) or str(text).strip() == "":
                continue  # Skip empty values
            session.write_transaction(create_legal_provision, country, col, str(text))

print("✅ Data structured and loaded into Neo4j.")

# Define a test query to check for a few LegalProvision nodes.
test_query = """
MATCH (lp:LegalProvision)
RETURN lp.country AS country, lp.section AS section, lp.text AS text
LIMIT 5
"""

# Use a read transaction to execute the query.
with driver.session() as session:
    result = session.run(test_query)
    # Convert the result to a list of dictionaries
    records = result.data()

# Print the results
print("Test query results:")
for record in records:
    print(record)