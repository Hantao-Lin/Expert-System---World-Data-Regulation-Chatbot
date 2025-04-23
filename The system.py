#!/usr/bin/env python
# coding: utf-8

# # Process pdf into df

# In[ ]:


import fitz  # PyMuPDF
import re
import pandas as pd

def extract_country_content_with_sections(file_path):
    data = []
    with fitz.open(file_path) as doc:
        first_pages_text = "\n".join([doc.load_page(i).get_text() for i in range(5)])
        country_pattern = r"(\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)(?:\s*\.\s*)+(\d+)"
        countries = re.findall(country_pattern, first_pages_text)

        section_headers = [
            "LAW", "DEFINITIONS", "NATIONAL DATA PROTECTION AUTHORITY",
            "REGISTRATION", "DATA PROTECTION OFFICERS", "COLLECTION & PROCESSING",
            "TRANSFER", "SECURITY", "BREACH NOTIFICATION", "ENFORCEMENT",
            "ELECTRONIC MARKETING", "ONLINE PRIVACY", "KEY CONTACTS"
        ]

        for i in range(len(countries) - 1):
            country, start_page = countries[i]
            _, next_page = countries[i + 1]
            start_page, next_page = int(start_page) - 1, int(next_page) - 1

            content = ""
            for p in range(start_page, next_page):
                page = doc.load_page(p)
                content += page.get_text()

            sections = {header: "" for header in section_headers}  # Initialize with empty sections
            current_section = None
            for line in content.splitlines():
                if line.isupper() and line.strip() in section_headers:
                    current_section = line.strip()
                elif current_section:
                    sections[current_section] += line + " "

            data.append([country] + [sections[header].strip() for header in section_headers])

    # Create DataFrame with each section as a separate column
    columns = ["Country"] + section_headers
    return pd.DataFrame(data, columns=columns)

def clean_content(df):
    patterns_to_remove = [
        'www.dlapiperdataprotection.com',
        'DATA PROTECTION LAWS OF THE WORLD  Data Protection Laws of the World',
        'DATA PROTECTION LAWS OF THE WORLD'
    ]

    # Iterate over each section column (excluding the "Country" column)
    for column in df.columns:
        if column != "Country":  # Skip the "Country" column
            df[column] = df[column].astype(str).apply(lambda content: clean_text(content, patterns_to_remove))

    return df

def clean_text(content, patterns_to_remove):
    """Helper function to clean text content by replacing patterns."""
    content = content.replace('\n', ' ')  # Replace newline with space
    for pattern in patterns_to_remove:
        content = content.replace(pattern, '')
    return content.strip()  # Remove leading/trailing spaces


# In[2]:


pdf_path="C:/Users/hanta/Downloads/Data-Protection-Full.pdf"
df = extract_country_content_with_sections(pdf_path)
cleaned_df = clean_content(df)


# # Load into Pinecone

# In[6]:


import re
import time
import pinecone
import pandas as pd
import numpy as np
import ollama  

api_key = "Your Pinecone API Key Here"
environment = "us-east-1"

# Initialize Pinecone 
pc = pinecone.Pinecone(api_key=api_key)

# Set index name and embedding dimension
index_name = "legal-index"
desired_dimension = 1024

# Create the index if it doesn't exist
if index_name not in [idx.name for idx in pc.list_indexes().indexes]:
    pc.create_index({
        "name": index_name,
        "dimension": desired_dimension,
        "metric": "cosine",
        "spec": {
            "serverless": {
                "cloud": "aws",
                "region": environment
            }
        }
    })
    # Allow a short delay for the index to be ready
    time.sleep(5)

# Connect to the index 
index = pc.Index(index_name)

# ----------------------------
# Define a Summarization Function using Ollama
# ----------------------------
def summarize_document(text, model="llama3.1", max_length=1000):
    """
    Uses Ollama to generate a concise summary of the input text, then truncates it to a maximum length.
    """
    prompt = (
        f"Please provide a concise summary of the following legal document in no more than 300 words, "
        f"and ensure the summary is under {max_length} characters:\n\n{text}\n\nSummary:"
    )
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    summary = response['message']['content'].strip()
    # Truncate summary if necessary
    return summary[:max_length]

# ----------------------------
# Process DataFrame and Prepare Metadata
# ----------------------------
print("Original DataFrame shape:", cleaned_df.shape)
cleaned_df.columns = cleaned_df.columns.str.replace(" ", "_")
print("Modified columns:", cleaned_df.columns.tolist())

documents = []
ids = []
metadatas = []

for idx, row in cleaned_df.iterrows():
    country = row["Country"]
    # Concatenate all text columns (except "Country") into one document per row.
    text = " ".join([str(row[col]).strip() for col in cleaned_df.columns if col != "Country"])
    if text:
        # Use Ollama to summarize the document.
        summary = summarize_document(text)
        documents.append(text)
        ids.append(f"{country}_{idx}")  # Unique ID per row
        metadatas.append({
            "country": country,
            "summary": summary    # Store the concise summary
        })

print("Total documents to process:", len(documents))

# ----------------------------
# Generate Embeddings in Smaller Batches to Avoid 413 Errors
# ----------------------------
batch_inference_size = 50
embedding_list = []

for i in range(0, len(documents), batch_inference_size):
    batch_texts = documents[i:i+batch_inference_size]
    print(f"Embedding batch {i//batch_inference_size + 1} of {((len(documents) - 1) // batch_inference_size) + 1}")
    
    embeddings_response = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=batch_texts,
        parameters={"input_type": "passage", "truncate": "END"}
    )
    
    batch_embeddings = [item["values"] for item in embeddings_response]
    embedding_list.extend(batch_embeddings)
    
# Convert list of embeddings to a numpy array
embeddings = np.array(embedding_list)
# Normalize embeddings (L2 norm = 1) for cosine similarity.
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
normalized_embeddings = embeddings / norms

# ----------------------------
# Upsert Embeddings into Pinecone in Batches
# ----------------------------
batch_size = 16
total_batches = (len(documents) + batch_size - 1) // batch_size
start_time = time.time()

for batch_idx in range(total_batches):
    start = batch_idx * batch_size
    end = min((batch_idx + 1) * batch_size, len(documents))
    
    batch_ids = ids[start:end]
    batch_embeddings = normalized_embeddings[start:end]
    batch_metadatas = metadatas[start:end]
    
    vectors = [
        (batch_ids[i], batch_embeddings[i].tolist(), batch_metadatas[i])
        for i in range(len(batch_ids))
    ]
    
    index.upsert(vectors=vectors, namespace="legal_data")
    print(f"Upserted batch {batch_idx + 1}/{total_batches}")

elapsed = time.time() - start_time
print(f"Successfully upserted {len(documents)} embeddings to Pinecone in {elapsed:.2f} seconds.")


# # Expert System

# In[2]:


import re
import difflib
import ollama
import pycountry
from neo4j import GraphDatabase
import pinecone
import time
import numpy as np  # For vector operations

# --- Aggregator Agent ---
class TeammateAgent3:
    """
    Aggregator agent that produces a unified final answer without exposing internal debugging or chain-of-thought details.
    """
    def __init__(self, model="deepseek-r1:8b", retriever=None):
        self.model = model
        self.retriever = retriever

    def detect_gaps(self, initial_query, combined_references):
        prompt = f"""
You are a legal compliance expert. Analyze the following retrieved references for any missing details needed to answer the query.

User Query: "{initial_query}"
Retrieved References:
{combined_references}

If any additional information is needed, list concise follow-up queries; otherwise, reply with "No gaps."
"""
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return response['message']['content'].strip()

    def final_aggregate_response(self, initial_query, sub_references, query_category, conversation_context=""):
        context_info = f"Conversation History:\n{conversation_context}\n\n" if conversation_context else ""
        combined_references = "\n\n".join(
            [f"Country: {ref.get('country', 'Global')}\nSection: {ref.get('section', 'General')}\nReference: {ref['reference']}"
             for ref in sub_references]
        )
        
        # Check for any missing information.
        gap_analysis = self.detect_gaps(initial_query, combined_references)
        if "follow-up queries" in gap_analysis.lower():
            follow_up_queries = gap_analysis.split("Follow-Up Queries (if any):")[-1].strip()
            for query in follow_up_queries.split("\n"):
                if query.strip():
                    additional_results = self.retriever.retrieve_information(query.strip())
                    for result in additional_results[:2]:
                        combined_info = ""
                        if result["results"]:
                            if isinstance(result["results"], list):
                                if isinstance(result["results"][0], dict):
                                    texts = [r["text"] for r in result["results"]]
                                    combined_info += "\n".join(texts) + "\n"
                                else:
                                    combined_info += "\n".join(result["results"]) + "\n"
                            else:
                                combined_info += str(result["results"]) + "\n"
                        if not combined_info.strip():
                            combined_info = "No relevant information was retrieved."
                        sub_references.append({
                            "country": result.get("country", "Global"),
                            "section": result.get("section", "General"),
                            "reference": f"Follow-Up Query: {query.strip()}\nRetrieved Information: {combined_info}"
                        })
        
        if query_category == "comparison":
            focus_instructions = "Focus on comparing the relevant aspects between countries and sections."
        elif query_category == "procedural":
            focus_instructions = "Provide a step-by-step process and list the required actions."
        elif query_category == "explanation":
            focus_instructions = "Explain the subject matter clearly and concisely."
        else:
            focus_instructions = "Provide a clear, concise answer directly addressing the query."
    
        prompt = f"""
{context_info}You are a legal compliance expert. Answer the following query:
"{initial_query}"
{focus_instructions}

Gap Analysis:
{gap_analysis}

Retrieved References:
{combined_references}

Final Answer (structured and concise):
"""
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return response['message']['content'].strip()

# --- Information Retrieval System ---
class InformationRetriever:
    def __init__(self, 
                 neo4j_uri="neo4j+s://66f142db.databases.neo4j.io", 
                 neo4j_user="neo4j", 
                 neo4j_password="",
                 pinecone_api_key="",
                 pinecone_environment="us-east-1",
                 pinecone_index_name="legal-index"):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.sections = [
            "LAW", "DEFINITIONS", "NATIONAL DATA PROTECTION AUTHORITY", "REGISTRATION",
            "DATA PROTECTION OFFICERS", "COLLECTION & PROCESSING", "TRANSFER", "SECURITY",
            "BREACH NOTIFICATION", "ENFORCEMENT", "ELECTRONIC MARKETING", "ONLINE PRIVACY", "KEY CONTACTS"
        ]
        self.countries = [country.name for country in pycountry.countries]

        self.pinecone_api_key = pinecone_api_key
        self.pinecone_environment = pinecone_environment
        self.pinecone_index_name = pinecone_index_name
        self.pc = pinecone.Pinecone(api_key=self.pinecone_api_key)
        if self.pinecone_index_name not in [idx.name for idx in self.pc.list_indexes().indexes]:
            self.pc.create_index({
                "name": self.pinecone_index_name,
                "dimension": 1024,
                "metric": "cosine",
                "spec": {
                    "serverless": {
                        "cloud": "aws",
                        "region": self.pinecone_environment
                    }
                }
            })
            time.sleep(5)
        self.pinecone_index = self.pc.Index(self.pinecone_index_name)

    def detect_countries(self, text):
        words = re.findall(r"\w+", text)
        detected_countries = []
        for word in words:
            matches = difflib.get_close_matches(word, self.countries, n=1, cutoff=0.85)
            if matches:
                detected_countries.append(matches[0])
        return list(set(detected_countries))

    def detect_sections(self, text):
        detected_sections = []
        query_lower = text.lower()
        keyword_mapping = {
            "registration": "REGISTRATION",
            "company registration": "REGISTRATION",
            "data protection": "NATIONAL DATA PROTECTION AUTHORITY",
            "collect": "COLLECTION & PROCESSING",
            "process": "COLLECTION & PROCESSING",
            "security": "SECURITY",
            "breach": "BREACH NOTIFICATION",
            "enforcement": "ENFORCEMENT",
            "electronic marketing": "ELECTRONIC MARKETING",
            "online privacy": "ONLINE PRIVACY",
            "key contacts": "KEY CONTACTS",
            "law": "LAW",
            "definitions": "DEFINITIONS",
            "officers": "DATA PROTECTION OFFICERS"
        }
        for keyword, section in keyword_mapping.items():
            if keyword in query_lower:
                detected_sections.append(section)
        return list(set(detected_sections))

    def fetch_from_neo4j(self, country, section):
        text_id = f"{country}_{section}_Provision".replace(" ", "_")
        cypher_query = """
        MATCH (t:Text {text_id: $text_id})
        RETURN t.content AS text
        """
        with self.driver.session() as session:
            result = session.run(cypher_query, text_id=text_id)
            provisions = [record["text"] for record in result]
        return provisions

    def fetch_from_pinecone(self, query_text):
        query_embedding_response = self.pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[query_text],
            parameters={"input_type": "query", "truncate": "END"}
        )
        query_embedding = query_embedding_response[0]["values"]
        query_response = self.pinecone_index.query(
            vector=query_embedding,
            top_k=1,
            include_metadata=True,
            include_values=False,
            namespace="legal_data"
        )
        return query_response

    def retrieve_information(self, query, limit=2):
        retrieval_results = []
        detected_countries = self.detect_countries(query)
        detected_sections = self.detect_sections(query)
        
        if not detected_countries:
            detected_countries = [None]
        if not detected_sections:
            detected_sections = ["DEFAULT_SECTION"]
        
        for country in detected_countries:
            for section in detected_sections:
                if section != "DEFAULT_SECTION" and country:
                    neo4j_results = self.fetch_from_neo4j(country, section)
                    if neo4j_results:
                        retrieval_results.append({
                            "source": "neo4j",
                            "country": country,
                            "section": section,
                            "query": query,
                            "results": neo4j_results[:limit]
                        })
                
                pinecone_response = self.fetch_from_pinecone(query)
                pinecone_results = []
                for match in pinecone_response.matches[:limit]:
                    text = match.metadata.get("text", "") or match.metadata.get("summary", "")
                    if text:
                        score = match.metadata.get("score", "N/A")
                        pinecone_results.append({
                            "text": text,
                            "score": score,
                            "metadata": match.metadata
                        })
                if pinecone_results:
                    retrieval_results.append({
                        "source": "pinecone",
                        "country": country,
                        "section": section,
                        "query": query,
                        "results": pinecone_results
                    })
        
        return retrieval_results

# --- Chatbot with Memory and Embedding-based Context Retrieval ---
class Chatbot:
    def __init__(self):
        self.retriever = InformationRetriever()
        self.aggregator = TeammateAgent3(retriever=self.retriever)
        self.conversation_history = []

    def compute_embedding(self, text):
        response = self.retriever.pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[text],
            parameters={"input_type": "query", "truncate": "END"}
        )
        return np.array(response[0]["values"])

    def update_history(self, role, message):
        embedding = self.compute_embedding(message)
        self.conversation_history.append({"role": role, "content": message, "embedding": embedding})

    def cosine_similarity(self, vec1, vec2):
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def get_relevant_context(self, query, top_k=5):
        query_embedding = self.compute_embedding(query)
        scored_turns = []
        for turn in self.conversation_history:
            score = self.cosine_similarity(query_embedding, turn["embedding"])
            scored_turns.append((score, turn))
        scored_turns.sort(key=lambda x: x[0], reverse=True)
        top_turns = [turn for score, turn in scored_turns[:top_k]]
        context = "\n".join([f"{turn['role'].capitalize()}: {turn['content']}" for turn in top_turns])
        return context

    def chat(self, user_input):
        self.update_history("user", user_input)
        retrieval_results = self.retriever.retrieve_information(user_input)
        sub_references = []
        for result in retrieval_results:
            combined_info = ""
            if result["results"]:
                if isinstance(result["results"], list):
                    if isinstance(result["results"][0], dict):
                        texts = [r["text"] for r in result["results"]]
                        combined_info += "\n".join(texts)
                    else:
                        combined_info += "\n".join(result["results"])
                else:
                    combined_info += str(result["results"])
            if not combined_info.strip():
                combined_info = "No relevant information was retrieved."
            sub_references.append({
                "country": result.get("country", "Global"),
                "section": result.get("section", "General"),
                "reference": f"Query: {result['query']}\nRetrieved Information: {combined_info}"
            })
        
        conversation_context = self.get_relevant_context(user_input, top_k=5)
        final_answer = self.aggregator.final_aggregate_response(user_input, sub_references, "other", conversation_context=conversation_context)
        self.update_history("assistant", final_answer)
        return final_answer

# --- Main Loop ---
if __name__ == "__main__":
    chatbot = Chatbot()
    print("Welcome to the Legal Compliance Chatbot. Type 'exit' to quit.")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        final_result = chatbot.chat(user_input)
        print("\nFinal Answer:")
        print(final_result)


# # Evaluation

# In[17]:


# Re-define the data and plot with the legend outside the plot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define the data
data = np.array([
    [9, 7.5, 7.3],
    [9, 7.8, 7.5],
    [2.5, 8.8, 8.8],
    [7.5, 9, 8.5],
    [7.3, 8.3, 9],
    [7.5, 8.5, 8.8],
    [7.8, 7.8, 9]
])

# Convert to DataFrame for visualization
df = pd.DataFrame(data, columns=['Expert System', 'deepseek-r1:8b', 'Multilingual-e5-large'])

# Plot the data
plt.figure(figsize=(8, 5))
for i, col in enumerate(df.columns):
    plt.plot(range(1, 8), df[col], marker='o', label=col)

# Place the legend outside the plot
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

# Customize the plot
plt.title("Data Visualization")
plt.xlabel("Question Number")
plt.ylabel("Average Rating")
plt.xticks(range(1, 8))  # Set x-axis ticks from 1 to 7
plt.grid(True)

# Adjust layout to fit legend
plt.tight_layout()

# Display the updated plot
plt.show()


# In[ ]:




