# 🧠 A Multi-Agent Expert System for Global Data Regulation Compliance
Welcome to the repository for our capstone project: A Multi-Agent Expert System for Global Data Regulation Compliance. This project tackles the complexity of modern data privacy laws by building an intelligent, multi-agent legal assistant capable of dynamically retrieving, synthesizing, and reasoning about regulatory requirements across jurisdictions.

# 🌍 Overview
Global data privacy regulations such as GDPR, CCPA, and LGPD are rapidly evolving, creating significant compliance challenges. Traditional legal AI tools—especially those relying on a single large language model—struggle with:
- Multi-jurisdictional comparisons
- Ambiguous or implicit queries
- Long-term conversational context
- Regulatory updates and interpretation

To address these challenges, this project introduces a retrieval-grounded, multi-agent expert system that combines structured graph-based search, semantic vector retrieval, and LLM-based reasoning.

#  ⚙️ System Architecture
The system consists of four primary components:
1. Manager Agent
   - Decomposes complex legal queries
   - Chooses between semantic and structured retrieval
   - Delegates tasks to teammate agents
3. Teamate Agents
   - Agent 1: Semantic search via Pinecone
   - Agent 2: Structured retrieval from Neo4j graph database
   - Agent 3: Aggregates, refines, and synthesizes final responses
5. Conversational Memory System
   - Embedding-based long-term memory
   - Enables contextual continuity in multi-turn interactions
7. Gap Dectection & Refinement
   - Detects missing legal provisions
   - Issues follow-up queries before final synthesis

# 🧪 Technologies Used

| Area              | Tools                     |
|-------------------|---------------------------|
| Embeddings        | `Multilingual-E5 Large`   |
| Language Model    | `DeepSeek-R1:8B`          |
| Semantic Search   | `Pinecone`                |
| Graph Retrieval   | `Neo4j`                   |
| Preprocessing     | `PyMuPDF`, `Pandas`       |
| Evaluation        | ChatGPT, DeepSeek         |


#  📊 Evaluation
The system was tested using 7 benchmark queries across three categories:
- Comparison: Jurisdictional differences
- Procedural: Step-by-step compliance
- Explanation: Legal concept interpretation

## 🔍 Key Findings
- Outperforms single-LLM baselines on procedural and comparative tasks
- Memory system improves multi-turn consistency
- Underperforms in purely explanatory queries (future area of improvement)

#  License
This code is available for non-commercial academic use only. For other use cases, please contact: hantaolin520@gmail.com
