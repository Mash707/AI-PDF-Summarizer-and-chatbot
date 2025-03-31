from main import Agent
import os

api_key = os.getenv('COHERE_API_KEY')

agent = Agent(api_key)

docs = agent.load_paper("1706.03762v7.pdf")

embeddings = agent.document_embeddings("1706.03762v7.pdf")

summary = agent.summarise("1706.03762v7.pdf")

query = 'how are transformers efficient for language generation'

query_embeddings = agent.query_embeddings(query)

similarities = agent.rag("1706.03762v7.pdf", query)

print(similarities)