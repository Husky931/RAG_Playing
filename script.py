import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.Client()

#1 Load the text
with open('Tesla2.txt', 'r') as current_file:
    text = current_file.read()

# --- B => ONE ROW PER SENTENCE (THE PRECISE WAY) -----
# sentences = [s.strip() for s in text.split('.') if s.strip()]
# coll_b = client.create_collection("sentence_strategy")
# coll_b.add(ids=[f"s{i}" for i in range(len(sentences))], documents=sentences)


# --- C => ONE ROW PER OVERLAPPING CHUNK (THE "PRO" WAY) -----
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=60)

chunks = splitter.split_text(text)
# coll_c = client.create_collection("pro_strategy")
coll_c = client.create_collection(
    name="pro_strategy",
    embedding_function=embedding_function
)
coll_c.add(ids=[f"c{i}" for i in range(len(chunks))], documents=chunks)


# --- THE TEST ---

# query3 = "Qurey3: What should never be shared with the public?!"
query3 = "Qurey3: What specific area does the blue badge grant access to?"
# query3 = "Qurey3: What color badge do I need for the cafeteria?"

print(f"Query: {query3}\n" + "="*30)
res = coll_c.query(query_texts=[query3], n_results=3)
print(f"[2]: {res['documents'][0]} \n")
print(f"[3]: {res['documents'][0][0]} \n")

