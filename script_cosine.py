import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load the text
with open('Tesla2.txt', 'r') as current_file:
    text = current_file.read()

client = chromadb.Client()

# 2. Setup the Splitter (The "Pro" Way)
# We use a slightly larger chunk here to ensure sentences stay intact
splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=10)
chunks = splitter.split_text(text)

# --- COLLECTION A: Default (L2 / Euclidean Distance) ---
# Measures straight-line distance. Smaller = More similar.
coll_l2 = client.create_collection(
    name="strategy_l2", 
    metadata={"hnsw:space": "l2"}
)
coll_l2.add(ids=[f"l2_{i}" for i in range(len(chunks))], documents=chunks)

# --- COLLECTION B: Cosine Similarity ---
# Measures the angle between vectors. Focuses on direction of meaning.
coll_cosine = client.create_collection(
    name="strategy_cosine", 
    metadata={"hnsw:space": "cosine"}
)
coll_cosine.add(ids=[f"cos_{i}" for i in range(len(chunks))], documents=chunks)

# --- 3. Testing the Query ---
query3 = "What should never be shared with the public?"

print(f"Query: {query3}")
print("="*50)

for name, coll in [("L2 (Default)", coll_l2), ("Cosine", coll_cosine)]:
    res = coll.query(query_texts=[query3], n_results=3)
    
    print(f"\n--- Strategy: {name} ---")
    # Chroma returns a list of lists because you can query multiple texts at once.
    # We take the first list [0] for our single query.
    for i, doc in enumerate(res['documents'][0]):
        # We also get 'distances' to see how confident the match is
        dist = res['distances'][0][i]
        print(f"Rank {i+1} (Dist: {dist:.4f}): {doc}")

# --- 4. Exercise: Inspecting Overlaps ---
print("\n" + "="*50)
print("CHUNK OVERLAP VISUALIZATION (First 3 Chunks):")
for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i}: [{chunk}]")