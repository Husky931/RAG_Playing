import chromadb
# from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

#1 Load the text
with open('Tesla2.txt', 'r') as current_file:
    text = current_file.read()

client = chromadb.Client()

# --- A => ONE ROW PER DOCUMENT (THE LAZY WAY) -----
coll_a = client.create_collection("lazy_strategy")
coll_a.add(ids=["doc1"], documents=[text])


# --- B => ONE ROW PER SENTENCE (THE PRECISE WAY) -----
sentences = [s.strip() for s in text.split('.') if s.strip()]
coll_b = client.create_collection("sentence_strategy")
coll_b.add(ids=[f"s{i}" for i in range(len(sentences))], documents=sentences)


# --- C => ONE ROW PER OVERLAPPING CHUNK (THE "PRO" WAY) -----
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_text(text)
coll_c = client.create_collection("pro_strategy")
coll_c.add(ids=[f"c{i}" for i in range(len(chunks))], documents=chunks)


# --- THE TEST ---
query = "Qurey1: How much coins can seniors chew per day"
# print(f"Query: {query}\n" + "="*30)
# for name, coll in [("Sentence", coll_b), ("Pro", coll_c)]:
#     res = coll.query(query_texts=[query], n_results=1)
#     print(f"[{name} Result]: {res['documents'][0][0]}")


    # --- THE TEST 2 ---
# query2 = "Qurey2: Whats the security rule for Tesla employees"
# # query = "What is the meal limit for Senior managers?"
# print(f"Query: {query2}\n" + "="*30)
# for name, coll in [("Sentence", coll_b), ("Pro", coll_c)]:
#     res = coll.query(query_texts=[query2], n_results=1)
#     print(f"[{name} Result]: {res['documents'][0][0]}")

query3 = "Qurey3: What should never be shared with the public?"
# query = "What is the meal limit for Senior managers?"
print(f"Query: {query3}\n" + "="*30)
for name, coll in [("Sentence", coll_b), ("Pro", coll_c)]:
    res = coll.query(query_texts=[query3], n_results=1)
    print(f"[{name} Result]: {res['documents'][0][0]}")