from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = Chroma(
    collection_name="qa_chat_collection",
    embedding_function=embeddings,
    persist_directory=".chroma_db/", 
)

# 類似検索を実行するには、similarity_search()関数を使用して自然言語で質問します。
# クエリを埋め込みに変換し、類似アルゴリズムを使用して類似の結果を生成します。
# この例では、3 つの類似した結果が返されています。
results = vector_store.similarity_search(
    "Databricksの導入支援実績は？",
    k=3,
    filter={"source": "https://www.clue-tec.com"},
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")

# langchain内での使用を容易にするために、Vector Storeをretrieverに変換することもできます。
# 渡すことができるさまざまな検索タイプと kwargs の詳細については、ここにあるAPI リファレンスを参照してください。
retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5}
)
response = retriever.invoke("Databricksの導入支援実績は？", filter={"source": "https://www.clue-tec.com"})

print(response)