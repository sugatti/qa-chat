import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
load_dotenv()

collection_name = "qa_chat_collection"
persistent_path = ".chroma_db/"

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ['OPENAI_API_KEY'],
                model_name="text-embedding-3-small"
            )

client = chromadb.PersistentClient(path=persistent_path)
collection = client.get_collection(name=collection_name, embedding_function=openai_ef)

# 類似検索を実行するには、query()関数を使用して自然言語で質問します。
# クエリを埋め込みに変換し、類似アルゴリズムを使用して類似の結果を生成します。
# この例では、3 つの類似した結果が返されています。
results = collection.query(
    query_texts=["Databricksの導入支援実績は？"],
    n_results=3
)
print(results)