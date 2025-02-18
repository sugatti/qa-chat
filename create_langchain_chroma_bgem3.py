from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

loader = WebBaseLoader(
    web_path="https://www.clue-tec.com",
)
pages = loader.load()
text_splitter = CharacterTextSplitter(
    chunk_size=200, 
    separator='\n', 
    chunk_overlap=50
)
documents = text_splitter.split_documents(pages)

#embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
model_name = 'BAAI/bge-m3'
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Chroma は、OpenAI の埋め込み API を包む便利なラッパーを提供します。
vector_store = Chroma(
    collection_name="qa_chat_collection",
    embedding_function=embeddings,
    persist_directory=".chroma_db_bge-m3/",  # Chromaがデータベースファイルをディスク上に保存し、起動時に読み込む場所です
)

# add_documents()関数を使用して、メタデータと一意の ID を含むテキスト データを追加します。
# その後、Chroma は自動的にコレクションで指定した埋め込み関数を使用してテキストを埋め込みに変換し、コレクションに保存します。
from uuid import uuid4
uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents,ids=uuids)