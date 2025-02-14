import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

collection_name = "qa_chat_collection"
persistent_path = ".chroma_db/"

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

# Chroma は、OpenAI の埋め込み API を包む便利なラッパーを提供します。
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ['OPENAI_API_KEY'],
                model_name="text-embedding-3-small"
            )

# Chromaがデータベースファイルをディスク上に保存し、起動時に読み込む場所です。パスを指定しない場合、デフォルトは ".chroma"になります
client = chromadb.PersistentClient(path=persistent_path)

# コレクションは、埋め込み、ドキュメント、その他のメタデータを保存する場所です。
# コレクションは埋め込みとドキュメントをインデックス化し、効率的な検索とフィルタリングを可能にします。
collection = client.create_collection(
    name=collection_name, 
    embedding_function=openai_ef,
    metadata={
        "description": "QA Chat Chroma collection",
        "created": str(datetime.now())
    }  
)

# add()関数を使用して、メタデータと一意の ID を含むテキスト データを追加します。
# その後、Chroma は自動的にコレクションで指定した埋め込み関数を使用してテキストを埋め込みに変換し、コレクションに保存します。
for i in range(len(documents)):
    collection.add(
        documents=documents[i].page_content,
        metadatas=documents[i].metadata,
        ids=str(i)
    )