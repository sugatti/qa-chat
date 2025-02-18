import streamlit as st
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import List
from dotenv import load_dotenv
load_dotenv()

import os
import torch

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# or simply:
torch.classes.__path__ = []

llm = ChatOpenAI(model="gpt-4o-mini")

model_name = 'BAAI/bge-m3'
embeddings = HuggingFaceEmbeddings(model_name=model_name)

vector_store = Chroma(
    collection_name="qa_chat_collection",
    embedding_function=embeddings,
    persist_directory=".chroma_db_bge-m3/", 
)

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        chat_history: list of messages
    """

    question: str
    generation: str
    documents: List[str]
    chat_history: List[BaseMessage]

def init_page():
    st.set_page_config(
        page_title="AI Chat Demo App",
        page_icon="🤖"
    )
    st.header("AI Chat Demo Application 🤖")
    st.sidebar.title("Options")

def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = []

def retrieve(state):
    """
    Retrieve documents
    Args:
        state (dict): The current graph state
    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents  
    """
    print("---RETRIEVE---")
    question = state["question"]
    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5}
    )
    documents = retriever.invoke(question)
    print(f"retrieve documents: {documents}")
    return {"documents": documents, "question": question}

def generate(state):
    
    print("---GENERATE---")
    documents = state["documents"]
    docs_contents =  "\n\n".join([doc.page_content for doc in documents])

    system_message_prompt = SystemMessagePromptTemplate.from_template("""
あなたは質問応答タスクのアシスタントです。
検索された以下の文脈と会話履歴の一部を使って質問に丁寧に答えてください。
答えがわからなければ、わからないと答えてください。
最大で3つの文章を使い、簡潔な回答を心がけてください。
日本語で回答してください。
                                                                  
文脈:
====
{context}
====
""")

    prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        MessagesPlaceholder("messages"),
    ])
    messages = prompt.invoke(
        {
            'context': docs_contents,
            'messages': state["chat_history"]
        }
    )
    print(f"chat prompt: {messages}")
    response = llm.invoke(messages)
    return {'generation': response}

def main():
    init_page()

    workflow = StateGraph(state_schema=GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    # thread_idは、チェックポインターによって保存された各チェックポイントに割り当てられる一意の ID です。
    config = {"configurable": {"thread_id": "abc123"}}
    
    # チャット履歴の初期化
    init_messages()
    
    # アプリの再実行の際に履歴のチャットメッセージを表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ユーザーの入力を監視
    if user_input := st.chat_input("質問を入力してください"):
        # チャット履歴にユーザーメッセージを追加
        st.session_state.messages.append({"role": "user", "content": user_input})
        # ユーザーメッセージを表示
        st.chat_message("human").write(user_input)
        with st.spinner("QABot is typing ..."):
            response = graph.invoke(
                {
                    "question": user_input,
                    "chat_history": st.session_state.messages
                 },
                config=config,
            )
            answer = response['generation'].content
        # チャット履歴にアシスタントのレスポンスを追加
        st.session_state.messages.append({"role": "assistant", "content": answer}) 
        # アシスタントメッセージを表示
        st.chat_message("ai").write(answer)


if __name__ == '__main__':
    main()