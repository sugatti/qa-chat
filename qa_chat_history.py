from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# サンプルデータをベクトルストアに追加
documents = [
    {"content": "須賀้秀和（すがひでかず）はタイ料理が好きです。", "metadata": {"source": "doc1"}},
    {"content": "須賀้秀和（すがひでかず）はフリーランスのエンジニアで、タイ在住です。", "metadata": {"source": "doc2"}},
]
texts = [doc["content"] for doc in documents]
metadatas = [doc["metadata"] for doc in documents]
vector_store = FAISS.from_texts(texts, embeddings, metadatas)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

def generate(state):
    question = state["messages"][-1].content
    retrieved_docs = vector_store.similarity_search(query=question,k=2)
    docs_contents =  "\n\n".join([doc.page_content for doc in retrieved_docs])

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
            'messages': state["messages"]
        }
    )
    for msg in messages.messages:
        print(f"prompt msg-->{msg}")
    response = llm.invoke(messages)
    return {'messages': response}
        

workflow = StateGraph(state_schema=MessagesState)

workflow.add_node("generate", generate)
workflow.add_edge(START, "generate")
workflow.add_edge("generate", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

# thread_idは、チェックポインターによって保存された各チェックポイントに割り当てられる一意の ID です。
config = {"configurable": {"thread_id": "abc123"}}

#Image(graph.get_graph().draw_mermaid_png())
graph.get_graph().print_ascii()

def main():
    print("質問をどうぞ。'quit' または 'exit' を入力すると終了します。")
    while True:
        user_input = input("user : ")

        # 終了条件
        if user_input.lower() in ["quit", "exit"]:
            print("アプリケーションを終了します。")
            break
        
        response = graph.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        # 履歴の最後をメッセージとして出力する
        print(f"AI : {response['messages'][-1].content}")


if __name__ == "__main__":
    main()
