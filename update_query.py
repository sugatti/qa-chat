from langgraph.graph import START, END, StateGraph
from langchain_community.vectorstores import FAISS
from databricks_langchain import ChatDatabricks, DatabricksEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
load_dotenv()

llm = ChatDatabricks(
    temperature=0, 
    endpoint="databricks-dbrx-instruct")

#embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

# サンプルデータをベクトルストアに追加
documents = [
    {"content": "須賀้秀和（すがひでかず）はタイ料理が好きです。", "metadata": {"source": "doc1"}},
    {"content": "須賀้秀和（すがひでかず）はメソドロジックのエンジニアで、タイ在住です。", "metadata": {"source": "doc2"}},
]
texts = [doc["content"] for doc in documents]
metadatas = [doc["metadata"] for doc in documents]
vector_store = FAISS.from_texts(texts, embeddings, metadatas)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

class State(MessagesState):
    question: str

def process_message(state):
    message_history = state["messages"][:-1]
    if len(message_history) >= 4:

        system_prompt = (
            "以下は、ユーザとQAアシスタントの会話です。"
            "あなたの仕事は、この会話履歴を踏まえて、ベクトルストアを検索するための自然言語クエリを生成することです。"
        )
        human_template = (
            "以下は上記の会話履歴のあとにユーザが行ったクエリです。このクエリを、会話履歴の文脈に関係なく解釈できる独立したクエリに書き換えてください:\n{question}"
        )
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder("history"),
            human_message_prompt,
        ])
        messages = prompt.invoke({'history': message_history, 'question': state["question"]})
        for msg in messages.messages:
            print(f"query msg-->{msg}")
        response = llm.invoke(messages.messages)
        print(f"update query: {response}")
        query_updates = response
    else:
        query_updates = state["messages"][-1]
        
    return {"question": query_updates}

def generate(state):
    print("--Retrieving Information--")
    question = state["question"].content
    retrieved_docs = vector_store.similarity_search(query=question,k=2)
    print("--Generating Response--")
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

    human_template="{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        MessagesPlaceholder("history"),
        human_message_prompt,
    ])
    messages = prompt.invoke(
        {
            'question': state["messages"][-1], 
            'history': state["messages"][:-1], 
            'context': docs_contents
        }
    )
    for msg in messages.messages:
        print(f"prompt msg-->{msg}")
    response = llm.invoke(messages)
    print(f"response--> {response}")
    return {'messages': response}
        

workflow = StateGraph(state_schema=State)

workflow.add_node("process_message", process_message)
workflow.add_node("generate", generate)
workflow.add_edge(START, "process_message")
workflow.add_edge("process_message", "generate")
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
        print(f"AI : {response["messages"][-1].content}")


if __name__ == "__main__":
    main()
