import openai
from langchain.text_splitter import CharacterTextSplitter
from langsmith.evaluation import LangChainStringEvaluator, evaluate
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from databricks_langchain import ChatDatabricks, DatabricksEmbeddings
from dotenv import load_dotenv
load_dotenv()


loader = WebBaseLoader(
    web_path="https://www.clue-tec.com",
)
pages = loader.load()
text_splitter = CharacterTextSplitter(
    chunk_size=200, 
    separator='', 
    chunk_overlap=50
)
documents = text_splitter.split_documents(pages)

embeddings = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

faiss_index = FAISS.from_documents(documents, embeddings)
retriever = faiss_index.as_retriever(search_kwargs={"k": 3})

llm = ChatDatabricks(temperature=0, endpoint="databricks-dbrx-instruct")

class RagBot:
    def __init__(self, retriever, model: str = "gpt-4o-mini"):
        self._retriever = retriever
        # Wrap the OpenAI client for tracing
        self._client = wrap_openai(openai.Client())
        self._model = model

    @traceable()
    def retrieve_docs(self, question):
        # Retrieve relevant documents for the given question
        return self._retriever.invoke(question)

    @traceable()
    def invoke_llm(self, question, docs):
        # Use the retrieved documents to generate a response
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
            {"role": "user", "content": question},
        ])
        messages = prompt.invoke(
            {
                'context': docs,
            }
        )
        response = llm.invoke(messages)

        return {
            #"answer": response.choices[0].message.content,
            "answer": response.content,
            "contexts": [str(doc) for doc in docs],
        }

    @traceable()
    def get_answer(self, question: str):
        # Get the answer by retrieving documents and invoking the LLM
        docs = self.retrieve_docs(question)
        return self.invoke_llm(question, docs)

# Instantiate the RAG bot
rag_bot = RagBot(retriever)

# Get an answer to a sample question
response = rag_bot.get_answer("須賀秀和の職業は何ですか？")

# RAG chain functions
def predict_rag_answer(example: dict):
    """Use this for answer evaluation"""
    response = rag_bot.get_answer(example["question"])
    return {"answer": response["answer"]}

def predict_rag_answer_with_context(example: dict):
    """Use this for evaluation of retrieved documents and hallucinations"""
    response = rag_bot.get_answer(example["question"])
    return {"answer": response["answer"], "contexts": response["contexts"]}

# Evaluator for comparing RAG answers to reference answers
qa_evaluator = [
    LangChainStringEvaluator(
        "cot_qa",  # Using Chain-of-Thought QA evaluator
        prepare_data=lambda run, example: {
            "prediction": run.outputs["answer"],  # RAG system's answer
            "reference": example.outputs["answer"],  # Ground truth answer
            "input": example.inputs["question"],  # Original question
        },
    )
]

dataset_name = "Sample dataset"
experiment_results = evaluate(
    predict_rag_answer,
    data=dataset_name,
    evaluators=qa_evaluator,
    experiment_prefix="rag-qa-oai",
    metadata={"variant": "LCEL context, gpt-3.5-turbo"},
)
