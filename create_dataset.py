from langsmith import wrappers, Client
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = Client()
openai_client = wrappers.wrap_openai(OpenAI())

# For other dataset creation methods, see: https://docs.smith.langchain.com/evaluation/how_to_guides/manage_datasets_programmatically https://docs.smith.langchain.com/evaluation/how_to_guides/manage_datasets_in_application

# Create inputs and reference outputs
examples = [
  (
      "須賀秀和はどのようなサービスを導入しましたか？",
      "DatabricksやFivetranとSnowflake、dbtのモダンデータスタックによるデータ分析基盤を導入しました",
  ),
  (
      "須賀秀和が支援している業務は何ですか？",
      "分散システムの開発やデータ分析基盤の導入を技術面から支援しています",
  ),
]

inputs = [{"question": input_prompt} for input_prompt, _ in examples]
outputs = [{"answer": output_answer} for _, output_answer in examples]

# Programmatically create a dataset in LangSmith
dataset = client.create_dataset(
	dataset_name = "Sample dataset",
	description = "A sample dataset in LangSmith."
)

# Add examples to the dataset
client.create_examples(inputs=inputs, outputs=outputs, dataset_id=dataset.id)