from chromadb.config import Settings
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import torch
import chromadb
from chromadb.config import Settings
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

MODEL_NAME = "IlyaGusev/saiga_mistral_7b"
config = PeftConfig.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

tokenizer.bos_token = "<s>"
tokenizer.eos_token = "</s>"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="cuda"
)
model = PeftModel.from_pretrained(
    model,
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cuda"
)
model = model.merge_and_unload() #  ОБЯЗАТЕЛЬНО!!!
model.eval()

generation_config = GenerationConfig(
  bos_token_id = 1,
  do_sample = True,
  eos_token_id = 2,
  max_length = 2048,
  repetition_penalty=1.1,
  no_repeat_ngram_size=15,
  pad_token_id = 0,
  temperature = 0.2,
  top_p = 0.9
)

pipe = pipeline(
        'text-generation',
        model = model,
        tokenizer = tokenizer,
        generation_config=generation_config
    )

llm = HuggingFacePipeline(pipeline=pipe)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

chroma_client = chromadb.HttpClient(settings=Settings(
    allow_reset=True,
    chroma_api_impl='chromadb.api.fastapi.FastAPI',
    chroma_server_host='localhost',
    chroma_server_http_port='8000')
)

db = Chroma(client=chroma_client,
            collection_name='book',
            embedding_function=embeddings)

retriever = db.as_retriever(
    search_kwargs={"k":3}
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Ответь на следующий вопрос, используя только эту информацию:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)

retriever_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "На основе диалога выше, сгенерируй поисковый запрос, чтобы найти релевантную информацию")
])

history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=retriever_prompt
)

retrieval_chain = create_retrieval_chain(
    history_aware_retriever,
    chain
)

chat_history = [
    HumanMessage(content='Привет, я обращаюсь к тебе за помощью только по вопросам лотерей Столото. Меня интересует только эта информация.'),
    AIMessage(content='Привет, меня зовут Степан, меня создали гениальные ML-разработчики из компании "Синхро". Я готов помочь тебе с твоими вопросами!')
]


while True:
    input_message = input("Вы: ")
    if input_message.lower() == 'exit':
        break

    response = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": input_message
    })['answer']
    chat_history.append(HumanMessage(content=input_message))
    chat_history.append(AIMessage(content=response))

    print(f'AI-ассистент: {response}\n')