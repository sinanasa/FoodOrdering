from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from config import load_config, get_groq_api, get_openai_api
from langchain.chat_models import ChatOpenAI

# load app configuration
load_config()


# setup groq LLM
def groq_llm():
    # llm = ChatGroq(groq_api_key=get_groq_api(), model_name='Llama3-8b-8192')
    llm = ChatOpenAI(api_key=get_openai_api(), model='gpt-3.5-turbo',temperature=0.8)
    return llm

# setup huggingface_instruct_embedding
def huggingface_instruct_embedding():
    embeddings = HuggingFaceBgeEmbeddings(
                model_name='BAAI/bge-small-en-v1.5',  #sentence-transformers/all-MiniLM-l6-v2
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings