from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.chat_engine import SimpleChatEngine, CondenseQuestionChatEngine
from llama_index.core.memory import ChatMemoryBuffer
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
# import socket
# from llama_index.llms.mistralai import MistralAI
# Need MISTRAL_API_KEY
# from llama_index.llms.huggingface import HuggingFaceLLM
# from openai import OpenAI
import os


llm = OpenAI(temperature=0.8, model="gpt-4")
# llm = OpenAI()
# llm = MistralAI(temperature=0.8, model="mistral-large-latest")
# llm = MistralAI()

try:
    chat_store = SimpleChatStore.from_persist_path(
        persist_path="chat_memory2.json"
    )
except FileNotFoundError:
    chat_store = SimpleChatStore()


memory = ChatMemoryBuffer.from_defaults(
    token_limit=2000,
    chat_store=chat_store,
    chat_store_key="user_X"
    )

db = chromadb.PersistentClient(path="chroma_database")
chroma_collection = db.get_or_create_collection(
    "my_chroma_store"
)
vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
)
storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

if chroma_collection.count() > 0:
    # Rebuild the Index from the ChromaDB in future sessions
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )
    print("Vector db found. Running using db to build index...")

    # context = [{'role': 'system', 'content': """
    # You are OrderBot, an automated service to collect orders for a pizza restaurant. \
    # You first greet the customer, then collects the order, \
    # and then asks if it's a pickup or delivery. \
    # You wait to collect the entire order, then summarize it and check for a final \
    # time if the customer wants to add anything else. \
    # If it's a delivery, you ask for an address. \
    # Finally you collect the payment.\
    # Make sure to clarify all options, extras and sizes to uniquely \
    # identify the item from the menu.\
    # You respond in a short, very conversational friendly style. \
    # The menu includes \
    # pepperoni pizza  12.95, 10.00, 7.00 \
    # cheese pizza   10.95, 9.25, 6.50 \
    # eggplant pizza   11.95, 9.75, 6.75 \
    # fries 4.50, 3.50 \
    # greek salad 7.25 \
    # Toppings: \
    # extra cheese 2.00, \
    # mushrooms 1.50 \
    # sausage 3.00 \
    # canadian bacon 3.50 \
    # AI sauce 1.50 \
    # peppers 1.00 \
    # Drinks: \
    # coke 3.00, 2.00, 1.00 \
    # sprite 3.00, 2.00, 1.00 \
    # bottled water 5.00 \
    # """}]  # accumulate messages

    # messages = context.copy()
    # messages.append(
    #     {'role': 'system', 'content': 'create a json summary of the previous food order. Itemize the price for each item\
    #  The fields should be 1) pizza, include size 2) list of toppings 3) list of drinks, include size   4) list of sides include size  5)total price '},
    # )
    # The fields should be 1) pizza, price 2) list of toppings 3) list of drinks, include size include price  4) list of sides include size include price, 5)total price '},

    # define prompt viewing function
    def display_prompt_dict(prompts_dict):
        for k, p in prompts_dict.items():
            text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
            print(text_md)
            print(p.get_template())
            print("***************")

    query_engine = index.as_query_engine(
        chat_mode="context",
        memory=memory,
        # messages=messages,
        system_prompt=(
            "You are a chatbot, able to take food orders from customers."
            "You take orders and provide information related to menu items only when asked. Answer questions based on the context provided only."
        ),
    )

    # Start the conversation by saying Thank you for calling Doner Point, how may I assist you today? \

    # Order taking prompt
    new_text_qa_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "You are Doner Point, an automated service to collect orders for a restaurant. \
        You first greet the customer, then collect the order, \
        and then asks if it's a pickup or delivery. \
        You wait to collect the entire order, then summarize it and check for a final \
        time if the customer wants to add anything else. \
        If it's a delivery, you ask for an address. \
        You do not collect the payment information, payments are collected at delivery.\
        Make sure to clarify all options, extras and sizes to uniquely \
        identify the item from the menu. \
        Offer items that are not in customers order but in specials. \
        You respond in a short, very conversational friendly style. \
        Create a json summary of the previous food order. Itemize the price for each item\
        The fields should be 1) menu item, include size 2) list of desserts, inlcude size or options 3) list of drinks, include size   4) pickup or delivery  5)total price\
        "
        "Query: {query_str}\n"
        "Answer: "
    )
    new_tmpl = PromptTemplate(new_text_qa_tmpl_str)
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": new_tmpl}
    )


    # Debug
    # prompts_dict = query_engine.get_prompts()
    # display_prompt_dict(prompts_dict)

    # chat_engine = SimpleChatEngine.from_defaults(memory=memory, messages=messages, query_engine=query_engine, llm=llm)
    # chat_engine = SimpleChatEngine.from_defaults(memory=memory, query_engine=query_engine, llm=llm)


    # chat_engine = CondenseQuestionChatEngine.from_defaults(memory=memory, condense_question_prompt=messages, query_engine=query_engine, llm=llm)
    chat_engine = CondenseQuestionChatEngine.from_defaults(memory=memory, query_engine=query_engine, llm=llm)

    #Debug
    # response = query_engine.query("What are the store hours?")
    # print(f" {response} ")


else:
    print("No vector db found.  Running without db...")
    chat_engine = SimpleChatEngine.from_defaults(memory=memory, llm=llm)


while True:
    user_message = input("You: ")
    if user_message.lower() == 'exit':
        print("Exiting chat...")
        break
    # response = get_completion_from_messages(messages, temperature=0)
    response = chat_engine.chat(user_message)
    print(f"Chatbot: {response}")

chat_store.persist(persist_path="chat_memory.json")



# # Server app
# # Define the host and port
# HOST = '127.0.0.1'  # Localhost
# PORT = 65432  # Port to listen on (non-privileged ports are > 1023)
#
# # Create a socket object
# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     # Bind the socket to the host and port
#     s.bind((HOST, PORT))
#
#     # Listen for incoming connections
#     s.listen()
#     print(f"Server listening on {HOST}:{PORT}")
#
#     while True:
#         # Accept a connection
#         conn, addr = s.accept()
#         with conn:
#             print(f"Connected by {addr}")
#
#             # Receive data from the client
#             while True:
#                 data = conn.recv(1024)
#                 if not data:
#                     break
#
#                 # Print the received data
#                 user_message = data.decode('utf-8')
#                 print(f"Received: {user_message}")
#
#                 if user_message.lower() == 'exit':
#                     print("Exiting chat...")
#                     break
#
#                 # Send a response to the client
#                 response = chat_engine.chat(user_message)
#                 print(f"Response: {response}")
#                 conn.sendall(str(response).encode('utf-8'))
#
# # End Server App