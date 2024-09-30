from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.chat_engine import SimpleChatEngine, CondenseQuestionChatEngine, ContextChatEngine
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
import json
from RestaurantOrder import RestaurantOrder

class orderChat:

    def __init__(
        self: object,
    ) -> None:
        llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
        # llm = MistralAI(temperature=0.8, model="mistral-large-latest")

        # try:
        #     self.chat_store = SimpleChatStore.from_persist_path(
        #         persist_path="chat_memory.json"
        #     )
        # except FileNotFoundError:
        #     self.chat_store = SimpleChatStore()

        # memory = ChatMemoryBuffer.from_defaults(
        #     token_limit=2000,
        #     chat_store=self.chat_store,
        #     chat_store_key="user_X"
        #     )

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

            self.retriever = index.as_retriever(retriever_mode='default')

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


            self.query_engine = index.as_query_engine(
                chat_mode="context",
                # memory=memory,
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
                After customer finishes ordering summarize it and check for a final \
                time if the customer wants to add anything else. \
                Make sure to clarify all options, extras and sizes to uniquely identify the item from the menu.\
                If customer did not order any appetizers or desserts, offer popular items from appetizers and desserts. \
                Once order is completed, then asks if it's a pickup or delivery. \
                If it's a delivery, you ask for an address. \
                You do not collect the payment information, payments are collected at delivery.\
                You respond in a short, very conversational friendly style. \
                Create a json summary of the food order. Itemize the price for each item\
                The fields should be 1)menu items ordered, include size, and price 2)pickup or delivery. Include address if delivery is selected 3)total price.\
                If the customer specifies delivery vs. pickup add this information to the response as a json message:\
                Example json message: {order_type: delivery} \
                "
                "Query: {query_str}\n"
                "Answer: "
            )
            new_tmpl = PromptTemplate(new_text_qa_tmpl_str)
            self.query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": new_tmpl}
            )

            # Debug
            # prompts_dict = self.query_engine.get_prompts()
            # self.display_prompt_dict(prompts_dict)


            # chat_engine = SimpleChatEngine.from_defaults(memory=memory, messages=messages, query_engine=query_engine, llm=llm)
            # self.chat_engine = SimpleChatEngine.from_defaults(memory=memory, query_engine=self.query_engine, llm=llm)


            # chat_engine = CondenseQuestionChatEngine.from_defaults(memory=memory, condense_question_prompt=messages, query_engine=query_engine, llm=llm)
            # self.chat_engine = CondenseQuestionChatEngine.from_defaults(query_engine=self.query_engine, llm=llm)
            self.chat_engine = ContextChatEngine.from_defaults(retriever=self.retriever, query_engine=self.query_engine, llm=llm)


        else:
            print("No vector db found.  Running without db...")
            chat_engine = SimpleChatEngine.from_defaults(llm=llm)

        #Debug
        # response = query_engine.query("What are the store hours?")
        # print(f" {response} ")

    # define prompt viewing function
    def display_prompt_dict(self, prompts_dict):
        for k, p in prompts_dict.items():
            text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
            print(text_md)
            print(p.get_template())
            print("***************")


    def setPromptInitiateConvo(self, query_engine):
            # Order taking prompt
            new_text_qa_tmpl_str = (
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the context information and not prior knowledge, "
                "You are Doner Point, an automated service to collect orders for a restaurant.\
                Customer already started the conversation. Please reply according to customers initial interaction. You have already greeted the customer.\
                "
                "Query: {query_str}\n"
                "Answer: "
            )
            new_tmpl = PromptTemplate(new_text_qa_tmpl_str)
            query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": new_tmpl}
            )

    def setPromptOrderTaking(self, query_engine, order):
            # Order taking prompt
            new_text_qa_tmpl_str = (
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the context information and not prior knowledge, "
                "You are Doner Point, an automated service to collect orders for a restaurant. \
                Identify what customers request is and respond in a short, very conversational friendly style.\
                Create a json summary of the request. and return a proper json message containing request type. \
                Request type can be add to order, remove from order, modify order item, special instructions, address, general information, order completed, pickup or delivery. \
                For request types of add to order, remove from order, modify order item, specify order item, \
                the fields should be 1) menu_item_ordered 2) quantity 3) size 4) price.\
                "
                "Query: {query_str}\n"
                "Answer: "
            )
            new_tmpl = PromptTemplate(new_text_qa_tmpl_str)
            query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": new_tmpl}
            )

    def setPromptOrderTaking2(self, query_engine, order):
            # Order taking prompt
            new_text_qa_tmpl_str = (
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the context information and not prior knowledge, "
                "You are Doner Point, an automated service to collect orders for a restaurant. \
                Identify what customers request is clearly and respond in a short, very conversational friendly style. \
                If the customer did not specify the size of the item, ask customer to specify.\
                Create a json summary of the request. and return a proper json message containing request type. \
                Request type can be add to order, remove from order, modify order item, special instructions, address, general information, order completed, pickup or delivery. \
                For request types of add to order, remove from order, modify order item, specify order item, \
                the fields should be 1) menu_item_ordered 2) quantity 3) size 4) price.\
                If the ordering is done return a json with only request_type value order completed.\
                Following is the current customer order: \
                1 small lentil soup, 1 large gobit"
                "Query: {query_str}\n"
                "Answer: "
            )
            new_tmpl = PromptTemplate(new_text_qa_tmpl_str)
            query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": new_tmpl}
            )

    def setPromptOrderTaking3(self, query_engine, order):
            # Order taking prompt
            new_text_qa_tmpl_str = (
                "Context information is below.\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Given the context information and not prior knowledge, "
                "You are Doner Point, an automated service to collect orders for a restaurant. \
                You have started the conversation and it possible that the customer is done ordering.\
                Customer have already ordered a small chiken soup and a small gobit.\
                If the customer wants to finish the order or the call return a json with only request_type order completed.\
                If the customer is done and does not want to add anything else, thank the customer and exit the chat.\
                Following is the current customer order:\
                "
                "Query: {query_str}\n"
                "Answer: "
            )
            new_tmpl = PromptTemplate(new_text_qa_tmpl_str)
            query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": new_tmpl}
            )

    def chatAway(self, user_message):
        return self.chat_engine.chat(user_message)

    def chatLoop(self):
        # main loop
        print(f"Chatbot: Thank you for calling Doner Point! How may I help you?")
        order = RestaurantOrder()
        self.setPromptInitiateConvo(self.query_engine)

        # Debug
        # prompts_dict = self.query_engine.get_prompts()
        # self.display_prompt_dict(prompts_dict)


        counter=0
        while True:
            user_message = input("You: ")
            if user_message.lower() == 'exit':
                print("Exiting chat...")
                break

            response = self.chat_engine.chat(user_message)
            # process response, if add/remove/modify add to orderList
            #                   if address, update address
            #                   if general info reply to customer


            try:
                #check if the reponse is proper json
                # if yes process
                # else reply to customer
                response_str = str(response).replace('\n','')
                jsonResponse = json.loads(response_str)
                if 'general' in jsonResponse['request_type'].replace('_', ' '):
                    print(f"Chatbot: {jsonResponse['response']}")
                elif 'add' in jsonResponse['request_type'].replace('_', ' '):
                    order.add_item(jsonResponse['menu_item_ordered'], jsonResponse['quantity'], jsonResponse['size'])
                    print(f"Chatbot: Added {jsonResponse['menu_item_ordered']} to the order")
                elif 'completed' in jsonResponse['request_type'].replace('_', ' '):
                    self.setPromptOrderTaking3(self.query_engine, order.order_summary())
                if len(order) == 0:
                    order.append(jsonResponse)
            except:
                print(f"Chatbot: {response}")

            if counter==0:
                self.setPromptOrderTaking(self.query_engine, order.order_summary())
            elif counter==1:
                self.setPromptOrderTaking2(self.query_engine, order.order_summary())


            counter=counter+1

        # self.chat_store.persist(persist_path="chat_memory.json")


# main app
foodOrderChat = orderChat()
foodOrderChat.chatLoop()



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


