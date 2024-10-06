from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.chat_engine import SimpleChatEngine, CondenseQuestionChatEngine, ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate, ChatPromptTemplate
# import socket
# from llama_index.llms.mistralai import MistralAI
# Need MISTRAL_API_KEY
# from llama_index.llms.huggingface import HuggingFaceLLM
# from openai import OpenAI
import os
import json
from RestaurantOrder import RestaurantOrder
from llama_index.core.llms import ChatMessage, MessageRole
from GmailSender import GmailSender


class orderChat:

    def __init__(
        self: object,
    ) -> None:
        # llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
        llm = OpenAI(temperature=0.1, model="gpt-4-turbo")
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

            # Text QA Prompt
            chat_text_qa_msgs = [
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=(
                        "You are an expert Q&A system that is trusted around the world.\n"
                        "Always answer the query using the provided context information, "
                        "Your name is Hasan. Introduce yourself in the beginning."
                        "and not prior knowledge.\n"
                        "Some rules to follow:\n"
                        "1. Never directly reference the given context in your answer.\n"
                        "2. Avoid statements like 'Based on the context, ...' or "
                        "'The context information ...' or anything along "
                        "those lines."
                    ),
                ),
                ChatMessage(
                    role=MessageRole.USER,
                    content=(
                        "Context information is below.\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Given the context information and not prior knowledge, "
                        "Your name is Husmen. Introduce yourself in the beginning."
                        "Start every response saying Dear Sir,"                        
                        "answer the query.\n"
                        "Query: {query_str}\n"
                        "Answer: "
                    ),
                ),
            ]
            text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)

            # Refine Prompt
            chat_refine_msgs = [
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=(
                        "You are an expert Q&A system that strictly operates in two modes "
                        "when refining existing answers:\n"
                        "1. **Rewrite** an original answer using the new context.\n"
                        "2. **Repeat** the original answer if the new context isn't useful.\n"
                        "Never reference the original answer or context directly in your answer.\n"
                        "When in doubt, just repeat the original answer."
                        "Your name is Macit. Introduce yourself in the beginning."
                    ),
                ),
                ChatMessage(
                    role=MessageRole.USER,
                    content=(
                        "New Context: {context_msg}\n"
                        "Query: {query_str}\n"
                        "Original Answer: {existing_answer}\n"
                        "New Answer: "
                    ),
                ),
            ]
            refine_template = ChatPromptTemplate(chat_refine_msgs)



            # Prompts defined here gets lost, not passed down to chatEngine - possible a bug in the lib
            self.query_engine = index.as_query_engine(
                chat_mode="context",
                text_qa_template=text_qa_template,
                refine_template=refine_template,
                system_prompt=(
                    "You are a chatbot, able to take food orders from customers."
                    "You take orders and provide information related to menu items only when asked. Answer questions based on the context provided only."
                ),
            )

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
                Make sure to clarify all options, extras and sizes to uniquely identify the item from the menu. \
                If the item is not on the menu tell the customer politely that item cannot be ordered.\
                You can use synonyms for menu items. \
                Your answers will be read over the phone so do not provide any formatting for reading. \
                If customer did not order any appetizers or desserts, offer popular items from appetizers and desserts. \
                Once order is completed, then asks if it's a pickup or delivery. \
                If it's a delivery, you ask for an address. \
                You do not collect the payment information, payments are collected at delivery.\
                You respond in a short, very conversational friendly style. \
                Create a json summary of the food order. Itemize the price for each item\
                The fields should be 1)menu items ordered, include size, quantity, and price 2)pickup or delivery. Include address if delivery is selected 3)total price.\
                Translate all values in json message to English. \
                If the customer specifies delivery vs. pickup add this information to the response as a json message: \
                Example json message: \
                    { \
                      'menu_items_ordered': [\
                        {\
                          'item': 'Shepherd Salad',\
                          'size': 'Regular', \
                          'quantity': 1, \
                          'price': '$8.95'\
                        },\
                        {\
                          'item': 'Gobit',\
                          'size': 'Small', \
                          'quantity': 1, \
                          'price': '$12.95'\
                        },\
                        {\
                          'item': 'Baklava with pistachios',\
                          'size': 'Regular', \
                          'quantity': 1, \
                          'price': '$6.95'\
                        }\
                      ],\
                      'pickup_or_delivery': 'delivery',\
                      'address': '5343 Bell Blvd, Bayside NY',\
                      'total_price': '$28.85'\
                    } "                
                "Query: {query_str}\n"
                "Answer: "
            )
            new_tmpl = PromptTemplate(new_text_qa_tmpl_str)
            # self.query_engine.update_prompts(
            #     {"response_synthesizer:text_qa_template": new_tmpl}
            # )

            # Debug
            # prompts_dict = self.query_engine.get_prompts()
            # self.display_prompt_dict(prompts_dict)

            # chat_engine = SimpleChatEngine.from_defaults(memory=memory, messages=messages, query_engine=query_engine, llm=llm)
            # self.chat_engine = SimpleChatEngine.from_defaults(memory=memory, query_engine=self.query_engine, llm=llm)

            # chat_engine = CondenseQuestionChatEngine.from_defaults(memory=memory, condense_question_prompt=messages, query_engine=query_engine, llm=llm)
            # self.chat_engine = CondenseQuestionChatEngine.from_defaults(query_engine=self.query_engine, llm=llm)


            self.chat_engine = ContextChatEngine.from_defaults(retriever=self.retriever, query_engine=self.query_engine, llm=llm, system_prompt=new_text_qa_tmpl_str)


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
                "You are Doner Point, an automated service to collect orders for a restaurant.\
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


    def chatAway(self, user_message):
        return self.chat_engine.chat(user_message)

    # Function to extract JSON with nested curly brackets
    def extract_json(self, text):
        bracket_count = 0
        json_start = None
        json_end = None

        for i, char in enumerate(text):
            if char == '{':
                if bracket_count == 0:
                    json_start = i
                bracket_count += 1
            elif char == '}':
                bracket_count -= 1
                if bracket_count == 0:
                    json_end = i
                    break

        if json_start is not None and json_end is not None:
            return text[json_start:json_end + 1]
        else:
            return None


    def chatLoop(self):
        # main loop
        print(f"Chatbot: Thank you for calling Doner Point! How may I help you?")
        order = RestaurantOrder()

        # this is not valid for ContextChatEngine
        self.setPromptInitiateConvo(self.query_engine)

        # init gmailSender
        gmailSender = GmailSender('algotrader506@gmail.com', 'sevm kgqb wbpo pcrr')

        # Debug
        # prompts_dict = self.query_engine.get_prompts()
        # self.display_prompt_dict(prompts_dict)


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

                response_str = str(response).replace('\n','')


                #check if the reponse is proper json
                # if yes process
                # else reply to customer

                json_string = self.extract_json(response_str)

                if json_string:
                    # Parse the JSON message
                    try:
                        order_data = json.loads(json_string)
                        print("Extracted JSON data:")
                        print(json.dumps(order_data, indent=2))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                else:
                    # print("No JSON data found in the text.")
                    raise Exception



                # order_data = json.loads(response_str)

                # One option was to maintain a separate order list by processing each response as json message
                # if 'general' in jsonResponse['request_type'].replace('_', ' '):
                #     print(f"Chatbot: {jsonResponse['response']}")
                # elif 'add' in jsonResponse['request_type'].replace('_', ' '):
                #     order.add_item(jsonResponse['menu_item_ordered'], jsonResponse['quantity'], jsonResponse['size'])
                #     print(f"Chatbot: Added {jsonResponse['menu_item_ordered']} to the order")
                # elif 'completed' in jsonResponse['request_type'].replace('_', ' '):
                #     self.setPromptOrderTaking3(self.query_engine, order.order_summary())
                # if len(order) == 0:
                #     order.append(jsonResponse)

                # Better option is to receive a json when customer order finalized....
                # Initialize RestaurantOrder object here and send to POS/email. etc.

                # Extract and process the menu items ordered
                items_ordered = order_data["menu_items_ordered"]
                total_price = 0

                print("menu_items_ordered")
                for item in items_ordered:
                    item_name = item["item"]
                    item_size = item["size"]
                    item_quantity = item["quantity"]
                    item_price = float(item["price"].replace('$', ''))  # Convert price to float
                    total_price += item_price*item_quantity
                    print(f"{item_quantity} {item_name} ({item_size}) - ${item_price:.2f}")

                # Extract total price from the JSON and compare with calculated total price
                json_total_price = float(order_data["total_price"].replace('$', ''))

                print(f"\nCalculated Total Price: ${total_price:.2f}")
                print(f"Total Price from JSON: ${json_total_price:.2f}")

                # Check if the calculated total matches the one from the JSON
                if total_price == json_total_price:
                    print("The total price matches!")
                else:
                    print("Warning: The total price does not match!")

                # Print delivery information
                if order_data["pickup_or_delivery"] != "pickup":
                    print(f"Delivery to: {order_data['address']}")
                else:
                    print("Pickup order")

                # Send an email with order content (use the same json to create order in pos)
                gmailSender.send_email("sinan.asa@me.com", "Customer Order 0001", str(json_string))

            except:
                print(f"Chatbot: {response}")

        # self.chat_store.persist(persist_path="chat_memory.json")


# main app
###################################
##########
# foodOrderChat = orderChat()
# foodOrderChat.chatLoop()



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


