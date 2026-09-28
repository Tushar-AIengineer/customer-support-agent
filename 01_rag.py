# -----------------import libraries-----------------

#----------------------For RAG------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma

#---------------------For connect mysql--------------------
import mysql.connector

#---------------------For hiding the private info----------------
import os
from dotenv import load_dotenv

#---------------------For creat tools----------------------
from langchain.tools import tool

#----------------------For store conversation history-----------------
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

#--------------------For creating agent-------------------
from langchain.agents import create_agent


#-----------------------Actual code starts...........---------------------


#------------------hide api key and mysql password-----------------

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
password = os.getenv("password")


#------------------load the pdf file for RAG--------------------

loader = PyPDFLoader(r"C:\Users\Tushar Tomar\OneDrive\Desktop\python program\31_new\Return_Refund_Policy.pdf")
document = loader.load()


#------------------split the data and convert into chunks------------------

split = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20
)
chunks = split.split_documents(document)


#-------------------Embedding the chunks-------------------

embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=API_KEY
)


#----------------Store into vector store with the help of chrome--------------------

vector_store = Chroma.from_documents(
    documents = chunks,
    embedding = embedding
)


#---------------------Connecting  the mysql database------------------- 

try:
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password=password,
        database="customer_support"
)

    
#------------------Check the connection------------------------

    if db.is_connected():
        print("MySQL Connected Successfully")

except Exception as e:
     print(" database connection failed")

cursor = db.cursor()


#-------------------------Tool 1-> RAG based tool------------------------

@tool
def get_policy(question):
          "if customer wanted to know the policy of return of product and days and other policy related terms because they ara a rag based tool"
          try:
             retriever = vector_store.as_retriever()
             docs = retriever.invoke(question)
             return docs
          except Exception as e:
              return f"database error {e}"


#------------------------Tool 2-> database tool for customer info-------------------

@tool
def get_customer(customer_id):
     """if customer_id can provide by user and want to know our email,name,phone"""
     """Get customer information such as name, email, and phone number using customer ID."""
     try:
          cursor.execute(
          "SELECT * FROM customer WHERE customer_id = %s",
           (customer_id,)
           )
          data = cursor.fetchone()
          return data
     except Exception as e:
         return f"database error {e}"
     

#------------------------Tool 3-> database tool for customer order-------------------

@tool
def get_order(customer_id):
     """if customer_id can provide by user and want to know our order status """
     """Get the customer's order information and order status using customer ID."""
     try:
          cursor.execute(
                    "SELECT * FROM orders WHERE customer_id = %s",
                    (customer_id,)
               )
          data = cursor.fetchone()
          return data 
     
     except Exception as e:
          return f"database error {e}"
     

#------------------------Tool 4-> database tool for customer ticket info-------------------

@tool
def get_ticket(customer_id):
     """Get the customer's support ticket information using customer ID."""
     try:
          cursor.execute(
                    "SELECT * FROM tickets WHERE customer_id = %s",
                    (customer_id,)
               )
          data = cursor.fetchone()
          return data 
     except Exception as e:
          return f"database error {e}"


#----------------------use large languaga model(llm)----------------------------

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash",
    google_api_key = API_KEY
    )


#---------------------creat our agent---------------------------
 
agent = create_agent(
     model="google_genai:gemini-3.6-flash",
     tools=[get_customer,get_order,get_ticket,get_policy],
     system_prompt="""
      You are a customer support agent.

      Always answer briefly and directly.
      For simple questions, answer in 1-3 short sentences or bullet points.
      Do not give unnecessary explanations.

      Only answer questions related to the company's
      products, orders, returns, refunds, payments, policies,
      tickets, and customer support.

      If the question is unrelated to the company or customer support,
      politely refuse to answer and say that you only handle
      customer support related queries.
          """
            )


#------------------------------use conversation history-----------------------

history = InMemoryChatMessageHistory()


#---------------------------chatbot to store conversation history-------------------

chatbot = RunnableWithMessageHistory(
    agent,
    lambda session_id: history,
    input_messages_key="messages"
)


#-------------------------user input------------------------------

while True:
     
     question = input("Enter Your Question : ")
     

     if question.lower() == "exit":
        break

 
     try:
         result = chatbot.invoke(
             {
                 "messages":[
                     {
                         "role":"user",
                         "content":question
                     }
                 ]
             },
             config={
                 "configurable":{
                     "session_id":"123"
                 }
             }
         )

     except Exception as e:
          print("something want wrong . please try again")
    

#-------------------------final response------------------------
    
     print(result["messages"][-1].content[0]["text"])
     print()