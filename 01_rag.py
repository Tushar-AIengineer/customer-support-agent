from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma

import mysql.connector

import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

loader = PyPDFLoader(r"C:\Users\Tushar Tomar\OneDrive\Desktop\python program\31_new\Return_Refund_Policy.pdf")
document = loader.load()

split = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20
)

chunks = split.split_documents(document)

embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=API_KEY
)

vector_store = Chroma.from_documents(
    documents = chunks,
    embedding = embedding
)

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash",
    google_api_key = API_KEY
    )


def get_policy(question):
     
    retriever = vector_store.as_retriever()
    docs = retriever.invoke(question)
    return docs


while True:
    question = input("enter your question : ")
          
    if question.lower() == "exit":
            break

    docs = get_policy(question)

    context = ""

    for doc in docs:
         context += doc.page_content + "\n"

    prompt = f"answer the question only using the context below    context: {context}  question:{question}"
    
    responce = llm.invoke(prompt)

    print("\nAI :")
    # print(responce.content)
    print(responce.content[0]["text"])

    print("\n")

# "Damaged product ka return kitne time mein kar sakta hoon?"
# "Mujhe wrong product mila hai, kya refund mil sakta hai?"
# "Return period kitna hai?"
# "Refund process hone mein kitna time lagta hai?"
# "Kya damaged product ka return shipping charge mujhe dena padega?"
# "Kya customer ke dwara damaged product return ho sakta hai?"
# "Order ship hone ke baad cancel kar sakte hain?"

password = os.getenv("password")

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password=password,
    database="customer_support"
)

if db.is_connected():
    print("MySQL Connected Successfully")


cursor = db.cursor()
# cursor.execute("SELECT * FROM customer")
# data = cursor.fetchall()
# print(data)



# cursor.execute("SELECT * FROM customer WHERE customer_id = 'C005'")
# data = cursor.fetchone()
# print(data)



# customer_id = input("Enter Customer ID: ")
# cursor.execute(
#     "SELECT * FROM customer WHERE customer_id = %s",
#     (customer_id,)
# )
# data = cursor.fetchone()
# print(data)




from langchain.tools import tool

customer_id = input("enter customer id : ")

@tool
def get_customer(customer_id):
     """if customer_id can provide by user and want to know our email,name,phone"""
     """Get customer information such as name, email, and phone number using customer ID."""
     cursor.execute(
          "SELECT * FROM customer WHERE customer_id = %s",
          (customer_id,)
     )

     data = cursor.fetchone()
     return data

data = get_customer.invoke({"customer_id":customer_id})
print(data)


@tool
def get_order(customer_id):
     """if customer_id can provide by user and want to know our order status """
     """Get the customer's order information and order status using customer ID."""
     cursor.execute(
          "SELECT * FROM orders WHERE customer_id = %s",
          (customer_id,)
     )
     data = cursor.fetchone()
     return data 

data = get_order.invoke({"customer_id":customer_id})
print(data)


@tool
def get_ticket(customer_id):
     """Get the customer's support ticket information using customer ID."""
     cursor.execute(
          "SELECT * FROM tickets WHERE customer_id = %s",
          (customer_id,)
     )
     data = cursor.fetchone()
     return(data)

data = get_ticket.invoke({"customer_id":customer_id})
print(data)
   
llm_with_tool = llm.bind_tools([get_customer,get_order,get_ticket])


from langchain.agents import create_agent

agent = create_agent(
     model="google_genai:gemini-3.6-flash",
     tools=[get_customer,get_order,get_ticket]

)

while True:
     
     question = input("Enter Your Question : ")


     if question.lower() == "exit":
        break

     result = agent.invoke(
         {
              "messages":[
                   {
                        "role":"user",
                        "content":question
                   }
              ]
         }
        )
     print(result["messages"][-1].content)



