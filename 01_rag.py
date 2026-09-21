from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma

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
    google_api_key=api_key
)

vector_store = Chroma.from_documents(
    documents = chunks,
    embedding = embedding
)

retriever = vector_store.as_retriever()

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash",
    google_api_key = api_key
)

while True:
    question = input("enter your question : ")

    if question.lower() == "exit":
            break

    docs = retriever.invoke(question)

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