from flask import Flask, request, render_template, session
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma  # ✅ Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader  # ✅ Updated import
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
import os
import fitz  # PyMuPDF for PDF processing
from langchain.schema import Document

app = Flask(__name__)
app.secret_key = os.urandom(24)

# ✅ Manually Set Google API Key
google_api_key = "API KEY"  # 🔥 Replace with your API key

# ✅ Load PDF Function
def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = "\n".join([page.get_text() for page in doc])
    return text

# ✅ Initialize QA System
def initialize_qa_system():
    pdf_text = load_pdf("budget_speech.pdf")
    docs = [Document(page_content=pdf_text)]  # Convert to LangChain Document format

    # ✅ Text Splitting for Better Retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)

    # ✅ Google AI Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model='models/embedding-001',
        google_api_key=google_api_key,
        task_type="retrieval_query"
    )

    # ✅ Create Vector Database
    vectordb = Chroma.from_documents(documents=texts, embedding=embeddings)

    # ✅ Define the Prompt Template
    prompt_template = """
    Context: \n {context}
    Question: \n {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    # ✅ Safety Settings
    safety_settings = {
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    }

    # ✅ Set Up Google Gemini Chat Model
    chat_model = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=google_api_key,
        temperature=0.7,
        safety_settings=safety_settings
    )

    # ✅ Multi-Query Retriever for Better Answering
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
        llm=chat_model
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        retriever=retriever_from_llm,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

# ✅ Initialize QA System
qa_chain = initialize_qa_system()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_question = request.form["question"]
        response = qa_chain.invoke({"query": user_question})  # ✅ Fixed Query Structure
        bot_response = response.get('result', 'Sorry, I could not generate a response.')

        return render_template("index.html", user_question=user_question, bot_response=bot_response)

    return render_template("index.html", user_question=None, bot_response=None)

# ✅ Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
