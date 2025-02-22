from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# Cấu hình
groq_api_key = "YOUR_API_KEY"
vector_db_path = "vectorstores/db_faiss"

# Load Language Model qua Groq API
def load_llm():
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="mixtral-8x7b-32768",
        temperature=0.01,
        max_tokens=1024
    )
    return llm

# Đọc từ VectorDB
def read_vectors_db():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        vector_db_path, 
        embedding_model, 
        allow_dangerous_deserialization=True
    )
    return db

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    # Khởi tạo các thành phần
    db = read_vectors_db()
    llm = load_llm()
    
    # Tạo template và chain mới với cú pháp LCEL
    template = """Sử dụng thông tin sau đây để trả lời câu hỏi. 
    Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời.
    
    Context: {context}
    Question: {question}
    
    Answer:"""
    
    prompt = PromptTemplate.from_template(template)
    
    # Tạo chain sử dụng LCEL (LangChain Expression Language)
    chain = (
        {"context": db.as_retriever().invoke, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Chạy thử nghiệm
    question = "View Class là gì?"
    response = chain.invoke(question)
    print('======================')
    print(response)

if __name__ == "__main__":
    main()
