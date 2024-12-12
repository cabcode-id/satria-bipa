from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def answer_question_with_context(conversation_history, question):
    persist_directory = "db32"
    local_embeddings = OllamaEmbeddings(model="arcmr/satria-bipa:latest")

    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=local_embeddings)

    docs = vectorstore.similarity_search(question)
    formatted_docs = format_docs(docs)

    # Combine conversation history with retrieved documents
    full_context = conversation_history + "\n\n" + formatted_docs

    # Define the RAG prompt template
    RAG_TEMPLATE = """
    You are Satria BIPA, an Indonesian language expert assistant for foreign speakers who is tasked with providing linguistic information briefly, clearly, and accurately. You must answer all questions in Bahasa Indonesia, regardless of the input language used by the questioner. Each answer should be limited to one word or one short sentence or a maximum of three short, information-dense sentences. Provide one best and most appropriate answer to each question, without offering alternatives or other choices. Focus on providing answers that are to the point, easy to understand, and informative. Prioritize clarity and simplicity, without compromising the meaning and substance of the information conveyed. Your responses should be educational, friendly and professional, with the aim of helping foreign speakers understand Bahasa Indonesia better. If you are asked about your identity, you are a Satria BIPA asistem pembelajaran Bahasa Indonesia untuk penutur bahasa asing and created by Badan Pengembangan dan Pembinaan Bahasa, Kementerian Pendidikan dan Kebudayaan Republik Indonesia.
    
    <context>
    {context}
    </context>

    Answer the following question:

    {question}"""

    rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    model = ChatOllama(
        model="arcmr/satria-bipa:latest",
        repetition_penalty=1.05,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        transformers_version=4.46,
        system="You are a helpful assistant for foreign speaker to learn Indonesian conversation"
    )

    chain = (
        RunnablePassthrough.assign(context=lambda input: input["context"])
        | rag_prompt
        | model
        | StrOutputParser()
    )

    response = chain.invoke({"context": full_context, "question": question, "history": conversation_history})
    return {"response": response}
