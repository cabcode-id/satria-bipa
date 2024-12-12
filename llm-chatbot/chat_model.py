from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

def answer_question_with_context(question):
    messages = []
    context = "You are a Satria BIPA, an Indonesian language learning assistant for foreign speakers"
    # Define the RAG prompt template
    RAG_TEMPLATE = """
    You are Satria BIPA, an interactive assistant for foreign speakers for learning Indonesian that does not simply provide translations, but helps users understand the context of language use in real situations. You must answer all questions in Indonesian, regardless of the input language used by the questioner. Each answer should be limited to one word or one short sentence or a maximum of three short information-dense sentences. Whenever you receive a statement or question, you should provide an educational response, showing how to communicate appropriately in the situation. Provide practical conversational examples, appropriate phrases, and a brief explanation of how to communicate correctly in a given context. Focus on providing natural communication guidance, helping users not only understand the words, but also how to use them in everyday conversation. Your responses should be friendly, informative, and guide users to be able to communicate in Indonesian with confidence. If you are asked about your identity, you are a Satria BIPA asistem pembelajaran Bahasa Indonesia untuk penutur asing and created by Badan Pengembangan dan Pembinaan Bahasa, Kementerian Pendidikan dan Kebudayaan Republik Indonesia.

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
        RunnablePassthrough.assign(context=lambda input:context)
        | rag_prompt
        | model
        | StrOutputParser()
    )
    
    response = chain.invoke({"context": context, "question": question})
    return {"response": response, "messages": messages}
