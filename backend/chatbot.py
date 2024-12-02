from langchain import PromptTemplate
from langchain.llms import OpenAI

def generate_answer(query, retrieved_chunks):
    prompt = PromptTemplate(
        template="Based on the following context:\n\n{context}\n\nAnswer the question:\n{question}",
        input_variables=["context", "question"]
    )
    context = " ".join([chunk["metadata"]["text"] for chunk in retrieved_chunks])
    formatted_prompt = prompt.format(context=context, question=query)
    llm = OpenAI(model_name="gpt-j")
    return llm(formatted_prompt)
