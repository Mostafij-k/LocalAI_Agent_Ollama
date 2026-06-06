from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriver

# Specify model Name
model=OllamaLLM(model='llama3.2')

# Create template
template= """
You are an expert in answering question about a pizza retuarant
Here, some relevant reviews:{reviews}
Here, is the questions to answer:{question}
"""
# Create prompt template
prompt=ChatPromptTemplate.from_template(template)

# Create a chain
chain=prompt | model

while True:
    print('\n\n--------------------')
    question=input('Ask your question(q to quit): ')
    print('\n\n')
    if question=='q':
        break  
    reviews=retriver.invoke(question) 
    result=chain.invoke({'reviews':reviews,'question': question})
    print(result)