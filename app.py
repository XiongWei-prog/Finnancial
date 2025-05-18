
# 3.0 初始化 Chroma 客户端

from langchain_community.embeddings import QianfanEmbeddingsEndpoint

from langchain_chroma import Chroma



embedding = QianfanEmbeddingsEndpoint(model='bge-large-zh')

persist_directory = 'docs/vector_db/chroma/600519-2022'



vectordb = Chroma(

    embedding_function=embedding,

    persist_directory=persist_directory

)


# 3.1 检索增强生成

from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate

from langchain_community.llms import QianfanLLMEndpoint



QA_CHAIN_PROMPT = PromptTemplate.from_template(

    """

    你是上市公司的董事会秘书，你需要针对投资者的问题作出准确、可靠和完整的回答。   \

    请使用以下的 context 来回答最后的问题。如果你不知道答案，就直接说你不知道，不要试图编造答案。\

    回答字数最多控制在500字左右。总是在回答的最后说“谢谢您的提问！”。

    

    <context>

    {context}

    </context>

    

    问题：{question}

    回答：

    """

)



llm = QianfanLLMEndpoint(model="ERNIE-Bot", temperature=0.1)



retriever = vectordb.as_retriever(

    search_type='similarity',

    search_kwargs={'k': 4} 

)



qa_chain = RetrievalQA.from_chain_type(

    llm, 

    chain_type='stuff',

    retriever=vectordb.as_retriever(), 

    return_source_documents=True, 

    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}

)


import gradio as gr



def chat(message, history):

    response = qa_chain.invoke({"query": message})

    return response['result']



demo = gr.ChatInterface(

    fn=chat,

    examples=['贵州茅台在环境和社会责任方面取得哪些成效？',

              '贵公司面临哪些主要风险？',

              '针对公司主要面临的风险，公司会采取哪些应对措施？',

              '贵州茅台在2023年有哪些经营规划？']

).queue(default_concurrency_limit=2)



demo.launch(

    auth=['caiwu', '2024']

)
