#Reference : https://github.com/kzhisa/rag-fusion/blob/main/rag_fusion.py

import os
from dotenv import load_dotenv
import bs4
import argparse
import os
import sys
from operator import itemgetter

from dotenv import load_dotenv
from langchain.load import dumps, loads
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.wikipedia import WikipediaLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import TokenTextSplitter
from langchain import hub
from operator import itemgetter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_milvus import BM25BuiltInFunction, Milvus
from uuid import uuid4
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from typing import List
from langchain_core.runnables import RunnableMap
from langchain_core.prompts import PromptTemplate




from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import logging




load_dotenv()



LANGSMITH_TRACING=os.environ['LANGSMITH_TRACING']
LANGSMITH_ENDPOINT=os.environ['LANGSMITH_ENDPOINT']
LANGSMITH_API_KEY=os.environ['LANGSMITH_API_KEY']
LANGSMITH_PROJECT=os.environ['LANGSMITH_PROJECT']
OPENAI_API_KEY=os.environ['OPENAI_API_KEY']
URI=os.environ['URI']




def load_and_split_document(DOCUMENT_URL):
    # # Load Documents
    loader = WebBaseLoader(
        web_paths=(DOCUMENT_URL,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # for TEST
    print("Original document: ", len(splits), " docs")

    return splits


def create_retriever(DOCUMENT_URL="https://lilianweng.github.io/posts/2023-06-23-agent/"):
    splits = load_and_split_document(DOCUMENT_URL)
    embeddings = OpenAIEmbeddings()

    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": URI,"db_name": "milvus_demo"},
        collection_name="simple_rag",
        index_params={"index_type": "FLAT", "metric_type": "L2"},
    )

    uuids = [str(uuid4()) for _ in range(len(splits))]
    vector_store.add_documents(documents=splits, ids=uuids)

    # retriever
    retriever = vector_store.as_retriever()

    return retriever


def retriever(query):
    """RRF retriever

    Args:
        query (str): Query string

    Returns:
        list[Document]: retrieved documents
    """

    # Retriever
    retriever = create_retriever() #Vector DB

    # RRF chain
    chain = (
        {"query": itemgetter("query")}
        | retriever.map()
    )

    # invoke
    result = chain.invoke({"query": query})
    print("result of rrf_retriever",retriever)

    return result

def generator(question_text: str, retriever: BaseRetriever):
    """
    Retrieves relevant documents using `retriever` and generates an AI response.
    """

    prompt = PromptTemplate.from_template(
        """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # rag_chain = (
    #     RunnableMap({
    #         "context": retriever | format_docs,         # Now expects just a string
    #         "question": RunnablePassthrough()
    #     })
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    rag_chain = (
            RunnableMap({
                "context": itemgetter("question") | retriever | format_docs,
                "question": itemgetter("question")
            })
            | prompt
            | llm
            | StrOutputParser()
    )

    result = rag_chain.invoke(question_text)  # Pass raw string

    return result



if __name__ == '__main__':

    questions ={"question": "What are the main components of an LLM-powered autonomous agent system?"}
    #questions = "Can you explain me in simply terms what is LLM and can it be autonomous agent?"
    #questions = "Who is Nebojsa Dimic? Say that you dont know if you have no info about him"

    retriever_instance = create_retriever()

    print(generator(questions, retriever_instance))


