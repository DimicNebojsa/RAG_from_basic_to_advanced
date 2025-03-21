#Reference : https://github.com/kzhisa/rag-fusion/blob/main/rag_fusion.py

import os
from dotenv import load_dotenv
import bs4
import argparse
import os
import sys
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap
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

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
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
        collection_name="decomposition",
        index_params={"index_type": "FLAT", "metric_type": "L2"},
    )

    uuids = [str(uuid4()) for _ in range(len(splits))]
    vector_store.add_documents(documents=splits, ids=uuids)

    # retriever
    retriever = vector_store.as_retriever()

    print("stored doucment into Vector DB Done",retriever)

    return retriever


def query_generator(original_query):


    #query = original_query.get("query")
    #print("Original Query:", query)

    class LineListOutputParser(BaseOutputParser[List[str]]):
        """Output parser for a list of lines."""

        def parse(self, text: str) -> List[str]:
            lines = text.strip().split("\n")
            return list(filter(None, lines))  # Remove empty lines

    output_parser = LineListOutputParser()

    QUERY_PROMPT = PromptTemplate.from_template(
        """You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. Original question: {query}
        """
    )

    llm = ChatOpenAI(temperature=0)

    # Chain
    #llm_chain = RunnableMap(QUERY_PROMPT | llm | output_parser)
    llm_chain = (
            RunnableMap({
                "query": itemgetter("query")
            })
            | QUERY_PROMPT
            | llm
            | output_parser
    )


    #queries = llm_chain.invoke({"question": query})
    queries = llm_chain.invoke(original_query)

    # Add original query
    queries.insert(0, "0. " + original_query['query'])

    # Print for debugging
    print('Generated queries:\n', '\n'.join(queries))

    return queries



def retriever(query: dict):
    # Step 1: Call the vector DB retriever
    vector_db_retriever = create_retriever()

    # Step 2: Generate queries once
    #queries = query_generator({"query": query})
    queries = query_generator(query)

    # Step 3: Retrieve documents for each query
    all_docs = []
    for q in queries:
        docs = vector_db_retriever.get_relevant_documents(q)
        all_docs.extend(docs)

    print(f"Retrieved {len(all_docs)} documents in total before deduplication")

    # Step 4: Deduplicate using set of string dumps
    unique_docs = list({dumps(doc): doc for doc in all_docs}.values())

    print(f"Returning {len(unique_docs)} unique documents after deduplication")
    return unique_docs



def final_answer_generate(question: dict):
    """
    Retrieves relevant documents using `retriever` and generates an AI response.
    """
    docs = retriever(question)

    # RAG
    template = """Answer the following question based on this context:

    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(temperature=0)
    context_text = "\n\n".join(doc.page_content for doc in docs)

    chain = (
            RunnableLambda(lambda x: {"context": context_text, "question": question["query"]})
            # RunnableMap({
            #     "question": dict(question),
            #     "context": lambda x: context_text
            # })
            | prompt
            | llm
            | StrOutputParser()
    )


    result = chain.invoke({})

    return result




if __name__ == '__main__':
    original_query = {"query": "What are the main components of an LLM-powered autonomous agent system?"}
    answer = final_answer_generate(original_query)
    print("\n\n=== FINAL ANSWER ===\n")
    print(answer)
