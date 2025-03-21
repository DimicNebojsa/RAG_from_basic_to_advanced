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
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_milvus import  Milvus
from uuid import uuid4
from langchain_openai import ChatOpenAI
from typing import List
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

load_dotenv()

#TOP_K = 2
MAX_DOCS_FOR_CONTEXT = 3

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
        collection_name="splits",
        index_params={"index_type": "FLAT", "metric_type": "L2"},
    )

    uuids = [str(uuid4()) for _ in range(len(splits))]
    vector_store.add_documents(documents=splits, ids=uuids)

    # retriever
    retriever = vector_store.as_retriever()

    return retriever


def query_generator(original_query):
    query = original_query.get("query")

    print("Original Query:", query)

    class LineListOutputParser(BaseOutputParser[List[str]]):
        """Output parser for a list of lines."""

        def parse(self, text: str) -> List[str]:
            lines = text.strip().split("\n")
            return list(filter(None, lines))  # Remove empty lines

    output_parser = LineListOutputParser()

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    llm = ChatOpenAI(temperature=0)

    # Chain
    llm_chain = QUERY_PROMPT | llm | output_parser

    queries = llm_chain.invoke({"question": query})

    # Add original query
    queries.insert(0, "0. " + query)

    # Print for debugging
    print('Generated queries:\n', '\n'.join(queries))

    return queries

def reciprocal_rank_fusion(results: list[list], k=60):
    """Rerank docs (Reciprocal Rank Fusion)

    Args:
        results (list[list]): retrieved documents
        k (int, optional): parameter k for RRF. Defaults to 60.

    Returns:
        ranked_results: list of documents reranked by RRF
    """

    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # for TEST (print reranked documentsand scores)
    print("Reranked documents: ", len(reranked_results))
    for doc in reranked_results:
        print('---')
        print('Docs: ', ' '.join(doc[0].page_content[:100].split()))
        print('RRF score: ', doc[1])

    # return only documents
    return [x[0] for x in reranked_results[:MAX_DOCS_FOR_CONTEXT]]

def rrf_retriever(query):
    """RRF retriever using external query generation.

    Args:
        query (str): Query string

    Returns:
        list[Document]: retrieved documents
    """

    # Step 1: Generate list of queries outside the chain
    generated_queries = query_generator({"query": query})

    # Step 2: Create the retriever
    retriever = create_retriever()

    # Step 3: Retrieve documents for each query
    #docs_per_query = [retriever.get_relevant_documents(q) for q in generated_queries]
    docs_per_query = []
    for q in generated_queries:
        document = retriever.get_relevant_documents(q)
        docs_per_query.append(document)

    # Step 4: Apply reciprocal rank fusion to rerank
    fused_docs = reciprocal_rank_fusion(docs_per_query)

    print("------------------------------------")
    print("result of rrf_retriever", fused_docs)
    return fused_docs

def result_generator(query: str):
    """
    Retrieves relevant documents using `retriever` and generates an AI response.
    """

    docs = rrf_retriever(query)

    # RAG
    template = """Answer the following question based on this context:

        {context}

        Question: {question}
        """

    prompt = ChatPromptTemplate.from_template(template)

    # Initialize OpenAI model
    llm = ChatOpenAI(temperature=0)
    context_text = "\n\n".join(doc.page_content for doc in docs)

    prompt_input = {
        "context": context_text,
        "question": query
    }

    chain = (prompt
            | llm
            | StrOutputParser()
    )

    result = chain.invoke(prompt_input)

    return result

if __name__ == '__main__':

    original_query = {"query": "What are the main components of an LLM-powered autonomous agent system?"}

    # Generate response
    result = result_generator(original_query["query"])

    # Print the result
    print('---\nAnswer:')
    print(result)


