#Reference : https://github.com/kzhisa/rag-fusion/blob/main/rag_fusion.py

import bs4
import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_milvus import Milvus
from uuid import uuid4
from langchain_openai import ChatOpenAI
from typing import List
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate

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
        template="""You are a helpful assistant that generates multiple sub-questions related to an input question. \n
        The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
        Generate multiple search queries related to: {question} \n
        Output (3 queries):""",
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

def retriever(query):
    """Retriever using external query generation.

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
    docs_per_query = []
    for q in generated_queries:
        document = retriever.get_relevant_documents(q)
        docs_per_query.append(document)

    #print("------------------------------------")
    #print("result of retriever", docs_per_query)
    return generated_queries, docs_per_query

def final_answer_generate(original_query):

    # retrieve documents after query decomposition
    questions, retrieved_docs_per_query = retriever(original_query["query"])

    # run retrieved documents through first LLM
    rag_prompt_template = """
    You are an expert assistant. Use the context below to answer the question as accurately and concisely as possible.

    If the answer is not in the context, say "I don't know".

    Question: {question}

    Context:
    {context}
    """

    prompt_rag = PromptTemplate(
        input_variables=["context", "question"],
        template=rag_prompt_template,
    )

    # Create LLM instance
    llm = ChatOpenAI(temperature=0)

    # Prepare the RAG chain
    rag_chain = prompt_rag | llm | StrOutputParser()

    # Step 3: Run RAG for each sub-question
    def answer_individual_subquestions(questions, retrieved_docs_per_query):
        answers = []

        for q, docs in zip(questions, retrieved_docs_per_query):
            # Combine all document texts into one context string
            context = "\n".join(doc.page_content for doc in docs)

            # Invoke the chain
            answer = rag_chain.invoke({"question": q, "context": context})
            answers.append(answer)

        return answers

    individual_answers = answer_individual_subquestions(questions, retrieved_docs_per_query)

    def format_qa_pairs(questions, answers):
        return "\n\n".join(
            f"Question: {q}\nAnswer: {a}" for q, a in zip(questions, answers)
        )

    qa_pairs_text = format_qa_pairs(questions, individual_answers)

    # Final Synthesis prompt

    final_prompt_template = """
    Here is a set of Q&A pairs:

    {qa_pairs}

    Use them to answer the main question:

    {main_question}
    """

    synthesis_prompt = PromptTemplate(
        input_variables=["qa_pairs", "main_question"],
        template=final_prompt_template
    )

    # Run final LLM to generate final answer

    final_chain = synthesis_prompt | llm | StrOutputParser()

    final_answer = final_chain.invoke({
        "qa_pairs": qa_pairs_text,
        "main_question": original_query["query"]
    })

    print('Returning Final Answer.....')

    return final_answer

if __name__ == '__main__':
    # Step 1: Define the original question
    original_query = {"query": "What are the main components of an LLM-powered autonomous agent system?"}

    # Step 2: Invoke chain for "Answer Individually" RAG approach
    print(final_answer_generate(original_query))
