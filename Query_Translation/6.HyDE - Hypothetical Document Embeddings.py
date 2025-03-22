#Reference : https://github.com/kzhisa/rag-fusion/blob/main/rag_fusion.py

import bs4
import os

from dotenv import load_dotenv
from langchain_core.retrievers import BaseRetriever
from operator import itemgetter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_milvus import Milvus
from uuid import uuid4
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
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

def retriever(query):
    """RRF retriever

    Args:
        query (str): Query string

    Returns:
        list[Document]: retrieved documents
    """

    # Retriever
    retriever = create_retriever()

    # RRF chain
    chain = (
        {"query": itemgetter("query")}
        | RunnableLambda(take_step_back)
        | retriever.map()
    )

    # invoke
    result = chain.invoke({"query": query})
    print("result of rrf_retriever",retriever)

    return result

def make_hypothetical_answer(question):

    #HyDE document genration
    template = """Please write a scientific paper passage to answer the question
    Question: {question}
    Passage:"""

    prompt_hyde = PromptTemplate(
        input_variables=["question"],
        template=template,
    )
    #prompt_hyde = ChatPromptTemplate.from_template(template)


    llm = ChatOpenAI(temperature=0)
    generate_docs_for_retrieval = (
        prompt_hyde | llm | StrOutputParser()
    )

    # Run
    answers = generate_docs_for_retrieval.invoke({"question":question})
    return answers

def make_hypothetical_embedding_vector(hypo_answer):
    embedding_model = OpenAIEmbeddings()
    embedding = embedding_model.embed_query(hypo_answer)
    return embedding

def final_answer_generate(query):
    question = query["query"]

    # Step 1: Generate hypothetical document
    hypo_passage = make_hypothetical_answer(question)

    # Step 2: Convert it to an embedding
    hypo_embedding = make_hypothetical_embedding_vector(hypo_passage)

    # Step 3: Get retriever and underlying vector store
    retriever = create_retriever()  # already builds and loads Milvus with document splits
    vector_store = retriever.vectorstore  # <- get access to raw vector store from retriever

    # Step 4: Retrieve top-k most similar real documents using HyDE embedding
    results = vector_store.similarity_search_by_vector(hypo_embedding, k=5)

    # Print or use the results
    print("Top 5 retrieved real docs based on hypothetical embedding:\n")
    for i, doc in enumerate(results):
        print(f"Document {i + 1}:\n{doc.page_content}\n")

    return results

if __name__ == '__main__':
    # Step 1: Define the original question
    original_query = {"query": "What is task decomposition for LLM agents?"}

    print(final_answer_generate(original_query))
