import pandas as pd
from langchain_community.vectorstores import Chroma
from azuremodels import llm, embeddings,serachRe
from pydantic import BaseModel
from typing import List
import os
from langchain_classic.schema import Document
from prompts import *

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

#Question Generation
class GeneratedQuestions(BaseModel):
    questions: List[str]


def analysisMarket(USER_TOPIC):
    finance_prompt  = FINANCE_RESEARCH_AGENT.format(USER_TOPIC=USER_TOPIC)

    structured_llm = llm.with_structured_output(GeneratedQuestions)

    questions = structured_llm.invoke(finance_prompt )
    questions = questions.questions


    #News search
    newslist = []
    for query in questions:
        res = serachRe.invoke(query)
        newslist.append(res[0].page_content)
        newslist.append(res[1].page_content)


    #Summary News
    trend_analyzer_prompt = TRENT_ANALYZER_AGENT.format(newslist=newslist)
    summary = llm.invoke(trend_analyzer_prompt)
    summary  = summary.content



    PERSIST_DIR = "marketCondtions"  # folder name on disk

    vectorstore = Chroma(
    embedding_function=embeddings,
    collection_name="marketCondtions",
    persist_directory="marketCondtions"
    )
        
    collection = vectorstore._collection
    vec = vectorstore.as_retriever(search_kwargs={"k": 2})

    all_docs = collection.get(include=['documents', 'metadatas'])

    documents = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(all_docs['documents'], all_docs['metadatas'])
    ]
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 2
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vec, bm25_retriever],
        weights=[0.7, 0.3]  # Equal weights for vector and BM25
    )

    # Use the hybrid retriever
    # results = ensemble_retriever.get_relevant_documents("india china war")

    # db = vectorstore.as_retriever()
    print(summary)
    res =  ensemble_retriever.get_relevant_documents(summary)
    print(res)
    return res[0].metadata



#generating updated general strategies
def apply_instrument_deltas(strategy_csv, output_csv, deltas: dict):
    """
    Applies instrument-level deltas directly and normalizes totals back to 100%.
    
    Assumptions:
    - strategy.csv contains all instrument columns as numeric weights summing to 100.
    - 'Client Type' remains untouched.
    - deltas dict keys must match exact CSV column names.
    """
    df = pd.read_csv(strategy_csv)

    # instrument columns = all except Client Type
    instrument_cols = [c for c in df.columns if c != "Client Type"]

    # Validate column presence
    missing = [k for k in deltas if k not in instrument_cols]
    if missing:
        raise ValueError(f"Delta keys not found in strategy columns: {missing}")

    # ---- Apply deltas ----
    for inst_col in instrument_cols:
        if inst_col in deltas:
            base = df[inst_col].astype(float)
            pct = deltas[inst_col]
            df[f"{inst_col}_adj"] = base + (base * pct/100.0)
        else:
            df[f"{inst_col}_adj"] = df[inst_col].astype(float)

    adj_cols = [f"{c}_adj" for c in instrument_cols]

    # ---- Normalize so total remains 100 ----
    df["row_total"] = df[adj_cols].sum(axis=1)
    df["row_total"] = df["row_total"].replace(0, 1.0)

    for col in instrument_cols:
        df[f"{col}_final"] = df[f"{col}_adj"] / df["row_total"] * 100.0

    # ---- Keep CLEAN output ----
    final_cols = ["Client Type"] + [f"{c}_final" for c in instrument_cols]

    df[final_cols].to_csv(output_csv, index=False)

    return df[final_cols]



def reason(USER_TOPIC):

    if USER_TOPIC==None:
        USER_TOPIC = "Current Market Trend"

    deltas = analysisMarket(USER_TOPIC=USER_TOPIC)
    
    updated = apply_instrument_deltas(
        "generalStrategyStandard.csv",
        "strategy_updated.csv",
        deltas
    )

    return  {"status": "success", "csv_file": "strategy_updated.csv"}
    

