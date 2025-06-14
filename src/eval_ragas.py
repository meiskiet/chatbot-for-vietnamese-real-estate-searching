# eval_ragas.py  – Ollama + RAGAS evaluation (no function calls)

import json, tqdm
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)

from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from local_ollama import get_retriever           # your retriever builder

# -------------------------------------------------
COLLECTION   = "data1k"          # Milvus collection name
GOLD_FILE    = "data/gold_set.json"   # queries ↔ UUIDs (not used by RAGAS)
TOP_K        = 5                      # # contexts to feed RAGAS
OLLAMA_MODEL = "deepseek-r1:7b"
# -------------------------------------------------

# 1. Build retriever and LLM
retriever = get_retriever(COLLECTION)
llm       = ChatOllama(model=OLLAMA_MODEL, temperature=0)

# 2. Simple RAG chain (stuffing contexts + prompt)
qa_prompt = PromptTemplate(
    template=(
        "Trả lời ngắn gọn dựa trên thông tin dưới đây.\n\n"
        "{context}\n\n"
        "Câu hỏi: {question}\nTrả lời:"
    ),
    input_variables=["question", "context"],
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": qa_prompt},
    return_source_documents=True,        # <-- add this flag
)

# 3. Load queries
queries = list(json.load(open(GOLD_FILE)))
records = []

for q in tqdm.tqdm(queries, desc="Evaluating"):
    res = rag_chain.invoke({"query": q, "k": TOP_K})
    answer     = res["result"]
    contexts   = res["source_documents"]   # list[Document]

    records.append(
        {
            "question": q,
            "answer":   answer,
            "contexts": [d.page_content for d in contexts],
        }
    )

# 4. Wrap as HF Dataset for RAGAS
dataset = Dataset.from_list(records)

# 5. Run RAGAS metrics
report = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy],
)

print(report)

