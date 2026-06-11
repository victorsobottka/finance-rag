import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import json
import asyncio
from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness, ContextPrecision
from vectorstore import load_vectorstore
from rag_chain import build_rag_chain
from rag_chain import build_retriever

# =============================================================================
# RAGAS EVALUATOR — Groq via OpenAI-compatible async client
# =============================================================================
groq_client = AsyncOpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)
ragas_llm = llm_factory(
    "llama-3.1-8b-instant",
    provider="openai",
    client=groq_client
)

faithfulness  = Faithfulness(llm=ragas_llm)
ctx_precision = ContextPrecision(llm=ragas_llm)

# =============================================================================
# TEST SET — verified against ChromaDB chunks
# =============================================================================
TEST_SET = [
    {
        "question": "What was Apple's gross margin percentage in 2025?",
        "ground_truth": "Apple's product gross margin percentage was 36.8% in 2025",
        "ticker": "AAPL"
    },
    {
        "question": "What were Apple's total net sales in 2025?",
        "ground_truth": "Apple's total net sales were $416,161 million in fiscal year 2025",
        "ticker": "AAPL"
    },
    {
        "question": "What are Apple's main manufacturing risks?",
        "ground_truth": "Apple's hardware is manufactured primarily by outsourcing partners in China, India, Japan, South Korea, Taiwan, and Vietnam",
        "ticker": "AAPL"
    },
    {
        "question": "What was Apple's services revenue in 2025?",
        "ground_truth": "Apple's services revenue was $109,158 million in 2025",
        "ticker": "AAPL"
    },
    {
        "question": "What was Apple's total gross margin in 2025?",
        "ground_truth": "Apple's total gross margin was $195,201 million in 2025",
        "ticker": "AAPL"
    },
]

# =============================================================================
# EVALUATION
# =============================================================================
async def run_eval_async():
    vs = load_vectorstore()
    results = []

    for item in TEST_SET:
        ticker   = item["ticker"]
        question = item["question"]
        print(f"\nEvaluating: {question[:60]}...")

        retriever = build_retriever(vs, ticker=ticker)
        chain    = build_rag_chain(vs, retriever=retriever)
        answer   = chain.invoke(question)
        docs     = retriever.invoke(question)
        contexts = [d.page_content for d in docs]

# Replace the sample + f_score + c_score block with:
        f_score = await faithfulness.ascore(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts
        )

        c_score = await ctx_precision.ascore(
            user_input=question,
            reference=item["ground_truth"],
            retrieved_contexts=contexts
        )


        row = {
            "question":          question,
            "answer":            answer[:150],
            "ground_truth":      item["ground_truth"],
            "faithfulness":      round(f_score.value, 3),
            "context_precision": round(c_score.value, 3),
        }
        results.append(row)

        print(f"   Faithfulness:      {row['faithfulness']}")
        print(f"   Context Precision: {row['context_precision']}")

    # Save results
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to eval_results.json")

    # Summary
    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_ctx   = sum(r["context_precision"] for r in results) / len(results)

    print("\n=== AVERAGE SCORES ===")
    print(f"Faithfulness:      {round(avg_faith, 3)} (target > 0.70)")
    print(f"Context Precision: {round(avg_ctx, 3)} (target > 0.70)")

    if avg_faith >= 0.70 and avg_ctx >= 0.70:
        print("\nAll scores above target. RAG quality is good.")
    else:
        if avg_faith < 0.70:
            print("\nFaithfulness below target — tighten FINANCE_PROMPT to prevent LLM straying from context.")
        if avg_ctx < 0.70:
            print("\nContext Precision below target — increase fetch_k or adjust chunk_size in ingest.py.")

def run_eval():
    asyncio.run(run_eval_async())

if __name__ == "__main__":
    run_eval()
