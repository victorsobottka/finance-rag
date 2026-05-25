from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
from vectorstore import load_vectorstore
from rag_chain import build_rag_chain, build_retriever

# Test questions with known answers (ground truth)
TEST_SET = [
    {
        "question": "What was Apple's gross margin percentage in 2025?",
        "ground_truth": "Apple's gross margin percentage was 36.8% in 2025"
        "ticker" : "APPL"
    },
    {
        "question": "What were Apple's total net sales in 2025?",
        "ground_truth": "Apple's total net sales were $416,161 million in 2025"
        "ticker" : "APPL"
    },
    {
        "question": "What are Apple's main risk factors?",
        "ground_truth": "Key risks include supply chain concentration, regulatory scrutiny, and macroeconomic conditions"
        "ticker" : "AAPL"
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

def run_eval():
    vs = load_vectorstore()
    chain = build_rag_chain(vs)
    retriever = build_retriever(vs)

    results = []
    for item in TEST_SET:
        question = item["question"]
        answer = chain.invoke(question)
        docs = retriever.invoke(question)
        contexts = [d.page_content for d in docs]

        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"]
        })

    dataset = Dataset.from_list(results)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision]
    )
    print(scores)
    scores.to_pandas().to_json("eval_results.json")

if __name__ == "__main__":
    run_eval()

### Change chunk size → run eval → see if faithfulness improved. This is how production AI teams work.
