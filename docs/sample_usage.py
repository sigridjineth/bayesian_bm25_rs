import bb25 as bb


def main() -> None:
    corpus = bb.build_default_corpus()
    docs = corpus.documents()
    queries = bb.build_default_queries()

    bm25 = bb.BM25Scorer(corpus, 1.2, 0.75)
    bayes = bb.BayesianBM25Scorer(bm25, 1.0, 0.5)
    vector = bb.VectorScorer()
    hybrid = bb.HybridScorer(bayes, vector)

    q = queries[0]
    doc = docs[0]

    print("query:", q.text)
    print("bm25:", bm25.score(q.terms, doc))
    print("bayes:", bayes.score(q.terms, doc))
    print("or:", hybrid.score_or(q.terms, q.embedding, doc))
    print("and:", hybrid.score_and(q.terms, q.embedding, doc))


if __name__ == "__main__":
    main()
