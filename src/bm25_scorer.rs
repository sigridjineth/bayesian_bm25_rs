use std::rc::Rc;

use crate::corpus::{Corpus, Document};

pub struct BM25Scorer {
    corpus: Rc<Corpus>,
    k1: f64,
    b: f64,
}

impl BM25Scorer {
    pub fn new(corpus: Rc<Corpus>, k1: f64, b: f64) -> Self {
        Self { corpus, k1, b }
    }

    pub fn idf(&self, term: &str) -> f64 {
        let n = self.corpus.n as f64;
        let df_t = *self.corpus.df.get(term).unwrap_or(&0) as f64;
        ((n - df_t + 0.5) / (df_t + 0.5)).ln()
    }

    fn length_norm(&self, doc: &Document) -> f64 {
        1.0 - self.b + self.b * (doc.length as f64) / self.corpus.avgdl
    }

    pub fn score_term_standard(&self, term: &str, doc: &Document) -> f64 {
        let tf = *doc.term_freq.get(term).unwrap_or(&0) as f64;
        if tf == 0.0 {
            return 0.0;
        }
        let norm = self.length_norm(doc);
        let idf_val = self.idf(term);
        idf_val * (self.k1 + 1.0) * tf / (self.k1 * norm + tf)
    }

    pub fn score_term_rewritten(&self, term: &str, doc: &Document) -> f64 {
        let tf = *doc.term_freq.get(term).unwrap_or(&0) as f64;
        if tf == 0.0 {
            return 0.0;
        }
        let norm = self.length_norm(doc);
        let boost = (self.k1 + 1.0) * tf / (self.k1 * norm + tf);
        let idf_val = self.idf(term);
        idf_val * boost
    }

    pub fn score(&self, query_terms: &[String], doc: &Document) -> f64 {
        query_terms
            .iter()
            .map(|term| self.score_term_standard(term, doc))
            .sum()
    }

    pub fn upper_bound(&self, term: &str) -> f64 {
        let idf_val = self.idf(term);
        if idf_val <= 0.0 {
            return 0.0;
        }
        (self.k1 + 1.0) * idf_val
    }

    pub fn avgdl(&self) -> f64 {
        self.corpus.avgdl
    }
}
