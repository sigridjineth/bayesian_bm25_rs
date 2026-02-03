use std::collections::HashMap;

use crate::tokenizer::Tokenizer;

#[derive(Clone)]
pub struct Document {
    pub id: String,
    pub text: String,
    pub embedding: Vec<f64>,
    pub tokens: Vec<String>,
    pub length: usize,
    pub term_freq: HashMap<String, usize>,
}

pub struct Corpus {
    tokenizer: Tokenizer,
    documents: Vec<Document>,
    doc_by_id: HashMap<String, usize>,
    pub n: usize,
    pub avgdl: f64,
    pub df: HashMap<String, usize>,
}

impl Corpus {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            documents: Vec::new(),
            doc_by_id: HashMap::new(),
            n: 0,
            avgdl: 0.0,
            df: HashMap::new(),
        }
    }

    pub fn add_document(&mut self, doc_id: &str, text: &str, embedding: Vec<f64>) {
        let tokens = self.tokenizer.tokenize(text);
        let mut term_freq = HashMap::new();
        for token in &tokens {
            *term_freq.entry(token.clone()).or_insert(0) += 1;
        }

        let doc = Document {
            id: doc_id.to_string(),
            text: text.to_string(),
            embedding,
            length: tokens.len(),
            tokens,
            term_freq,
        };

        let idx = self.documents.len();
        self.documents.push(doc);
        self.doc_by_id.insert(doc_id.to_string(), idx);
    }

    pub fn build_index(&mut self) {
        self.n = self.documents.len();
        self.df.clear();
        let mut total_length = 0usize;

        for doc in &self.documents {
            total_length += doc.length;
            for term in doc.term_freq.keys() {
                *self.df.entry(term.clone()).or_insert(0) += 1;
            }
        }

        self.avgdl = if self.n > 0 {
            total_length as f64 / self.n as f64
        } else {
            0.0
        };
    }

    pub fn get_document(&self, doc_id: &str) -> Option<&Document> {
        self.doc_by_id
            .get(doc_id)
            .and_then(|idx| self.documents.get(*idx))
    }

    pub fn documents(&self) -> &[Document] {
        &self.documents
    }
}
