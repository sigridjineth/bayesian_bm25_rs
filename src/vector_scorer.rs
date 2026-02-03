use crate::corpus::Document;
use crate::math_utils::{clamp, cosine_similarity};

#[derive(Default)]
pub struct VectorScorer;

impl VectorScorer {
    pub fn new() -> Self {
        Self
    }

    pub fn score_to_probability(&self, sim: f64) -> f64 {
        clamp((1.0 + sim) / 2.0, 0.0, 1.0)
    }

    pub fn score(&self, query_embedding: &[f64], doc: &Document) -> f64 {
        let sim = cosine_similarity(query_embedding, &doc.embedding);
        self.score_to_probability(sim)
    }
}
