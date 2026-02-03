use crate::math_utils::{sigmoid, EPSILON};

pub struct ParameterLearner {
    learning_rate: f64,
    max_iterations: usize,
    tolerance: f64,
}

pub struct ParameterLearnerResult {
    pub alpha: f64,
    pub beta: f64,
    pub loss_history: Vec<f64>,
    pub converged: bool,
}

impl ParameterLearner {
    pub fn new(learning_rate: f64, max_iterations: usize, tolerance: f64) -> Self {
        Self {
            learning_rate,
            max_iterations,
            tolerance,
        }
    }

    pub fn cross_entropy_loss(
        &self,
        scores: &[f64],
        labels: &[f64],
        alpha: f64,
        beta: f64,
    ) -> f64 {
        let n = scores.len();
        let mut total_loss = 0.0;
        for (s, y) in scores.iter().zip(labels.iter()) {
            let mut p = sigmoid(alpha * (s - beta));
            p = p.max(EPSILON).min(1.0 - EPSILON);
            total_loss -= y * p.ln() + (1.0 - y) * (1.0 - p).ln();
        }
        total_loss / n as f64
    }

    pub fn learn(&self, scores: &[f64], labels: &[f64]) -> ParameterLearnerResult {
        let mut alpha = 1.0;
        let mut beta = 0.0;
        let n = scores.len();
        let mut loss_history = Vec::new();

        for iteration in 0..self.max_iterations {
            let loss = self.cross_entropy_loss(scores, labels, alpha, beta);
            loss_history.push(loss);

            if iteration > 0 {
                let prev = loss_history[loss_history.len() - 2];
                if (prev - loss).abs() < self.tolerance {
                    return ParameterLearnerResult {
                        alpha,
                        beta,
                        loss_history,
                        converged: true,
                    };
                }
            }

            let mut grad_alpha = 0.0;
            let mut grad_beta = 0.0;
            for (s, y) in scores.iter().zip(labels.iter()) {
                let mut p = sigmoid(alpha * (s - beta));
                p = p.max(EPSILON).min(1.0 - EPSILON);
                let error = p - y;
                grad_alpha += error * (s - beta);
                grad_beta += error * (-alpha);
            }
            grad_alpha /= n as f64;
            grad_beta /= n as f64;

            alpha -= self.learning_rate * grad_alpha;
            beta -= self.learning_rate * grad_beta;
        }

        let final_loss = self.cross_entropy_loss(scores, labels, alpha, beta);
        loss_history.push(final_loss);

        ParameterLearnerResult {
            alpha,
            beta,
            loss_history,
            converged: false,
        }
    }
}
