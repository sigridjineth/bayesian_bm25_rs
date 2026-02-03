pub struct Tokenizer;

impl Tokenizer {
    pub fn new() -> Self {
        Self
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            if ch.is_ascii_alphanumeric() {
                current.push(ch.to_ascii_lowercase());
            } else if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        }

        if !current.is_empty() {
            tokens.push(current);
        }

        tokens
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}
