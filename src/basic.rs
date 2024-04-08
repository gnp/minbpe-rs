use indexmap::IndexMap;

use crate::base::{
    get_max_entry, get_stats, merge, Loadable, Saveable, Token, Tokenizer, Trainable,
};

/// Minimal (byte-level) Byte Pair Encoding tokenizer.
///
/// Algorithmically follows along the GPT tokenizer:
/// https://github.com/openai/gpt-2/blob/master/src/encoder.py
///
/// But:
/// - Does not handle the regular expression splitting pattern.
/// - Does not handle any special tokens.
///
/// # Examples
///
/// ```
/// use minbpe::BasicTokenizer;
/// use minbpe::Tokenizer;
/// use minbpe::Trainable;
///
/// let mut tokenizer = BasicTokenizer::new();
/// let text = "Hello, world!";
/// let vocab_size = 256;
/// let verbose = true;
///
/// tokenizer.train(text, vocab_size, verbose);
/// let encoded = tokenizer.encode(text);
/// let decoded = tokenizer.decode(&encoded);
///
/// assert_eq!(text, decoded);
/// ```
pub struct BasicTokenizer {
    special_tokens: IndexMap<String, Token>,
    merges: IndexMap<(Token, Token), Token>,
    vocab: IndexMap<Token, Vec<u8>>,
}

impl BasicTokenizer {
    pub fn new() -> Self {
        BasicTokenizer {
            special_tokens: IndexMap::new(),
            merges: IndexMap::new(),
            vocab: IndexMap::new(),
        }
    }
}

impl Default for BasicTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for BasicTokenizer {
    fn special_tokens(&self) -> &IndexMap<String, Token> {
        &self.special_tokens
    }

    fn merges(&self) -> &IndexMap<(Token, Token), Token> {
        &self.merges
    }

    fn vocab(&self) -> &IndexMap<Token, Vec<u8>> {
        &self.vocab
    }

    fn decode(&self, ids: &[Token]) -> String {
        // Given ids (list of integers), return Rust string
        let text_bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&idx| self.vocab[&idx].clone())
            .collect();
        String::from_utf8_lossy(&text_bytes).into_owned()
    }

    fn encode(&self, text: &str) -> Vec<Token> {
        // Given a string text, return the token ids
        let text_bytes = text.as_bytes();
        let mut ids: Vec<Token> = text_bytes.iter().map(|&b| b as Token).collect();
        while ids.len() >= 2 {
            // Find the pair with the lowest merge index
            let stats = get_stats(&ids);

            let pair_opt = stats
                .keys()
                .filter_map(|&pair| self.merges.get(&pair).map(|_| pair))
                .min_by_key(|&pair| self.merges[&pair]);

            match pair_opt {
                None => break, // If there are no more merges available, break
                Some(pair) => {
                    // Otherwise, merge the best pair (lowest merge index)
                    let idx = self.merges[&pair];
                    ids = merge(&ids, pair, idx);
                }
            };
        }
        ids
    }
}

impl Trainable for BasicTokenizer {
    fn train(&mut self, text: &str, vocab_size: Token, verbose: bool) {
        assert!(vocab_size >= 256, "Vocab size must be at least 256");
        let num_merges = vocab_size - 256;

        // Input text preprocessing
        let text_bytes = text.as_bytes();
        let mut ids: Vec<Token> = text_bytes.iter().map(|&b| b as Token).collect();

        // Iteratively merge the most common pairs to create new tokens
        let mut merges: IndexMap<(Token, Token), Token> = IndexMap::new();
        let mut vocab: IndexMap<Token, Vec<u8>> =
            (0..256).map(|idx| (idx, vec![idx as u8])).collect();
        for i in 0..num_merges {
            // Count up the number of times every consecutive pair appears
            let stats = get_stats(&ids);
            // Find the pair with the highest count
            let pair = get_max_entry(&stats).unwrap().0;
            // Mint a new token: assign it the next available id
            let idx = 256 + i;
            // Replace all occurrences of pair in ids with idx
            ids = merge(&ids, *pair, idx);
            // Save the merge
            merges.insert(*pair, idx);
            vocab.insert(
                idx,
                [vocab[&pair.0].clone(), vocab[&pair.1].clone()].concat(),
            );
            // Prints
            if verbose {
                println!(
                    "merge {}/{}: {:?} -> {} ({:?}) had {} occurrences",
                    i + 1,
                    num_merges,
                    pair,
                    idx,
                    vocab[&idx],
                    stats[pair]
                );
            }
        }

        // Save instance variables
        self.merges = merges;
        self.vocab = vocab; // FIXME: vs. build_vocab(&self.special_tokens, &self.merges);
    }
}

impl Saveable for BasicTokenizer {
    fn pattern(&self) -> &str {
        ""
    }
}

impl Loadable for BasicTokenizer {
    fn set_pattern(&mut self, pattern: &str) {
        let temp = pattern.trim();

        if !temp.is_empty() {
            panic!("Cannot set a non-empty pattern!")
        }
    }

    fn set_special_tokens(&mut self, special_tokens: IndexMap<String, Token>) {
        self.special_tokens = special_tokens;
    }

    fn set_merges(&mut self, merges: IndexMap<(Token, Token), Token>) {
        self.merges = merges;
    }

    fn set_vocab(&mut self, vocab: IndexMap<Token, Vec<u8>>) {
        self.vocab = vocab;
    }
}
