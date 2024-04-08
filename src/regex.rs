use fancy_regex::Regex;
use indexmap::IndexMap;
use std::collections::HashSet;

use crate::{get_max_entry, Loadable, Saveable, Trainable};
use crate::{get_stats, merge, update_stats, Token, Tokenizer};

/// The main GPT text split patterns, see
/// https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
pub const GPT2_SPLIT_PATTERN: &str =
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

pub const GPT4_SPLIT_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

/// Specifies how to handle special tokens during encoding.
///
/// This enum is used to control the behavior of the `encode_special` function
/// when encountering special tokens in the text.
///
/// # Variants
///
/// - `All`: Allow all special tokens during encoding.
///   Special tokens will be encoded according to their corresponding token IDs.
///
/// - `None`: Ignore all special tokens during encoding.
///   Special tokens will be treated as regular text and encoded using the standard encoding process.
///
/// - `NoneRaise`: Raise an error if any special token is encountered in the text during encoding.
///   This is the default behavior of the `tiktoken` library.
///
/// - `Set(HashSet<String>)`: Allow only the special tokens specified in the provided `HashSet`.
///   Special tokens not included in the set will be treated as regular text and encoded using the standard encoding process.
///
/// # Examples
///
/// ```
/// use minbpe::AllowedSpecial;
/// use std::collections::HashSet;
///
/// // Allow all special tokens
/// let allowed_all = AllowedSpecial::All;
///
/// // Ignore all special tokens
/// let allowed_none = AllowedSpecial::None;
///
/// // Raise an error if any special token is encountered
/// let allowed_none_raise = AllowedSpecial::NoneRaise;
///
/// // Allow only specific special tokens
/// let custom_set = HashSet::from(["<|endoftext|>".to_string(), "<|startoftext|>".to_string()]);
/// let allowed_custom = AllowedSpecial::Set(custom_set);
/// ```
pub enum AllowedSpecial {
    All,
    None,
    NoneRaise,
    Set(HashSet<String>),
}

pub trait RegexTokenizerTrait: Tokenizer {
    fn encode_chunk_inner(&self, text_bytes: &[u8]) -> Vec<Token> {
        let merges = self.merges();
        let mut ids: Vec<Token> = text_bytes.iter().map(|&b| b as Token).collect();
        while ids.len() >= 2 {
            // Find the pair with the lowest merge index
            let stats = get_stats(&ids);

            let pair_opt = stats
                .keys()
                .filter_map(|&pair| merges.get(&pair).map(|_| pair))
                .min_by_key(|&pair| merges[&pair]);

            match pair_opt {
                None => break, // If there are no more merges available, break
                Some(pair) => {
                    // Otherwise, merge the best pair (lowest merge index)
                    let idx = merges[&pair];
                    ids = merge(&ids, pair, idx);
                }
            };
        }
        ids
    }

    fn encode_chunk(&self, text_bytes: &[u8]) -> Vec<Token> {
        self.encode_chunk_inner(text_bytes)
    }

    // fn pattern(&self) -> &str;
    // fn set_pattern(&mut self, pattern: &str);

    fn compiled_pattern(&self) -> &Regex;

    // fn special_tokens(&self) -> &IndexMap<String, Token>;
    // fn set_special_tokens(&mut self, special_tokens: IndexMap<String, Token>);

    fn inverse_special_tokens(&self) -> &IndexMap<Token, String>;

    // fn merges(&self) -> &IndexMap<(Token, Token), Token>;
    // fn set_merges(&mut self, merges: IndexMap<(Token, Token), Token>);

    // fn vocab(&self) -> &IndexMap<Token, Vec<u8>>;
    // fn set_vocab(&mut self, vocab: IndexMap<Token, Vec<u8>>);

    // fn train(&mut self, text: &str, vocab_size: Token, verbose: bool);
    // fn decode(&self, ids: &[Token]) -> String;
    // fn encode(&self, text: &str) -> Vec<Token>;

    fn decode(&self, ids: &[Token]) -> String {
        let mut part_bytes = Vec::new();
        for &idx in ids {
            if let Some(bytes) = self.vocab().get(&idx) {
                part_bytes.extend_from_slice(bytes);
            } else if let Some(special_token) = self.inverse_special_tokens().get(&idx) {
                part_bytes.extend_from_slice(special_token.as_bytes());
            } else {
                panic!("Invalid token id: {}", idx);
            }
        }
        String::from_utf8_lossy(&part_bytes).into_owned()
    }

    fn encode(&self, text: &str) -> Vec<Token> {
        self.encode_special(text, AllowedSpecial::NoneRaise)
    }

    /// Encoding that ignores any special tokens.
    fn encode_ordinary(&self, text: &str) -> Vec<Token> {
        let text_chunks: Vec<&str> = self
            .compiled_pattern()
            .find_iter(text)
            .map(|m| {
                let matched = m.unwrap();
                &text[matched.start()..matched.end()]
            })
            .collect();
        let mut ids = Vec::new();
        for chunk in text_chunks {
            let chunk_bytes = chunk.as_bytes();
            let chunk_ids = self.encode_chunk(chunk_bytes);
            ids.extend(chunk_ids);
        }
        ids
    }

    /// Encodes the given text into token IDs, handling special tokens.
    ///
    /// Unlike `encode_ordinary`, this function handles special tokens based on the `allowed_special` parameter.
    ///
    /// # Arguments
    ///
    /// * `text` - The text to encode.
    /// * `allowed_special` - Specifies how to handle special tokens. It can be one of the following:
    ///   - `AllowedSpecial::All`: Allow all special tokens.
    ///   - `AllowedSpecial::None`: Ignore all special tokens.
    ///   - `AllowedSpecial::NoneRaise`: Raise an error if any special token is encountered in the text.
    ///     This is the default behavior of the `tiktoken` library.
    ///   - `AllowedSpecial::Set(HashSet<String>)`: A custom set of allowed special tokens.
    ///
    /// # Panics
    ///
    /// Panics if `allowed_special` is set to `AllowedSpecial::NoneRaise` and any special token is encountered in the text.
    fn encode_special(&self, text: &str, allowed_special: AllowedSpecial) -> Vec<Token> {
        let special = match allowed_special {
            AllowedSpecial::All => self.special_tokens().clone(),
            AllowedSpecial::None => IndexMap::new(),
            AllowedSpecial::NoneRaise => {
                assert!(
                    self.special_tokens()
                        .keys()
                        .all(|token| !text.contains(token)),
                    "Special token found in text"
                );
                IndexMap::new()
            }
            AllowedSpecial::Set(special_tokens) => {
                let mut special = IndexMap::new();
                for token in special_tokens {
                    if let Some(&idx) = self.special_tokens().get(&token) {
                        special.insert(token, idx);
                    }
                }
                special
            }
        };

        if special.is_empty() {
            return self.encode_ordinary(text);
        }

        let special_pattern = "(".to_string()
            + &special
                .keys()
                .map(|k| regex::escape(k))
                .collect::<Vec<String>>()
                .join("|")
            + ")";

        let re = fancy_regex::Regex::new(&special_pattern).unwrap();
        let mut last_end = 0;
        let mut special_chunks = Vec::new();
        for m in re.find_iter(text) {
            let m = m.unwrap();
            // Push the text between matches
            special_chunks.push(&text[last_end..m.start()]);
            // Push the matched text
            special_chunks.push(&text[m.start()..m.end()]);
            last_end = m.end();
        }
        let remaining = &text[last_end..];
        if !remaining.is_empty() {
            special_chunks.push(remaining);
        }

        let mut ids = Vec::new();
        for part in special_chunks {
            if let Some(&idx) = special.get(part) {
                ids.push(idx);
            } else {
                ids.extend(self.encode_ordinary(part));
            }
        }
        ids
    }
}

/// Minimal (byte-level) Byte Pair Encoding tokenizer.
///
/// Algorithmically follows along the GPT tokenizer:
/// https://github.com/openai/gpt-2/blob/master/src/encoder.py
///
/// Unlike `BasicTokenizer`:
/// - `RegexTokenizer` handles an optional regex splitting pattern.
/// - `RegexTokenizer` handles optional special tokens.
///
/// # Examples
///
/// ```
/// use fancy_regex::Regex;
/// use minbpe::base::Loadable;
/// use minbpe::base::Tokenizer;
/// use minbpe::base::Trainable;
/// use minbpe::RegexTokenizerStruct;
/// use minbpe::RegexTokenizerTrait;
/// use minbpe::AllowedSpecial;
/// use indexmap::IndexMap;
///
/// let pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
/// let mut tokenizer = RegexTokenizerStruct::new(pattern.to_string());
/// let special_tokens = IndexMap::from([("<|endoftext|>".to_string(), 100257)]);
/// tokenizer.set_special_tokens(special_tokens);
///
/// let text = "Hello, world! This is a test.";
/// let vocab_size = 256 + 10;
/// let verbose = true;
///
/// tokenizer.train(text, vocab_size, verbose);
///
/// let encoded = tokenizer.encode_special(text, AllowedSpecial::NoneRaise);
/// let decoded = RegexTokenizerTrait::decode(&tokenizer, &encoded);
///
/// assert_eq!(text, decoded);
/// ```
pub struct RegexTokenizerStruct {
    pattern: String,
    compiled_pattern: Regex,
    special_tokens: IndexMap<String, Token>,
    inverse_special_tokens: IndexMap<Token, String>,
    merges: IndexMap<(Token, Token), Token>,
    vocab: IndexMap<Token, Vec<u8>>,
}

impl Default for RegexTokenizerStruct {
    fn default() -> Self {
        Self::new(GPT4_SPLIT_PATTERN.to_string())
    }
}

impl RegexTokenizerStruct {
    fn make(pattern: String) -> Self {
        let compiled_pattern = Regex::new(&pattern).unwrap();

        RegexTokenizerStruct {
            pattern,
            compiled_pattern,
            special_tokens: IndexMap::new(),
            inverse_special_tokens: IndexMap::new(),
            merges: IndexMap::new(),
            vocab: IndexMap::new(),
        }
    }

    pub fn new(pattern: String) -> Self {
        Self::make(pattern)
    }
}

impl Tokenizer for RegexTokenizerStruct {
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
        // Forwarding to the default implementation provided by RegexTokenizerTrait
        <Self as RegexTokenizerTrait>::decode(self, ids)
    }

    fn encode(&self, text: &str) -> Vec<Token> {
        // Forwarding to the default implementation provided by RegexTokenizerTrait
        <Self as RegexTokenizerTrait>::encode(self, text)
    }
}

impl Trainable for RegexTokenizerStruct {
    fn train(&mut self, text: &str, vocab_size: Token, verbose: bool) {
        assert!(vocab_size >= 256, "Vocab size must be at least 256");
        let num_merges = vocab_size - 256;

        // Split the text into chunks
        let text_chunks: Vec<&str> = self
            .compiled_pattern()
            .find_iter(text)
            .map(|m| {
                let matched = m.unwrap();
                &text[matched.start()..matched.end()]
            })
            .collect();

        // Input text preprocessing
        let mut ids: Vec<Vec<Token>> = text_chunks
            .iter()
            .map(|chunk| chunk.as_bytes().iter().map(|b| *b as Token).collect())
            .collect();

        // Iteratively merge the most common pairs to create new tokens
        let mut merges: IndexMap<(Token, Token), Token> = IndexMap::new();
        let mut vocab: IndexMap<Token, Vec<u8>> =
            (0..256).map(|idx| (idx, vec![idx as u8])).collect();

        for i in 0..num_merges {
            // Count the number of times every consecutive pair appears
            let mut stats = IndexMap::new();
            for chunk_ids in &ids {
                update_stats(chunk_ids, &mut stats);
            }

            // Find the pair with the highest count
            let pair = get_max_entry(&stats).unwrap().0;

            // Mint a new token: assign it the next available id
            let idx = 256 + i;

            // Replace all occurrences of pair in ids with idx
            ids = ids
                .iter()
                .map(|chunk_ids| merge(chunk_ids, *pair, idx))
                .collect();

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

impl Saveable for RegexTokenizerStruct {
    fn pattern(&self) -> &str {
        &self.pattern
    }
}

impl Loadable for RegexTokenizerStruct {
    fn set_pattern(&mut self, pattern: &str) {
        self.pattern = pattern.to_string();
        self.compiled_pattern = Regex::new(pattern).unwrap();
    }

    fn set_special_tokens(&mut self, special_tokens: IndexMap<String, Token>) {
        self.special_tokens = special_tokens.clone();
        self.inverse_special_tokens = special_tokens
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();
    }

    fn set_merges(&mut self, merges: IndexMap<(Token, Token), Token>) {
        self.merges = merges;
    }

    fn set_vocab(&mut self, vocab: IndexMap<Token, Vec<u8>>) {
        self.vocab = vocab;
    }
}

impl RegexTokenizerTrait for RegexTokenizerStruct {
    fn compiled_pattern(&self) -> &Regex {
        &self.compiled_pattern
    }

    fn inverse_special_tokens(&self) -> &IndexMap<Token, String> {
        &self.inverse_special_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;
    use std::collections::HashSet;

    #[test]
    fn test_pattern_matching() {
        let text = "Hello, world! <|endoftext|>";

        let pattern = "(<\\|endoftext\\|>)";
        let re = fancy_regex::Regex::new(pattern).unwrap();

        let mut last_end = 0;
        let mut special_chunks = Vec::new();
        for m in re.find_iter(text) {
            let m = m.unwrap();
            // Push the text between matches
            special_chunks.push(&text[last_end..m.start()]);
            // Push the matched text
            special_chunks.push(&text[m.start()..m.end()]);
            last_end = m.end();
        }
        let remaining = &text[last_end..];
        if !remaining.is_empty() {
            special_chunks.push(remaining);
        }
    }

    #[test]
    fn test_encode_special() {
        let mut tokenizer = RegexTokenizerStruct::default();
        tokenizer.train("Hello, world! Goodbye, world!, So long...", 256 + 10, true);

        let text = "Hello, world! <|endoftext|>";

        let special_tokens = IndexMap::from([("<|endoftext|>".to_string(), 100257)]);
        tokenizer.set_special_tokens(special_tokens);

        let encoded_all = tokenizer.encode_special(text, AllowedSpecial::All);
        let encoded_none = tokenizer.encode_special(text, AllowedSpecial::None);

        let custom_set = HashSet::from(["<|endoftext|>".to_string()]);
        let encoded_custom = tokenizer.encode_special(text, AllowedSpecial::Set(custom_set));

        assert!(encoded_all.contains(&100257));
        assert!(!encoded_none.contains(&100257));
        assert!(encoded_custom.contains(&100257));
    }

    #[test]
    #[should_panic]
    fn test_encode_special_panic() {
        let mut tokenizer = RegexTokenizerStruct::default();
        let text = "Hello, world! <|endofext|>";

        let special_tokens = IndexMap::from([("<|endofext|>".to_string(), 100257)]);
        tokenizer.set_special_tokens(special_tokens);

        // This should panic
        let _ = tokenizer.encode_special(text, AllowedSpecial::NoneRaise);
    }
}
