//! Contains the base Tokenizer struct and a few common helper functions.
//! The base struct also contains the (common) save/load functionality.
//! It would be possible to be a lot more strict about the interface and
//! e.g. isolating all regex/pattern parts to the RegexTokenizer, but
//! some concessions are made for simplicity.

use std::io::Write;
use std::path::Path;
use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use indexmap::IndexMap;

/// Token type to support up to 2^31 distinct tokens. It is signed in case a Tokenizer
/// needs to use negative values for special tokens.
pub type Token = i32;

/// Count type to support up to 2^64 occurences of any token pair.
pub type Count = u64;

/// Base trait for Tokenizers to implement.
pub trait Tokenizer {
    fn special_tokens(&self) -> &IndexMap<String, Token>;

    fn merges(&self) -> &IndexMap<(Token, Token), Token>;

    fn vocab(&self) -> &IndexMap<Token, Vec<u8>>;

    /// A Tokenizer can encode a string into a list of integers.
    fn encode(&self, text: &str) -> Vec<Token>;

    /// A Tokenizer can decode a list of integers into a string.
    fn decode(&self, ids: &[Token]) -> String;
}

/// A Tokenizer that can be trained.
pub trait Trainable: Tokenizer {
    /// Train a vocabulary of size `vocab_size` in distinct Tokens from `text`.
    fn train(&mut self, text: &str, vocab_size: Token, verbose: bool);
}

pub trait Saveable: Tokenizer {
    fn pattern(&self) -> &str;

    /// Saves the tokenizer's model and vocabulary to two files:
    /// - `file_prefix.model`: The model file used for loading the tokenizer.
    /// - `file_prefix.vocab`: A human-readable version of the vocabulary for inspection.
    ///
    /// This is inspired by (but not equivalent to) SentencePiece's model saving.
    ///
    /// # Arguments
    ///
    /// * `dir` - The path to the output directory.
    /// * `prefix` - The prefix for the output file name.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tempfile::tempdir;
    /// use minbpe::Saveable;
    /// use minbpe::Tokenizer;
    /// use minbpe::BasicTokenizer;
    /// let tokenizer = BasicTokenizer::new();
    /// let dir = tempdir().unwrap();
    /// let path = dir.path();
    /// tokenizer.save(&path, "prefix");
    /// ```
    fn save(&self, dir: &Path, prefix: &str) {
        // let dir = dir.as_ref();

        // Write the model file (used for loading the tokenizer later)
        let model_file_path = dir.join(format!("{}.model", prefix));
        let mut model_file = File::create(model_file_path).expect("Unable to create model file");

        // Write the version, pattern, and merges
        writeln!(model_file, "minbpe v1").expect("Unable to write to model file");
        writeln!(model_file, "{}", self.pattern()).expect("Unable to write to model file");

        // Write the special tokens (first the number, then each token and its index)
        writeln!(model_file, "{}", self.special_tokens().len())
            .expect("Unable to write to model file");
        for (special, idx) in self.special_tokens() {
            writeln!(model_file, "{} {}", special, idx).expect("Unable to write to model file");
        }

        let mut merges: Vec<(&(Token, Token), &Token)> = self.merges().iter().collect();
        merges.sort_by_key(|&k| k.1);

        // Write the merges dictionary
        for (token_pair, _new_token) in merges {
            writeln!(model_file, "{} {}", token_pair.0, token_pair.1)
                .expect("Unable to write to model file");
        }

        // Write the vocabulary file (for human inspection)
        let vocab_file_path = dir.join(format!("{}.vocab", prefix));
        let mut vocab_file = File::create(vocab_file_path).expect("Unable to create vocab file");

        // Invert the merges dictionary for easier lookup
        let inverted_merges: IndexMap<Token, (Token, Token)> = self
            .merges()
            .iter()
            .map(|((idx1, idx2), idx)| (*idx, (*idx1, *idx2)))
            .collect();

        let vocab = self.vocab();

        for (idx, token) in vocab {
            // Render the token, replacing invalid UTF-8 sequences with the replacement character
            let s = render_token(token);

            if let Some((idx0, idx1)) = inverted_merges.get(idx) {
                // If the token has children, render it as a merge
                let s0 = render_token(&vocab[idx0]);
                let s1 = render_token(&vocab[idx1]);
                writeln!(vocab_file, "[{}][{}] -> [{}] {}", s0, s1, s, idx)
                    .expect("Unable to write to vocab file");
            } else {
                // Otherwise, it's a leaf token (one of the first 256 bytes)
                writeln!(vocab_file, "[{}] {}", s, idx).expect("Unable to write to vocab file");
            }
        }
    }
}

pub trait Loadable: Tokenizer {
    fn set_pattern(&mut self, pattern: &str);
    fn set_special_tokens(&mut self, special_tokens: IndexMap<String, Token>);
    fn set_merges(&mut self, merges: IndexMap<(Token, Token), Token>);
    fn set_vocab(&mut self, vocab: IndexMap<Token, Vec<u8>>);

    /// Loads the tokenizer's model from a file.
    ///
    /// This is the inverse of `save` but only for the model file.
    ///
    /// # Arguments
    ///
    /// * `model_file` - The path to the model file.
    ///
    /// # Panics
    ///
    /// Panics if the model file does not have a ".model" extension or if the file format is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::path::PathBuf;
    /// use minbpe::Loadable;
    /// use minbpe::Tokenizer;
    /// use minbpe::BasicTokenizer;
    /// let mut tokenizer = BasicTokenizer::new();
    /// let model_path = PathBuf::from("examples/basic_example.model");
    /// tokenizer.load(&model_path);
    /// ```
    fn load(&mut self, model_file: &Path) {
        // FIXME: Return a Result instead of panicking
        // let model_file = model_file.as_ref();
        assert!(
            model_file.extension().map_or(false, |ext| ext == "model"),
            "Model file must have a .model extension"
        );

        let mut merges: IndexMap<(Token, Token), Token> = IndexMap::new();
        let mut special_tokens: IndexMap<String, Token> = IndexMap::new();
        let mut idx: Token = 256;

        let file = File::open(model_file).expect("Unable to open model file");
        let reader = BufReader::new(file);

        let lines: Vec<String> = reader
            .lines()
            .map(|line| line.expect("Unable to read line from model file"))
            .collect();

        let mut line_iter = lines.iter();

        if let Some(version) = line_iter.next() {
            assert_eq!(version, "minbpe v1", "Invalid model file version");
        } else {
            panic!("Missing version line in model file");
        }

        // FIXME: Check whether Tokenizer supports a Pattern at all.

        if let Some(pattern) = line_iter.next() {
            self.set_pattern(pattern);
        } else {
            panic!("Missing pattern line in model file");
        }

        if let Some(num_special_str) = line_iter.next() {
            let num_special = num_special_str
                .parse::<Token>()
                .expect("Invalid number of special tokens");

            // FIXME: Check whether Tokenizer supports Special Tokens at all.
            // FIXME: Ensure it is >= 0 because Token type is signed.
            // FIXME: Enforce some reasonable maximum less than 2^31.

            for _ in 0..num_special {
                if let Some(special_line) = line_iter.next() {
                    let mut parts = special_line.split_whitespace();
                    let special = parts.next().expect("Missing special token").to_string();
                    let special_idx = parts
                        .next()
                        .expect("Missing special token index")
                        .parse::<Token>()
                        .expect("Invalid special token index");
                    special_tokens.insert(special, special_idx);
                } else {
                    panic!("Missing special token line in model file");
                }
            }
        } else {
            panic!("Missing number of special tokens line in model file");
        }

        for merge_line in line_iter {
            let mut parts = merge_line.split_whitespace();
            let idx1 = parts
                .next()
                .expect("Missing first index")
                .parse::<Token>()
                .expect("Invalid first index");
            let idx2 = parts
                .next()
                .expect("Missing second index")
                .parse::<Token>()
                .expect("Invalid second index");
            merges.insert((idx1, idx2), idx);
            idx += 1;
        }

        let vocab = build_vocab(&special_tokens, &merges);

        self.set_special_tokens(special_tokens);
        self.set_merges(merges);
        self.set_vocab(vocab);
    }
}

/// Additional operations for Tokenizers.
/// Given a slice of integers, returns a new `IndexMap` containing the counts of consecutive pairs.
///
/// Example:
/// ```
/// # use indexmap::IndexMap;
/// # use minbpe::get_stats;
/// let ids = vec![1, 2, 3, 1, 2];
/// let counts = get_stats(&ids);
/// assert_eq!(counts, IndexMap::from([((1, 2), 2), ((2, 3), 1), ((3, 1), 1)]));
/// ```
pub fn get_stats(ids: &[Token]) -> IndexMap<(Token, Token), Count> {
    let mut counts = IndexMap::new();
    update_stats(ids, &mut counts);
    counts
}

/// Updates an existing `IndexMap` with the counts of consecutive pairs from the given slice of integers.
///
/// Example:
/// ```
/// # use indexmap::IndexMap;
/// # use minbpe::update_stats;
/// let ids = vec![1, 2, 3, 1, 2];
/// let mut existing_counts = IndexMap::from([((1, 2), 1), ((2, 3), 1)]);
/// update_stats(&ids, &mut existing_counts);
/// assert_eq!(existing_counts, IndexMap::from([((1, 2), 3), ((2, 3), 2), ((3, 1), 1)]));
/// ```
pub fn update_stats(ids: &[Token], counts: &mut IndexMap<(Token, Token), Count>) {
    for pair in ids.windows(2) {
        let pair = (pair[0], pair[1]);
        *counts.entry(pair).or_insert(0) += 1;
    }
}

/// Given an `IndexMap` of consecutive pair counts, returns the pair with the highest count. This
/// technique preserves the insertion order of the pairs that IndexMap maintains, returning the
/// first-inserted pair with the highest count.
pub fn get_max_entry(stats: &IndexMap<(Token, Token), Count>) -> Option<(&(Token, Token), &Count)> {
    let mut max_entry = None;

    for entry in stats.iter() {
        match max_entry {
            None => max_entry = Some(entry),
            Some((_, max_count)) => {
                let (_, count) = entry;
                if count > max_count {
                    max_entry = Some(entry);
                }
            }
        }
    }

    max_entry
}

/// Merges consecutive occurrences of a pair of integers in the given slice,
/// replacing them with a new integer.
///
/// Arguments:
/// - `ids`: The slice of Tokens to merge.
/// - `pair`: The pair of consecutive integers to replace.
/// - `new_id`: The new integer to replace the consecutive pairs with.
///
/// Returns:
/// A new `Vec<Token>` with the merged Tokens.
///
/// Example:
/// ```
/// # use minbpe::merge;
/// let ids = vec![1, 2, 3, 1, 2];
/// let pair = (1, 2);
/// let new_id = 4;
/// let merged = merge(&ids, pair, new_id);
/// assert_eq!(merged, vec![4, 3, 4]);
/// ```
pub fn merge(ids: &[Token], pair: (Token, Token), new_id: Token) -> Vec<Token> {
    let mut new_ids = Vec::with_capacity(ids.len());
    let mut i = 0;

    while i < ids.len() {
        if i < ids.len() - 1 && ids[i] == pair.0 && ids[i + 1] == pair.1 {
            new_ids.push(new_id);
            i += 2;
        } else {
            new_ids.push(ids[i]);
            i += 1;
        }
    }

    new_ids
}

/// vocab is simply and deterministically derived from merges
pub fn build_vocab(
    special_tokens: &IndexMap<String, Token>,
    merges: &IndexMap<(Token, Token), Token>,
) -> IndexMap<Token, Vec<u8>> {
    let mut vocab: IndexMap<Token, Vec<u8>> = (0..256).map(|idx| (idx, vec![idx as u8])).collect();

    for ((p0, p1), idx) in merges {
        let mut token = vocab[p0].clone();
        token.extend_from_slice(&vocab[p1]);
        vocab.insert(*idx, token);
    }

    for (special, idx) in special_tokens {
        vocab.insert(*idx, special.as_bytes().to_vec());
    }

    vocab
}

/// Replaces control characters in the given string with their Unicode escape sequences.
///
/// Control characters are characters that distort the output, such as newline ('\n') or
/// other characters that fall under the Unicode category "C" (Other).
///
/// References:
/// - https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
/// - http://www.unicode.org/reports/tr44/#GC_Values_Table
///
/// Arguments:
/// - `s`: The string to process.
///
/// Returns:
/// A new `String` with control characters replaced by their Unicode escape sequences.
///
/// Example:
/// ```ignore
/// # use minbpe::tokenizer::replace_control_characters;
/// let s = "Hello\nWorld\u{7}!";
/// let result = replace_control_characters(s);
/// assert_eq!(result, "Hello\\u000aWorld\\u0007!");
/// ```
fn replace_control_characters(s: &str) -> String {
    let mut chars = String::with_capacity(s.len());

    for ch in s.chars() {
        if ch.is_control() {
            let escaped = format!("\\u{:04x}", ch as u32);
            chars.push_str(&escaped);
        } else {
            chars.push(ch);
        }
    }

    chars
}

/// Pretty-prints a token by decoding it as UTF-8 and escaping control characters.
///
/// Arguments:
/// - `token`: The token as a byte slice.
///
/// Returns:
/// A `String` representation of the token with control characters escaped.
///
/// Example:
/// ```ignore
/// # use minbpe::tokenizer::render_token;
/// let token = b"Hello\nWorld\x07!";
/// let result = render_token(token);
/// assert_eq!(result, "Hello\\u000aWorld\\u0007!");
/// ```
fn render_token(token: &[u8]) -> String {
    let s = String::from_utf8_lossy(token);
    replace_control_characters(&s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replace_control_characters() {
        let s = "Hello\nWorld\u{7}!";
        let result = replace_control_characters(s);
        assert_eq!(result, "Hello\\u000aWorld\\u0007!");
    }

    #[test]
    fn test_render_token() {
        let token = b"Hello\nWorld\x07!";
        let result = render_token(token);
        assert_eq!(result, "Hello\\u000aWorld\\u0007!");
    }

    #[test]
    fn test_indexmap_order() {
        let input_data: Vec<((Token, Token), Count)> = vec![
            ((0, 0), 2),
            ((1, 1), 12),
            ((2, 2), 18),
            ((3, 3), 11),
            ((4, 4), 1),
            ((5, 5), 9),
            ((6, 6), 99),
            ((7, 7), 7),
            ((8, 8), 20),
            ((9, 9), 99),
            ((10, 10), 99),
            ((11, 11), 99),
            ((12, 12), 4),
            ((13, 13), 99),
            ((14, 14), 19),
            ((15, 15), 99),
            ((16, 16), 5),
            ((17, 17), 99),
            ((18, 18), 99),
            ((19, 19), 7),
        ];

        let expected_max_key: (Token, Token) = (6, 6);

        let stats: IndexMap<(Token, Token), Count> = IndexMap::from_iter(input_data.clone());

        let keys: Vec<_> = stats.keys().collect();
        let input_keys: Vec<_> = input_data.iter().map(|(k, _)| k).collect();

        assert_eq!(keys, input_keys, "Keys are not in insertion order");

        let entries: Vec<_> = stats.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(
            entries,
            input_data.as_slice(),
            "Entries are not in insertion order"
        );

        let max_entry = get_max_entry(&stats);

        let pair = max_entry.expect("Stats is empty");

        assert_eq!(*pair.0, expected_max_key);
    }
}
