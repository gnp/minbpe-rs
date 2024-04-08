use base64::{engine::general_purpose, Engine as _};
use core::panic;
use fancy_regex::Regex;
use indexmap::IndexMap;
use lazy_static::lazy_static;

use crate::{RegexTokenizerTrait, Token, Tokenizer};

const GPT4_SPLIT_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

lazy_static! {
    static ref GPT4_SPLIT_COMPILED_PATTERN: Regex = Regex::new(GPT4_SPLIT_PATTERN).unwrap();
}

lazy_static! {
    static ref GPT4_SPECIAL_TOKENS: IndexMap<&'static str, Token> = {
        let mut map = IndexMap::new();
        map.insert("<|endoftext|>", 100257);
        map.insert("<|fim_prefix|>", 100258);
        map.insert("<|fim_middle|>", 100259);
        map.insert("<|fim_suffix|>", 100260);
        map.insert("<|endofprompt|>", 100276);
        map
    };
}

// We need this because tiktoken-rs does not expose the encoder and we need to recover the merges. If it did, we would
// use tiktoken_rs::cl100k_base() and get the encoder from there.
lazy_static! {
    static ref GPT4_MERGEABLE_RANKS: IndexMap<Vec<u8>, Token> = {
        // https://github.com/zurawiki/tiktoken-rs/blob/main/tiktoken-rs/assets/cl100k_base.tiktoken
        let cl100k_base: &str = include_str!("../assets/cl100k_base.tiktoken");

        // Also from tiktoken-rs's constructor
        let mut encoder = IndexMap::default();
        for line in cl100k_base.lines() {
            let mut parts = line.split(' ');
            let raw = parts.next().unwrap();
            let token = &general_purpose::STANDARD.decode(raw).unwrap();
            let rank: Token = parts.next().unwrap().parse().unwrap();
            if rank < 0 {
                panic!("Rank {} for token {:?} is negative", rank, token);
            }
            encoder.insert(token.clone(), rank);
        }
        encoder
    };
}

fn bpe(
    mergeable_ranks: &IndexMap<Vec<u8>, Token>,
    token: &[u8],
    max_rank: Option<Token>,
) -> Vec<Vec<u8>> {
    let mut parts: Vec<Vec<u8>> = token.iter().map(|&b| vec![b]).collect();
    loop {
        let mut min_idx = None;
        let mut min_rank = None;
        for (i, pair) in parts.windows(2).enumerate() {
            let rank = mergeable_ranks.get(&[pair[0].clone(), pair[1].clone()].concat());
            if let Some(rank) = rank {
                if min_rank.is_none() || rank < min_rank.unwrap() {
                    min_idx = Some(i);
                    min_rank = Some(rank);
                }
            }
        }
        if min_rank.is_none() || (max_rank.is_some() && *min_rank.unwrap() >= max_rank.unwrap()) {
            break;
        }
        let min_idx = min_idx.unwrap();
        parts[min_idx] = [parts[min_idx].clone(), parts[min_idx + 1].clone()].concat();
        parts.remove(min_idx + 1);
    }
    parts
}

fn recover_merges(mergeable_ranks: &IndexMap<Vec<u8>, Token>) -> IndexMap<(Token, Token), Token> {
    let mut merges = IndexMap::new();
    for (token, &rank) in mergeable_ranks {
        if token.len() == 1 {
            continue;
        }
        let pair = bpe(mergeable_ranks, token, Some(rank));
        assert_eq!(pair.len(), 2);
        let ix0 = mergeable_ranks[&pair[0]];
        let ix1 = mergeable_ranks[&pair[1]];
        merges.insert((ix0, ix1), rank);
    }
    merges
}

/// Does not implement Tokenizer trait because it cannot be trained, loaded or saved.
pub struct GPT4Tokenizer {
    special_tokens: IndexMap<String, Token>,
    inverse_special_tokens: IndexMap<Token, String>,
    merges: IndexMap<(Token, Token), Token>,
    vocab: IndexMap<Token, Vec<u8>>,

    byte_shuffle: IndexMap<u8, u8>,
    inverse_byte_shuffle: IndexMap<u8, u8>,
}

impl Default for GPT4Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl GPT4Tokenizer {
    /// This method may be called before any other method in this module, in case you want to ensure all the
    /// lazy static initializations are done before any other operation.
    pub fn initialize() {
        let _ = &*GPT4_SPLIT_COMPILED_PATTERN;
        let _ = &*GPT4_MERGEABLE_RANKS;
    }

    pub fn new() -> Self {
        // let enc = cl100k_base().unwrap();
        let mergeable_ranks = &GPT4_MERGEABLE_RANKS;
        let merges = recover_merges(mergeable_ranks);
        let mut vocab: IndexMap<Token, Vec<u8>> =
            (0..=255).map(|i| (i as Token, vec![i])).collect();
        for (&(p0, p1), &idx) in &merges {
            let mut token = vocab[&p0].clone();
            token.extend(vocab[&p1].clone());
            vocab.insert(idx, token);
        }
        let byte_shuffle: IndexMap<u8, u8> = (0..=255)
            .map(|i| {
                let value = mergeable_ranks[&vec![i]];
                if value < 0 || value > u8::MAX as Token {
                    panic!(
                        "Value {} for key {} in mergeable_ranks does not fit in u8",
                        value, i
                    );
                }
                (i, value as u8)
            })
            .collect();
        let inverse_byte_shuffle: IndexMap<u8, u8> =
            byte_shuffle.iter().map(|(&k, &v)| (v, k)).collect();
        let special_tokens = GPT4_SPECIAL_TOKENS
            .iter()
            .map(|(&k, &v)| (k.to_string(), v))
            .collect::<IndexMap<String, Token>>();

        let inverse_special_tokens = special_tokens
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();

        GPT4Tokenizer {
            special_tokens,
            inverse_special_tokens,
            merges,
            vocab,

            byte_shuffle,
            inverse_byte_shuffle,
        }
    }

    pub fn decode(&self, ids: &[Token]) -> String {
        let text_bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&idx| self.vocab[&idx].clone())
            .collect();
        let text_bytes: Vec<u8> = text_bytes
            .into_iter()
            .map(|b| self.inverse_byte_shuffle[&b])
            .collect();
        String::from_utf8_lossy(&text_bytes).to_string()
    }

    pub fn register_special_tokens_x(&mut self, tokens: &IndexMap<String, Token>) {
        self.special_tokens
            .extend(tokens.iter().map(|(k, &v)| (k.clone(), v)));

        self.inverse_special_tokens = self
            .special_tokens
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();
    }
}

impl Tokenizer for GPT4Tokenizer {
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
        let mut text = String::new();
        for &id in ids {
            if let Some(token) = self.vocab.get(&id) {
                text.push_str(std::str::from_utf8(token).expect("Invalid UTF-8 sequence"));
            } else if let Some(token) = self.inverse_special_tokens.get(&id) {
                text.push_str(token);
            }
        }
        text
    }

    fn encode(&self, text: &str) -> Vec<Token> {
        RegexTokenizerTrait::encode(self, text)
    }
}

impl RegexTokenizerTrait for GPT4Tokenizer {
    fn encode_chunk(&self, text_bytes: &[u8]) -> Vec<Token> {
        let text_bytes: Vec<u8> = text_bytes.iter().map(|&b| self.byte_shuffle[&b]).collect();
        <Self as RegexTokenizerTrait>::encode_chunk_inner(self, &text_bytes)
    }

    fn compiled_pattern(&self) -> &Regex {
        &GPT4_SPLIT_COMPILED_PATTERN
    }

    fn inverse_special_tokens(&self) -> &IndexMap<Token, String> {
        &self.inverse_special_tokens
    }
}
