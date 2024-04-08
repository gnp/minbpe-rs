#[cfg(all(test, feature = "tiktoken_tests"))]
mod tests {
    use std::collections::HashSet;

    use lazy_static::lazy_static;
    use minbpe::{GPT4Tokenizer, RegexTokenizerTrait, Token};
    use proptest::prelude::*;
    use tiktoken_rs::{cl100k_base, CoreBPE};

    lazy_static! {
        static ref SPECIAL_TOKENS: HashSet<&'static str> = HashSet::new();
        static ref TIKTOKEN_ENC: CoreBPE = cl100k_base().unwrap();
        static ref GPT4_TOKENIZER: GPT4Tokenizer = GPT4Tokenizer::default();
    }

    fn test_one(s: &str) {
        let special_tokens = HashSet::new();

        let tiktoken_ids = TIKTOKEN_ENC.encode(s, special_tokens);
        let tiktoken_tokens: Vec<Token> = tiktoken_ids.iter().map(|&id| id as Token).collect();

        let gpt4_tokenizer_tokens = GPT4_TOKENIZER.encode(s);

        assert_eq!(tiktoken_tokens, gpt4_tokenizer_tokens);
    }

    #[test]
    fn test_high_char() {
        test_one("\u{1e01b}%SΣ");
    }

    proptest! {
        #[test]
        #[allow(unused_must_use)]
        fn gpt4_tokenizer_matches_tiktoken(s in "\\PC*") {
          test_one(&s);
        }
    }
}
