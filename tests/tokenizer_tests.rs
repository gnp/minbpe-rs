#[cfg(test)]
mod tests {
    use minbpe::test_common::{LLAMA_TEXT, SPECIAL_TOKENS};
    use minbpe::AllowedSpecial;
    use minbpe::BasicTokenizer;
    use minbpe::Loadable;
    use minbpe::RegexTokenizerStruct;
    use minbpe::RegexTokenizerTrait;
    use minbpe::Saveable;
    use minbpe::Token;
    use minbpe::Trainable;

    use indexmap::IndexMap;
    use tempfile::tempdir;

    // Quick unit test, following along the Wikipedia example:
    // https://en.wikipedia.org/wiki/Byte_pair_encoding
    //
    // According to Wikipedia, running bpe on the input string:
    // "aaabdaaabac"
    //
    // for 3 merges will result in string:
    // "XdXac"
    //
    // where:
    // X=ZY
    // Y=ab
    // Z=aa
    //
    // Keep in mind that for us a=97, b=98, c=99, d=100 (ASCII values)
    // so Z will be 256, Y will be 257, X will be 258.
    //
    // So we expect the output list of ids to be [258, 100, 258, 97, 99]
    fn test_wikipedia_example_inner(tokenizer: &mut Box<dyn Trainable>) {
        let text = "aaabdaaabac";
        tokenizer.train(text, 256 + 3, false);
        let ids = tokenizer.encode(text);
        assert_eq!(ids, [258, 100, 258, 97, 99]);
        let encoded = tokenizer.encode(text);
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_wikipedia_example() {
        let tokenizers: Vec<Box<dyn Trainable>> = vec![
            Box::new(BasicTokenizer::new()),
            Box::<RegexTokenizerStruct>::default(),
        ];

        for mut tokenizer in tokenizers {
            test_wikipedia_example_inner(&mut tokenizer);
        }
    }

    fn test_save_load_inner(special_tokens: &IndexMap<String, Token>) {
        // take a bit more complex piece of text and train the tokenizer
        let text = LLAMA_TEXT;
        // create a Tokenizer and do 64 merges
        let mut tokenizer = RegexTokenizerStruct::default();
        tokenizer.train(text, 256 + 64, false);
        tokenizer.set_special_tokens(special_tokens.clone()); // Feels weird to do this after training, not part of setup

        // verify that decode(encode(x)) == x
        let encoded = tokenizer.encode_special(text, AllowedSpecial::All);
        let decoded = tokenizer.decode(&encoded);
        assert_eq!(decoded, text);

        // verify that save/load work as expected; save the tokenizer
        let dir = tempdir().unwrap();
        tokenizer.save(dir.path(), "test_tokenizer_tmp");

        // re-load the tokenizer
        let mut tokenizer = RegexTokenizerStruct::default();
        let model_file = dir.path().join("test_tokenizer_tmp.model");
        tokenizer.load(&model_file);

        // verify that decode(encode(x)) == x
        assert_eq!(tokenizer.decode(&encoded), text);
        assert_eq!(
            tokenizer.decode(&tokenizer.encode_special(text, AllowedSpecial::All)),
            text
        );
        assert_eq!(tokenizer.encode_special(text, AllowedSpecial::All), encoded);
    }

    #[test]
    fn test_save_load() {
        let special_tokens = IndexMap::new();
        test_save_load_inner(&special_tokens);
        let special_tokens = &SPECIAL_TOKENS;
        test_save_load_inner(special_tokens);
    }
}
