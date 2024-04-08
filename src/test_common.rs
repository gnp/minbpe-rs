use std::fs;
use std::path::PathBuf;

use crate::Token;

use indexmap::indexmap;
use indexmap::IndexMap;
use lazy_static::lazy_static;

lazy_static! {
    pub static ref SPECIAL_TOKENS: IndexMap<String, Token> = indexmap! {
      "<|endoftext|>".to_string() => 100257,
      "<|fim_prefix|>".to_string()=> 100258,
      "<|fim_middle|>".to_string()=> 100259,
      "<|fim_suffix|>".to_string()=> 100260,
      "<|endofprompt|>".to_string()=> 100276
    };
}

pub const LLAMA_TEXT: &str = r###"<|endoftext|>The llama (/ˈlɑːmə/; Spanish pronunciation: [ˈʎama] or [ˈʝama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era.
Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5–8 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4]
The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000–12,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5]
<|fim_prefix|>In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology,<|fim_suffix|> where they come from at the end of time.[6]<|fim_middle|> llamas will return to the water springs and ponds<|endofprompt|>"###;

// a few strings to test the tokenizers on
pub const TEST_STRINGS: [&str; 4] = [
    "",                                        // empty string
    "?",                                       // single character
    "hello world!!!? (안녕하세요!) lol123 😉", // fun small string
    "FILE:../tests/taylorswift.txt",           // FILE: is handled as a special string in unpack()
];

pub fn test_strings() -> [&'static str; 4] {
    TEST_STRINGS
}

pub fn unpack(text: &str) -> std::io::Result<String> {
    if let Some(filename) = text.strip_prefix("FILE:") {
        let dirname = PathBuf::from(file!()).parent().unwrap().to_path_buf();
        let file_path = dirname.join(filename);
        println!("Reading file: {:?}...", file_path);
        fs::read_to_string(file_path)
    } else {
        Ok(text.to_string())
    }
}
