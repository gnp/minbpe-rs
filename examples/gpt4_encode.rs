use std::fs;
use std::path::PathBuf;

use minbpe::GPT4Tokenizer;
use minbpe::RegexTokenizerTrait;

fn main() -> std::io::Result<()> {
    let file_path = PathBuf::from("tests/taylorswift.txt");

    // Pre-initialize the tokenizer
    println!("Pre-initializing the tokenizer...");
    let start = std::time::Instant::now();
    GPT4Tokenizer::initialize();
    let duration = start.elapsed();
    println!(
        "GPT4Tokenizer static initialization completed in: {:?}",
        duration
    );

    // Get default instance of the tokenizer
    println!("Getting a default instance of GPT4Tokenizer...");
    let start = std::time::Instant::now();
    let tokenizer = GPT4Tokenizer::default();
    let duration = start.elapsed();
    println!(
        "GPT4Tokenizer default instance construction completed in: {:?}",
        duration
    );

    // Read the input file
    println!("Reading file: {:?}...", file_path);
    let start = std::time::Instant::now();
    let text = fs::read_to_string(file_path)?;
    let duration = start.elapsed();
    println!(
        "Reading {} characters completed in: {:?}",
        text.len(),
        duration
    );

    // Timing the encoding process, optional.
    let start = std::time::Instant::now();
    let tokens = tokenizer.encode(&text);
    let duration = start.elapsed();

    println!("Encoding completed in: {:?}", duration);
    println!("Produced {} encoded tokens", tokens.len());

    Ok(())
}
