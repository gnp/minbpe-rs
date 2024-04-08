use minbpe::GPT4Tokenizer;
use minbpe::RegexTokenizerTrait;

fn main() {
    let text = "\u{1e01b}%SΣ";

    // Pre-initialize the tokenizer
    let start = std::time::Instant::now();
    GPT4Tokenizer::initialize();
    let duration = start.elapsed();
    println!(
        "GPT4Tokenizer static initialization completed in: {:?}",
        duration
    );

    // Initialize the tokenizer
    let start = std::time::Instant::now();
    let tokenizer = GPT4Tokenizer::default();
    let duration = start.elapsed();
    println!(
        "GPT4Tokenizer default instance construction completed in: {:?}",
        duration
    );

    // Encode the string
    let start = std::time::Instant::now();
    let tokens = tokenizer.encode(text);
    let duration = start.elapsed();
    println!(
        "GPT4Tokenizer encoding of {} character string completed in: {:?}",
        text.len(),
        duration
    );

    // Print the resulting tokens
    println!("{:?}", tokens);
}
