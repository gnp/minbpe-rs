# `minbpe-rs` 

> Port of Andrej Karpathy's [minbpe](https://github.com/karpathy/minbpe) to Rust.

[![minbpe-rs crate](https://img.shields.io/crates/v/minbpe.svg)](https://crates.io/crates/minbpe)
[![minbpe-rs documentation](https://docs.rs/minbpe/badge.svg)](https://docs.rs/minbpe)


## Quick Start

Create a Rust application crate with `cargo`,

```
$> cargo new minbpe-test
```

In the resulting project, add `minbpe` to `Cargo.toml`,

```toml
[dependencies]
minbpe = "0.1.0"
```

Refer [`crates.io`](https://crates.io/crates/minbpe) for selecting the latest version. Next in `src/main.rs`,

```rust
use std::path::Path;
use minbpe::{BasicTokenizer, Saveable, Tokenizer, Trainable};

fn main() {
    let text = "aaabdaaabac" ;
    let mut tokenizer = BasicTokenizer::new() ;
    tokenizer.train( text , 256 + 3 , false ) ;
    println!( "{:?}" , tokenizer.encode(text) ) ;
    println!( "{:?}" , tokenizer.decode( &[258, 100, 258, 97, 99] ) ) ;
    tokenizer.save( Path::new( "./" ) , "toy" ) ;
}
```

Execute the binary with `cargo run`,

```
$> cargo run

   ...
   Compiling minbpe-test v0.1.0 (~/minbpe-test)
    Finished dev [unoptimized + debuginfo] target(s) in 15.71s
     Running `target/debug/minbpe-test`
[258, 100, 258, 97, 99]
"aaabdaaabac"

```


## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.


## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
