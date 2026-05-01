#!/bin/bash
cd knowledge
unzip rust-lang.zip
cd ..
cargo run --release
cd knowledge
zip -r rust-lang.zip rust-lang/
rm -rf rust-lang/
