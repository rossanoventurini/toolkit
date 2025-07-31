 perf_get_bits:
    RUSTFLAGS='-C target-cpu=native' cargo build --release
    ./target/release/perf_get_bits