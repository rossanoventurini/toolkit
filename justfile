
perf_get_bits:
    RUSTFLAGS='-C target-cpu=native' cargo build --release
    ./target/release/perf_get_bits

perf_darray:
    RUSTFLAGS='-C target-cpu=native' cargo build --release
    ./target/release/perf_darray

perf_ef:
    RUSTFLAGS='-C target-cpu=native' cargo build --release
    ./target/release/perf_elias_fano