perf_get_bits:
    RUSTFLAGS='-C target-cpu=native' cargo build --release
    ./target/release/perf_get_bits

perf_darray:
    RUSTFLAGS='-C target-cpu=native' cargo build --release
    ./target/release/perf_darray

perf_ef:
    RUSTFLAGS='-C target-cpu=native' cargo build --release
    ./target/release/perf_elias_fano

perf_vb:
    RUSTFLAGS='-C target-cpu=native' cargo build --release
    ./target/release/perf_streamvbyte

perf_fenwick_tree:
    RUSTFLAGS='-C target-cpu=native' cargo build --release
    ./target/release/perf_fenwick_tree -n 10000000 --iter --update --sum