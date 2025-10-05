use mem_dbg::*;
use std::time::Instant;
use toolkit::stream_vbyte::{StreamVByte, StreamVByteRandomAccess};

fn generate_random_data(size: usize, max_value: u32) -> Vec<u32> {
    let mut data = Vec::with_capacity(size);
    let mut state = 12345u64; // Simple PRNG seed

    for _ in 0..size {
        // Simple linear congruential generator
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        data.push((state as u32) % max_value);
    }

    data
}

fn generate_random_data_u16(size: usize, max_value: u16) -> Vec<u16> {
    let mut data = Vec::with_capacity(size);
    let mut state = 12345u64; // Simple PRNG seed

    for _ in 0..size {
        // Simple linear congruential generator
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        data.push((state as u16) % max_value);
    }

    data
}

fn benchmark_random_access(data: &[u32]) {
    println!("    === RANDOM ACCESS BENCHMARK ===");

    // Create StreamVByteRandomAccess with block size 128
    let block_size = 256;
    let start_time = Instant::now();
    let svb_ra = StreamVByteRandomAccess::new(data, block_size);
    let create_time = start_time.elapsed();

    println!(
        "    Created random access structure in {:?} (block size: {})",
        create_time, block_size
    );

    // Generate random positions and lengths
    let num_queries = 1_000_000;
    let mut queries = Vec::with_capacity(num_queries);
    let mut state = 67890u64; // Different seed from data generation

    for _ in 0..num_queries {
        // Generate random range length between 150 and 250
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let range_length = 150 + (state % 101) as usize; // 150-250

        // Generate random start position
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let max_start = data.len().saturating_sub(range_length);
        let start_pos = (state % max_start as u64) as usize;

        queries.push(start_pos..start_pos + range_length);
    }

    println!(
        "    Generated {} random queries (range length: 150-250)",
        num_queries
    );

    // First run: Pure performance benchmark (no verification)
    let start_time = Instant::now();
    let mut total_elements = 0;

    // Allocate buffer once for the largest possible query (250 elements)
    let mut buffer = vec![0u32; 250];

    // Make next query depends on the previous one (seems to be unnneded becuse of the buffer)
    let mut prev = 0;
    for range in &queries {
        let mut range = range.clone();
        range.end = range.end - if prev % 2 == 0 { 0 } else { 1 };

        let range_len = range.len();
        svb_ra.get_range(&mut buffer[..range_len], range.clone());
        prev = buffer[0] as usize; // Just to introduce some dependency

        total_elements += range_len;
    }

    let query_time = start_time.elapsed();

    // Second run: Verification of all queries
    println!(
        "    Verifying correctness of all {} queries...",
        num_queries
    );
    let verification_start = Instant::now();

    for range in &queries {
        svb_ra.get_range(&mut buffer[..range.len()], range.clone());

        let expected = &data[range.clone()];
        assert_eq!(
            &buffer[..range.len()],
            expected,
            "Random access query failed for range {:?}",
            range
        );
    }

    println!(
        "    Memory usage (bytes): {}",
        svb_ra.mem_size(SizeFlags::default())
    );

    let verification_time = verification_start.elapsed();
    println!("    Verification completed in {:?}", verification_time);

    // Calculate rates (based on pure performance run, no verification overhead)
    let queries_per_second = num_queries as f64 / query_time.as_secs_f64();
    let elements_per_second = total_elements as f64 / query_time.as_secs_f64() / 1_000_000.0;
    let avg_query_time_us = query_time.as_micros() as f64 / num_queries as f64;

    println!("    Random access results (pure performance, no verification overhead):");
    println!("      Total queries: {}", num_queries);
    println!("      Total elements retrieved: {}", total_elements);
    println!("      Pure query time: {:?}", query_time);
    println!("      Queries per second: {:.0}", queries_per_second);
    println!("      Elements rate: {:.2} M int/s", elements_per_second);
    println!("      Avg time per query: {:.3} μs", avg_query_time_us);
    println!("    ✓ Random access correctness verified");

    let _ = svb_ra.mem_dbg(DbgFlags::default());
}

fn benchmark_random_access_u16(data: &[u16]) {
    println!("    === RANDOM ACCESS BENCHMARK (u16) ===");

    // Create StreamVByteRandomAccess with block size 128
    let block_size = 256;
    let start_time = Instant::now();
    let svb_ra = StreamVByteRandomAccess::new(data, block_size);
    let create_time = start_time.elapsed();

    println!(
        "    Created random access structure in {:?} (block size: {})",
        create_time, block_size
    );

    // Generate random positions and lengths
    let num_queries = 1_000_000;
    let mut queries = Vec::with_capacity(num_queries);
    let mut state = 67890u64; // Different seed from data generation

    for _ in 0..num_queries {
        // Generate random range length between 150 and 250
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let range_length = 150 + (state % 101) as usize; // 150-250

        // Generate random start position
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let max_start = data.len().saturating_sub(range_length);
        let start_pos = (state % max_start as u64) as usize;

        queries.push(start_pos..start_pos + range_length);
    }

    println!(
        "    Generated {} random queries (range length: 150-250)",
        num_queries
    );

    // First run: Pure performance benchmark (no verification)
    let start_time = Instant::now();
    let mut total_elements = 0;

    // Allocate buffer once for the largest possible query (250 elements)
    let mut buffer = vec![0u16; 250];

    // Make next query depends on the previous one (seems to be unnneded becuse of the buffer)
    let mut prev = 0;
    for range in &queries {
        let mut range = range.clone();
        range.end = range.end - if prev % 2 == 0 { 0 } else { 1 };

        let range_len = range.len();
        svb_ra.get_range(&mut buffer[..range_len], range.clone());
        prev = buffer[0] as usize; // Just to introduce some dependency

        total_elements += range_len;
    }

    let query_time = start_time.elapsed();

    // Second run: Verification of all queries
    println!(
        "    Verifying correctness of all {} queries...",
        num_queries
    );
    let verification_start = Instant::now();

    for range in &queries {
        svb_ra.get_range(&mut buffer[..range.len()], range.clone());

        let expected = &data[range.clone()];
        assert_eq!(
            &buffer[..range.len()],
            expected,
            "Random access query failed for range {:?}",
            range
        );
    }

    let verification_time = verification_start.elapsed();
    println!("    Verification completed in {:?}", verification_time);

    // Calculate rates (based on pure performance run, no verification overhead)
    let queries_per_second = num_queries as f64 / query_time.as_secs_f64();
    let elements_per_second = total_elements as f64 / query_time.as_secs_f64() / 1_000_000.0;
    let avg_query_time_us = query_time.as_micros() as f64 / num_queries as f64;

    println!("    Random access results (pure performance, no verification overhead):");
    println!("      Total queries: {}", num_queries);
    println!("      Total elements retrieved: {}", total_elements);
    println!("      Pure query time: {:?}", query_time);
    println!("      Queries per second: {:.0}", queries_per_second);
    println!("      Elements rate: {:.2} M int/s", elements_per_second);
    println!("      Avg time per query: {:.3} μs", avg_query_time_us);
    println!("    ✓ Random access correctness verified");

    let _ = svb_ra.mem_dbg(DbgFlags::default());
}

fn benchmark_streamvbyte() {
    println!("StreamVByte Benchmark");
    println!("====================");

    let test_sizes = vec![10_000]; // vec![1_000, 10_000, 100_000, 1_000_000];
    let max_values = vec![256, 65_536, 16_777_216, u32::MAX];

    for &size in &test_sizes {
        for &max_val in &max_values {
            println!("\nTesting {} elements, max value: {}", size, max_val);

            // Generate test data
            let data = generate_random_data(size, max_val);

            // Encode benchmark
            let start = Instant::now();
            let encoded = StreamVByte::encode(&data);
            let encode_time = start.elapsed();

            // Decode benchmark
            let start = Instant::now();
            let decoded = encoded.decode();
            let decode_time = start.elapsed();

            // Iterator benchmark
            let start = Instant::now();
            let iter_result: Vec<u32> = encoded.iter().collect();
            let iter_time = start.elapsed();

            // Verify correctness
            assert_eq!(
                data, decoded,
                "Decode failed for size {} max_val {}",
                size, max_val
            );
            assert_eq!(
                data, iter_result,
                "Iterator failed for size {} max_val {}",
                size, max_val
            );

            // Calculate metrics
            let original_bytes = size * 4; // 4 bytes per u32
            let encode_rate = size as f64 / encode_time.as_secs_f64() / 1_000_000.0;
            let decode_rate = size as f64 / decode_time.as_secs_f64() / 1_000_000.0;
            let iter_rate = size as f64 / iter_time.as_secs_f64() / 1_000_000.0;

            // Print results
            println!("  Encode time: {:?}", encode_time);
            println!("  Decode time: {:?}", decode_time);
            println!("  Iterator time: {:?}", iter_time);
            println!("  Original: {} bytes", original_bytes);
            println!("  Elements: {}", encoded.len());
            println!("  Encode rate: {:.2} M int/s", encode_rate);
            println!("  Decode rate: {:.2} M int/s", decode_rate);
            println!("  Iterator rate: {:.2} M int/s", iter_rate);
            println!("  Iterator vs Decode: {:.2}x", iter_rate / decode_rate);

            // Random access benchmark for larger datasets
            if size >= 10_000 {
                benchmark_random_access(&data);
            }
        }
    }
}

fn benchmark_streamvbyte_u16() {
    println!("\n\nStreamVByte Benchmark (u16)");
    println!("===========================");

    let test_sizes = vec![10_000]; // vec![1_000, 10_000, 100_000, 1_000_000];
    let max_values = vec![256u16, 65_535u16];

    for &size in &test_sizes {
        for &max_val in &max_values {
            println!("\nTesting {} elements (u16), max value: {}", size, max_val);

            // Generate test data
            let data = generate_random_data_u16(size, max_val);

            // Encode benchmark
            let start = Instant::now();
            let encoded = StreamVByte::encode(&data);
            let encode_time = start.elapsed();

            // Decode benchmark
            let start = Instant::now();
            let decoded = encoded.decode();
            let decode_time = start.elapsed();

            // Verify correctness
            assert_eq!(
                data, decoded,
                "Decode failed for size {} max_val {}",
                size, max_val
            );

            // Calculate metrics
            let original_bytes = size * 2; // 2 bytes per u16
            let encode_rate = size as f64 / encode_time.as_secs_f64() / 1_000_000.0;
            let decode_rate = size as f64 / decode_time.as_secs_f64() / 1_000_000.0;

            // Print results
            println!("  Encode time: {:?}", encode_time);
            println!("  Decode time: {:?}", decode_time);
            println!("  Original: {} bytes", original_bytes);
            println!("  Elements: {}", encoded.len());
            println!("  Encode rate: {:.2} M int/s", encode_rate);
            println!("  Decode rate: {:.2} M int/s", decode_rate);

            // Random access benchmark for larger datasets
            if size >= 10_000 {
                benchmark_random_access_u16(&data);
            }
        }
    }
}

fn main() {
    benchmark_streamvbyte();
    benchmark_streamvbyte_u16();
}
