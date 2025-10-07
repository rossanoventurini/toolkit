use mem_dbg::{MemDbg, MemSize};
use serde::{Deserialize, Serialize};

/// StreamNibble encoding for u16 values.
///
/// This structure compresses sequences of u16 integers by encoding them using
/// StreamNibbles (4-bit units). Each control byte encodes information for 4 values
/// using 2 bits per value to indicate how many StreamNibbles are needed.
///
/// # Format
///
/// - Control bytes: Each control byte uses 2 bits per value to encode the number
///   of StreamNibbles needed (0-3 representing 1-4 StreamNibbles)
/// - Data: Packed StreamNibbles (4 bits each), stored 2 per byte
///
/// # Examples
///
/// ```rust
/// use toolkit::stream_nibble::StreamNibble;
///
/// let data = vec![1u16, 15, 255, 4095];
/// let encoded = StreamNibble::encode(&data);
/// let decoded = encoded.decode();
/// assert_eq!(decoded, data);
/// ```
#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq, MemSize, MemDbg)]
pub struct StreamNibble {
    control_bytes: Box<[u8]>,
    data: Box<[u8]>,
    size: usize,
}

impl StreamNibble {
    pub fn encode(data: &[u16]) -> Self {
        if data.is_empty() {
            return Self::default();
        }
        let size = data.len();
        
        let mut control_bytes = Vec::with_capacity(data.len().div_ceil(4));
        let mut encoded_data = Vec::with_capacity(data.len() * std::mem::size_of::<u16>());

        let mut odd = false;
        let mut next_byte = 0u8;
        for chunk in data.chunks(4) {
            let mut control_byte = 0u8;
            for (i, &value) in chunk.iter().enumerate() {
                let length = 1.max(((16 - value.leading_zeros()) as usize).div_ceil(4));
                control_byte <<= 2;
                control_byte |= (length - 1) as u8;
                for n in 0..length {
                    if odd {
                        next_byte |= (value >> (n * 4) & 0x0F) as u8;
                        encoded_data.push(next_byte);
                        next_byte = 0;
                    }
                    else {
                        next_byte |= ((value >> (n * 4) & 0x0F) as u8) << 4;
                    }
                    odd = !odd;
                }
            }
            if chunk.len() < 4 {
                control_byte <<= 2 * (4 - chunk.len());
            }
            control_bytes.push(control_byte);
        }
        if odd {
            encoded_data.push(next_byte);
        }

        dbg!(size, encoded_data.len(), control_bytes.len());

       Self {
            control_bytes: control_bytes.into_boxed_slice(),
            data: encoded_data.into_boxed_slice(),
            size,
        }
    }

    pub fn decode(&self) -> Vec<u16> {
        if self.size == 0 {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(self.size);
        let mut odd = false;

        let mut last_byte = self.data[0];
        let mut data_index = 1;
        for &control_byte in self.control_bytes.iter() {
            for i in 0..4 {
                if result.len() >= self.size {
                    break;
                }
                let length = ((control_byte >> (6 - i * 2)) & 0x03) as usize + 1;
                let mut current_value = 0;
                for i in 0..length {
                    if odd {
                        let byte = (last_byte & 0x0F) as u16;
                        current_value |= byte << (i * 4);
                        if data_index < self.data.len() {
                            last_byte = self.data[data_index];
                            data_index += 1;
                        }
                    } else {
                        let byte = (last_byte >> 4) as u16;
                        current_value |= byte << (i * 4);
                    }
                    odd = !odd;
                }
                result.push(current_value);
            }
        }
        result
    }

    /// Returns the number of integers currently stored in the `Nibble`.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Checks if the `Nibble` is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============ BASIC FUNCTIONALITY TESTS ============

    #[test]
    fn test_empty_sequence() {
        let data: Vec<u16> = vec![];
        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        assert_eq!(encoded.len(), 0);
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_single_element() {
        let data: Vec<u16> = vec![42];
        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        assert_eq!(encoded.len(), 1);
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_basic_sequence() {
        let data: Vec<u16> = vec![1, 2, 3, 4, 5, 15, 255, 4095];
        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        assert_eq!(encoded.len(), data.len());
        assert!(!encoded.is_empty());
    }

    // ============ SIZE VARIATION TESTS ============

    #[test]
    fn test_exact_multiples_of_4() {
        // Test sequences that are exact multiples of 4 elements (one control byte per 4 values)
        for &size in &[4, 8, 12, 16, 20, 100, 1000] {
            let data: Vec<u16> = (0..size).collect();
            let encoded = StreamNibble::encode(&data);
            let decoded = encoded.decode();

            assert_eq!(decoded, data, "Failed for size {}", size);
            assert_eq!(encoded.len(), data.len());
        }
    }

    #[test]
    fn test_non_multiples_of_4() {
        // Test sequences that are NOT multiples of 4 elements
        for &size in &[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 101, 1001] {
            let data: Vec<u16> = (0..size).collect();
            let encoded = StreamNibble::encode(&data);
            let decoded = encoded.decode();

            assert_eq!(decoded, data, "Failed for size {}", size);
            assert_eq!(encoded.len(), data.len());
        }
    }

    // ============ VALUE RANGE TESTS ============

    #[test]
    fn test_single_nibble_values() {
        // Test with values that fit in 1 StreamNibble (0-15)
        let data: Vec<u16> = (0..16).collect();
        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
    }

    #[test]
    fn test_two_nibble_values() {
        // Test with values that need 2 StreamNibbles (16-255)
        let data: Vec<u16> = (16..256).collect();
        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
    }

    #[test]
    fn test_three_nibble_values() {
        // Test with values that need 3 StreamNibbles (256-4095)
        let data: Vec<u16> = vec![256, 512, 1024, 2048, 4095];
        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
    }

    #[test]
    fn test_four_nibble_values() {
        // Test with values that need 4 StreamNibbles (4096-65535)
        let data: Vec<u16> = vec![4096, 8192, 16384, 32768, 65535];
        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
    }

    #[test]
    fn test_mixed_nibble_sizes() {
        // Test with a mix of 1, 2, 3, and 4 StreamNibble values
        let data: Vec<u16> = vec![
            // 1 StreamNibble values (0-15)
            0, 1, 7, 15,
            // 2 StreamNibble values (16-255)
            16, 100, 200, 255,
            // 3 StreamNibble values (256-4095)
            256, 1000, 2000, 4095,
            // 4 StreamNibble values (4096-65535)
            4096, 10000, 50000, 65535,
        ];
        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
    }

    // ============ BOUNDARY VALUE TESTS ============

    #[test]
    fn test_nibble_boundaries() {
        // Test specific boundary values for each StreamNibble length
        let data: Vec<u16> = vec![
            // 1-nibble boundaries
            0, 1, 14, 15,
            // 2-nibble boundaries
            16, 17, 254, 255,
            // 3-nibble boundaries
            256, 257, 4094, 4095,
            // 4-nibble boundaries
            4096, 4097, 65534, 65535,
        ];
        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
    }

    #[test]
    fn test_powers_of_two() {
        // Test all powers of 2 that fit in u16
        let data: Vec<u16> = (0..16).map(|i| 1u16 << i).collect();
        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
    }

    #[test]
    fn test_powers_of_two_minus_one() {
        // Test values like 2^n - 1 (all bits set in lower positions)
        let data: Vec<u16> = (1..16).map(|i| (1u16 << i) - 1).collect();
        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
    }

    // ============ PATTERN TESTS ============

    #[test]
    fn test_repeated_values() {
        // Test sequences with many repeated values
        let mut data: Vec<u16> = Vec::new();

        // Many small values (1 StreamNibble)
        for _ in 0..100 {
            data.push(7);
        }

        // Many medium values (2 StreamNibbles)
        for _ in 0..100 {
            data.push(123);
        }

        // Many large values (3 StreamNibbles)
        for _ in 0..100 {
            data.push(1234);
        }

        // Many max-range values (4 StreamNibbles)
        for _ in 0..100 {
            data.push(12345);
        }

        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        assert_eq!(encoded.len(), data.len());
    }

    #[test]
    fn test_ascending_sequence() {
        let data: Vec<u16> = (0..1000).collect();
        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        assert_eq!(encoded.len(), data.len());
    }

    #[test]
    fn test_descending_sequence() {
        let data: Vec<u16> = (0..1000).rev().collect();
        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        assert_eq!(encoded.len(), data.len());
    }

    #[test]
    fn test_alternating_small_large() {
        // Alternating between small and large values
        let mut data: Vec<u16> = Vec::new();
        for i in 0..1000 {
            if i % 2 == 0 {
                data.push((i % 16) as u16); // Small value (1 StreamNibble)
            } else {
                data.push(60000 + (i % 100) as u16); // Large value (4 StreamNibbles)
            }
        }

        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        assert_eq!(encoded.len(), data.len());
    }

    #[test]
    fn test_fibonacci_sequence() {
        // Test Fibonacci numbers (common in real data)
        let mut data: Vec<u16> = vec![0, 1];
        while data.len() < 25 {
            let next = data[data.len() - 1].saturating_add(data[data.len() - 2]);
            if next == data[data.len() - 1] {
                // Overflow detection
                break;
            }
            data.push(next);
        }

        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        assert_eq!(encoded.len(), data.len());
    }

    // ============ LARGE SEQUENCE TESTS ============

    #[test]
    fn test_large_sequences() {
        // Test with large sequences to ensure scalability
        for &size in &[10000, 50000, 100000] {
            let data: Vec<u16> = (0..size).map(|i| ((i * 17 + 13) % 65536) as u16).collect();
            let encoded = StreamNibble::encode(&data);
            let decoded = encoded.decode();

            assert_eq!(decoded, data, "Failed for large size {}", size);
            assert_eq!(encoded.len(), data.len());
        }
    }

    #[test]
    fn test_compression_effectiveness() {
        // Test that compression actually reduces size for appropriate data
        let small_values: Vec<u16> = (0..10000).map(|i| (i % 16) as u16).collect(); // All 1-nibble values
        let encoded = StreamNibble::encode(&small_values);
        let original_size = small_values.len() * 2; // 2 bytes per u16
        let compressed_size = encoded.control_bytes.len() + encoded.data.len();

        // Should achieve significant compression for small values
        assert!(compressed_size < original_size);
        assert!(compressed_size < original_size / 2); // At least 2x compression

        let decoded = encoded.decode();
        assert_eq!(decoded, small_values);
    }

    // ============ STRESS TESTS ============

    #[test]
    fn test_all_zeros() {
        let data: Vec<u16> = vec![0; 10000];
        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
    }

    #[test]
    fn test_all_max_values() {
        let data: Vec<u16> = vec![u16::MAX; 1000];
        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
    }

    #[test]
    fn test_random_like_sequence() {
        // Generate a pseudo-random sequence using a simple PRNG
        let mut data: Vec<u16> = Vec::new();
        let mut state = 12345u64; // Simple seed

        for _ in 0..10000 {
            // Simple linear congruential generator
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            data.push(((state >> 16) & 0xFFFF) as u16);
        }

        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        assert_eq!(encoded.len(), data.len());
    }

    // ============ CONTROL BYTE TESTS ============

    #[test]
    fn test_control_byte_patterns() {
        // Test all possible combinations of StreamNibble lengths within a single control byte
        let test_cases = vec![
            // All 1-nibble values
            vec![1, 2, 3, 4],
            // All 2-nibble values
            vec![16, 32, 64, 128],
            // All 3-nibble values
            vec![256, 512, 1024, 2048],
            // All 4-nibble values
            vec![4096, 8192, 16384, 32768],
            // Mixed: 1,2,3,4 StreamNibbles
            vec![15, 255, 4095, 65535],
        ];

        for data in test_cases {
            let encoded = StreamNibble::encode(&data);
            let decoded = encoded.decode();
            assert_eq!(decoded, data, "Failed for pattern: {:?}", data);
        }
    }

    #[test]
    fn test_partial_control_bytes() {
        // Test sequences that don't fill complete control bytes (not multiples of 4)
        let test_cases = vec![
            vec![1_u16],                  // 1 element
            vec![1, 2],                   // 2 elements
            vec![1, 2, 3],                // 3 elements
            vec![1, 2, 3, 4, 5],          // 5 elements (1 full + 1 partial control byte)
            vec![1, 2, 3, 4, 5, 6],       // 6 elements
            vec![1, 2, 3, 4, 5, 6, 7],    // 7 elements
        ];

        for data in test_cases {
            let encoded = StreamNibble::encode(&data);
            let decoded = encoded.decode();
            assert_eq!(decoded, data, "Failed for partial sequence: {:?}", data);
            assert_eq!(encoded.len(), data.len());
        }
    }

    // ============ EDGE CASE TESTS ============

    #[test]
    fn test_single_large_value() {
        let data: Vec<u16> = vec![u16::MAX];
        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        assert_eq!(encoded.len(), 1);
    }

    #[test]
    fn test_alternating_min_max() {
        let data: Vec<u16> = vec![0, u16::MAX, 0, u16::MAX, 0, u16::MAX];
        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
    }

    #[test]
    fn test_nibble_patterns() {
        // Test various StreamNibble patterns that might reveal encoding issues
        let data: Vec<u16> = vec![
            0x0000, // All zeros
            0x0001, // Single bit set
            0x000F, // Single StreamNibble all ones
            0x00F0, // Second StreamNibble all ones
            0x0F00, // Third StreamNibble all ones
            0xF000, // Fourth StreamNibble all ones
            0x5555, // Alternating bits
            0xAAAA, // Alternating bits (inverse)
            0x0F0F, // Alternating StreamNibbles
            0xF0F0, // Alternating StreamNibbles (inverse)
            0xFFFF, // All ones
        ];

        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
    }

    // ============ PERFORMANCE REGRESSION TESTS ============

    #[test]
    fn test_worst_case_compression() {
        // Create data that should compress poorly (all 4-nibble values)
        let data: Vec<u16> = (0..1000).map(|i| 60000 + i).collect();
        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        // Should still be correct even if compression is poor
        assert_eq!(encoded.len(), data.len());
    }

    #[test]
    fn test_best_case_compression() {
        // Create data that should compress very well (all small values)
        let data: Vec<u16> = vec![1; 10000]; // All identical small values
        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        assert_eq!(encoded.len(), data.len());
    }

    // ============ SPECIFIC StreamNibble ENCODING TESTS ============

    #[test]
    fn test_nibble_encoding_details() {
        // Test that StreamNibbles are correctly packed
        let data: Vec<u16> = vec![0x0F, 0x0F]; // Two values, each 1 StreamNibble
        let encoded = StreamNibble::encode(&data);
        
        // Two 1-nibble values should fit in 1 byte of data
        assert_eq!(encoded.data.len(), 1);
        assert_eq!(encoded.data[0], 0xFF); // Both StreamNibbles should be 0xF
        
        let decoded = encoded.decode();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_odd_number_of_nibbles() {
        // Test sequences that result in odd number of StreamNibbles
        let data: Vec<u16> = vec![0x0F]; // 1 value = 1 StreamNibble (odd)
        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();
        assert_eq!(decoded, data);

        let data: Vec<u16> = vec![0x0F, 0x0F, 0x0F]; // 3 values = 3 StreamNibbles (odd)
        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_even_number_of_nibbles() {
        // Test sequences that result in even number of StreamNibbles
        let data: Vec<u16> = vec![0x0F, 0x0F]; // 2 values = 2 StreamNibbles (even)
        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();
        assert_eq!(decoded, data);

        let data: Vec<u16> = vec![0x0F, 0x0F, 0x0F, 0x0F]; // 4 values = 4 StreamNibbles (even)
        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_nibble_order() {
        // Test that StreamNibbles are extracted in correct order (from LSB to MSB)
        let data: Vec<u16> = vec![0x1234]; // 4 StreamNibbles: 4, 3, 2, 1 (from LSB to MSB)
        let encoded = StreamNibble::encode(&data);
        let decoded = encoded.decode();
        assert_eq!(decoded, data);
    }
}
