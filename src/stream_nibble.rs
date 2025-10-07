use mem_dbg::{MemDbg, MemSize};
use serde::{Deserialize, Serialize};

/// Random access wrapper for StreamNibble encoded data.
///
/// This structure enables efficient random access to compressed data by maintaining
/// a block-based index. It divides the encoded sequence into blocks and stores offsets
/// to quickly locate any range without full decompression.
///
/// # Fields
///
/// * `nibble` - The underlying StreamNibble encoded data
/// * `offsets` - Nibble offsets for each block in the data section
/// * `block_size` - Number of values per block (must be multiple of 4)
///
/// # Examples
///
/// ```rust
/// use toolkit::stream_nibble::StreamNibbleRandomAccess;
///
/// let data: Vec<u16> = (0..1000).map(|x| x as u16).collect();
/// let ra = StreamNibbleRandomAccess::new(&data, 64);
///
/// let mut buffer = vec![0u16; 10];
/// ra.get_range(&mut buffer, 500..510);
/// assert_eq!(buffer, (500..510).map(|x| x as u16).collect::<Vec<u16>>());
/// ```
#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq, MemSize, MemDbg)]
pub struct StreamNibbleRandomAccess {
    nibble: StreamNibble,
    offsets: Box<[usize]>, // Nibble offsets (not byte offsets)
    block_size: usize,     // must be multiple of 4
}

impl StreamNibbleRandomAccess {
    /// Creates a new random access structure from the input data.
    ///
    /// The data is first encoded using StreamNibble compression, then divided into blocks
    /// of the specified size. An index of nibble offsets is built to enable
    /// efficient random access to any block.
    ///
    /// # Arguments
    ///
    /// * `input` - The slice of u16 values to encode
    /// * `block_size` - Number of values per block (must be multiple of 4)
    ///
    /// # Panics
    ///
    /// Panics if `block_size` is not a multiple of 4.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use toolkit::stream_nibble::StreamNibbleRandomAccess;
    ///
    /// let data: Vec<u16> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    /// let ra = StreamNibbleRandomAccess::new(&data, 4); // 4 values per block
    /// ```
    pub fn new(input: &[u16], block_size: usize) -> Self {
        assert!(block_size % 4 == 0, "block_size must be multiple of 4");

        let nibble = StreamNibble::encode(input);
        let mut offsets = Vec::with_capacity((input.len() + block_size - 1) / block_size);
        offsets.push(0);

        // Calculate nibble offsets for each block
        for chunk in nibble.control_bytes.chunks(block_size / 4) {
            let mut offset = *offsets.last().unwrap();
            offset += chunk
                .iter()
                .map(|&control_byte| {
                    // Each control byte encodes 4 values with 2 bits each
                    // Extract the number of nibbles for each value
                    let mut nibble_count = 0;
                    for i in 0..4 {
                        let length = ((control_byte >> (6 - i * 2)) & 0x03) as usize + 1;
                        nibble_count += length;
                    }
                    nibble_count
                })
                .sum::<usize>();

            offsets.push(offset);
        }

        Self {
            nibble,
            offsets: offsets.into_boxed_slice(),
            block_size,
        }
    }

    /// Skips the given range and returns the nibble offset in the data section.
    ///
    /// This method efficiently calculates how many nibbles to skip in the compressed
    /// data to reach a specific position, using only the control bytes for the
    /// calculation.
    ///
    /// # Arguments
    ///
    /// * `range` - The range of values to skip
    ///
    /// # Returns
    ///
    /// The nibble offset in the data section after skipping the range.
    pub fn skip(&self, range: std::ops::Range<usize>) -> usize {
        let mut to_skip = range.end - range.start;

        let block_id = range.start / self.block_size;
        let mut nibble_offset = self.offsets[block_id];
        let mut control_bytes_index = range.start / 4;

        // Skip complete control bytes
        nibble_offset += self.nibble.control_bytes
            [control_bytes_index..control_bytes_index + (to_skip / 4)]
            .iter()
            .map(|&control_byte| {
                control_bytes_index += 1;
                to_skip -= 4;
                // Count nibbles in this control byte
                let mut nibble_count = 0;
                for i in 0..4 {
                    let length = ((control_byte >> (6 - i * 2)) & 0x03) as usize + 1;
                    nibble_count += length;
                }
                nibble_count
            })
            .sum::<usize>();

        // Skip the remaining values (less than 4)
        if to_skip > 0 {
            let control_byte = self.nibble.control_bytes[control_bytes_index];
            for i in 0..to_skip {
                let length = ((control_byte >> (6 - i * 2)) & 0x03) as usize + 1;
                nibble_offset += length;
            }
        }

        nibble_offset
    }

    /// Decodes a specific range of values into the provided buffer.
    ///
    /// This is the main random access operation. It efficiently decodes only
    /// the requested range without decompressing the entire sequence.
    ///
    /// # Arguments
    ///
    /// * `buffer` - Output buffer to write decoded values (must be large enough)
    /// * `range` - The range of indices to decode
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `range.start >= range.end`
    /// - `range.end > self.nibble.size`
    /// - `buffer.len() < range.len()`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use toolkit::stream_nibble::StreamNibbleRandomAccess;
    ///
    /// let data: Vec<u16> = (0..100).map(|x| x as u16).collect();
    /// let ra = StreamNibbleRandomAccess::new(&data, 16);
    ///
    /// let mut buffer = vec![0u16; 20];
    /// ra.get_range(&mut buffer, 40..60);
    /// assert_eq!(buffer, (40..60).map(|x| x as u16).collect::<Vec<u16>>());
    /// ```
    pub fn get_range(&self, buffer: &mut [u16], range: std::ops::Range<usize>) {
        assert!(
            range.start < range.end && range.end <= self.nibble.size,
            "Invalid range",
        );

        assert!(
            buffer.len() >= range.len(),
            "Output buffer is not large enough"
        );

        let block_id = range.start / self.block_size;

        // Skip to the start of the range
        let nibble_offset = self.skip(block_id * self.block_size..range.start);

        // Decode the range using the trivial decompression approach
        self.decode_range(nibble_offset, range.clone(), buffer);
    }

    /// Decodes a range of values starting from a given nibble offset.
    ///
    /// This is a helper method that performs the actual decoding.
    ///
    /// # Arguments
    ///
    /// * `nibble_offset` - Starting nibble offset in the data
    /// * `range` - The range of values to decode
    /// * `buffer` - Output buffer for decoded values
    fn decode_range(
        &self,
        mut nibble_offset: usize,
        range: std::ops::Range<usize>,
        buffer: &mut [u16],
    ) {
        let mut control_byte_index = range.start / 4;
        let mut value_index_in_control_byte = range.start % 4;

        // Helper to extract nibble at given offset
        let get_nibble = |nibble_offset: usize| -> u8 {
            let byte_index = nibble_offset / 2;
            if nibble_offset % 2 == 0 {
                // High nibble
                (self.nibble.data[byte_index] >> 4) & 0x0F
            } else {
                // Low nibble
                self.nibble.data[byte_index] & 0x0F
            }
        };

        for i in 0..range.len() {
            let control_byte = self.nibble.control_bytes[control_byte_index];
            let length =
                ((control_byte >> (6 - value_index_in_control_byte * 2)) & 0x03) as usize + 1;

            let mut value = 0u16;
            for j in 0..length {
                let nibble = get_nibble(nibble_offset) as u16;
                value |= nibble << (j * 4);
                nibble_offset += 1;
            }

            buffer[i] = value;

            value_index_in_control_byte += 1;
            if value_index_in_control_byte >= 4 {
                value_index_in_control_byte = 0;
                control_byte_index += 1;
            }
        }
    }
}

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
            for (_i, &value) in chunk.iter().enumerate() {
                let length = 1.max(((16 - value.leading_zeros()) as usize).div_ceil(4));
                control_byte <<= 2;
                control_byte |= (length - 1) as u8;
                for n in 0..length {
                    if odd {
                        next_byte |= (value >> (n * 4) & 0x0F) as u8;
                        encoded_data.push(next_byte);
                        next_byte = 0;
                    } else {
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
            0, 1, 7, 15, // 2 StreamNibble values (16-255)
            16, 100, 200, 255, // 3 StreamNibble values (256-4095)
            256, 1000, 2000, 4095, // 4 StreamNibble values (4096-65535)
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
            0, 1, 14, 15, // 2-nibble boundaries
            16, 17, 254, 255, // 3-nibble boundaries
            256, 257, 4094, 4095, // 4-nibble boundaries
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
            vec![1_u16],               // 1 element
            vec![1, 2],                // 2 elements
            vec![1, 2, 3],             // 3 elements
            vec![1, 2, 3, 4, 5],       // 5 elements (1 full + 1 partial control byte)
            vec![1, 2, 3, 4, 5, 6],    // 6 elements
            vec![1, 2, 3, 4, 5, 6, 7], // 7 elements
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

    // ============ RANDOM ACCESS TESTS ============

    #[test]
    fn test_random_access_basic() {
        let data: Vec<u16> = (0..100).map(|x| x as u16).collect();
        let ra = StreamNibbleRandomAccess::new(&data, 16);

        // Test single range
        let mut buffer = vec![0u16; 10];
        ra.get_range(&mut buffer, 10..20);
        assert_eq!(buffer, (10..20).map(|x| x as u16).collect::<Vec<u16>>());
    }

    #[test]
    fn test_random_access_different_block_sizes() {
        let data: Vec<u16> = (0..1000).map(|x| x as u16).collect();

        for &block_size in &[4, 8, 16, 32, 64, 128] {
            let ra = StreamNibbleRandomAccess::new(&data, block_size);

            // Test multiple ranges
            let test_ranges = vec![0..10, 50..75, 100..150, 500..600, 900..950, 990..1000];

            for range in test_ranges {
                let mut buffer = vec![0u16; range.len()];
                ra.get_range(&mut buffer, range.clone());
                assert_eq!(
                    buffer,
                    range.map(|x| x as u16).collect::<Vec<u16>>(),
                    "Failed for block_size {}",
                    block_size
                );
            }
        }
    }

    #[test]
    fn test_random_access_edge_cases() {
        let data: Vec<u16> = (0..100).map(|x| x as u16).collect();
        let ra = StreamNibbleRandomAccess::new(&data, 16);

        // Test single element
        let mut buffer = vec![0u16; 1];
        ra.get_range(&mut buffer, 42..43);
        assert_eq!(buffer, vec![42]);

        // Test at beginning
        let mut buffer = vec![0u16; 5];
        ra.get_range(&mut buffer, 0..5);
        assert_eq!(buffer, vec![0, 1, 2, 3, 4]);

        // Test at end
        let mut buffer = vec![0u16; 5];
        ra.get_range(&mut buffer, 95..100);
        assert_eq!(buffer, vec![95, 96, 97, 98, 99]);
    }

    #[test]
    fn test_random_access_mixed_value_sizes() {
        let data: Vec<u16> = vec![
            // 1 nibble values
            0, 1, 7, 15, // 2 nibble values
            16, 100, 200, 255, // 3 nibble values
            256, 1000, 2000, 4095, // 4 nibble values
            4096, 10000, 50000, 65535,
        ];
        let ra = StreamNibbleRandomAccess::new(&data, 8);

        // Test various ranges
        let test_cases = vec![
            (0..4, vec![0u16, 1, 7, 15]),
            (4..8, vec![16u16, 100, 200, 255]),
            (8..12, vec![256u16, 1000, 2000, 4095]),
            (12..16, vec![4096u16, 10000, 50000, 65535]),
            (2..6, vec![7u16, 15, 16, 100]),
        ];

        for (range, expected) in test_cases {
            let mut buffer = vec![0u16; range.len()];
            ra.get_range(&mut buffer, range.clone());
            assert_eq!(buffer, expected, "Failed for range {:?}", range);
        }
    }

    #[test]
    fn test_random_access_vs_full_decode() {
        let mut data: Vec<u16> = Vec::new();
        let mut state = 12345u64;

        for _ in 0..1000 {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            data.push(((state >> 16) & 0xFFFF) as u16);
        }

        let ra = StreamNibbleRandomAccess::new(&data, 64);

        // Test multiple random ranges
        let test_ranges = vec![0..50, 100..200, 300..350, 500..750, 800..900, 950..1000];

        for range in test_ranges {
            let mut buffer = vec![0u16; range.len()];
            ra.get_range(&mut buffer, range.clone());

            let expected = &data[range.clone()];
            assert_eq!(buffer, expected, "Failed for range {:?}", range);
        }
    }

    #[test]
    fn test_random_access_large_ranges() {
        let data: Vec<u16> = (0..10000).map(|i| ((i * 17 + 13) % 65536) as u16).collect();
        let ra = StreamNibbleRandomAccess::new(&data, 128);

        // Test large ranges
        let test_ranges = vec![0..1000, 2000..3500, 5000..7500, 8000..10000];

        for range in test_ranges {
            let mut buffer = vec![0u16; range.len()];
            ra.get_range(&mut buffer, range.clone());

            let expected = &data[range.clone()];
            assert_eq!(buffer, expected, "Failed for range {:?}", range);
        }
    }

    #[test]
    #[should_panic(expected = "Invalid range")]
    fn test_random_access_invalid_range_start_after_end() {
        let data: Vec<u16> = (0..100).map(|x| x as u16).collect();
        let ra = StreamNibbleRandomAccess::new(&data, 16);
        let mut buffer = vec![0u16; 10];
        ra.get_range(&mut buffer, 50..40); // start > end
    }

    #[test]
    #[should_panic(expected = "Invalid range")]
    fn test_random_access_invalid_range_out_of_bounds() {
        let data: Vec<u16> = (0..100).map(|x| x as u16).collect();
        let ra = StreamNibbleRandomAccess::new(&data, 16);
        let mut buffer = vec![0u16; 10];
        ra.get_range(&mut buffer, 90..110); // end > data.len()
    }

    #[test]
    #[should_panic(expected = "Output buffer is not large enough")]
    fn test_random_access_buffer_too_small() {
        let data: Vec<u16> = (0..100).map(|x| x as u16).collect();
        let ra = StreamNibbleRandomAccess::new(&data, 16);
        let mut buffer = vec![0u16; 5]; // buffer too small for range
        ra.get_range(&mut buffer, 10..20); // range needs 10 elements but buffer has 5
    }
}
