use mem_dbg::*;
use serde::{Deserialize, Serialize};

// use crate::SVBEncodable;
use crate::stream_vbyte::{
    self,
    utils::{U32_LENGTHS, decode_less_than_four, decode_slice_aligned},
};

pub mod utils;

#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq, MemSize, MemDbg)]
pub struct StreamVByteRandomAccess {
    svb: StreamVByte,
    offsets: Box<[usize]>,
    block_size: usize, // must be multiple of 4
}

impl StreamVByteRandomAccess {
    pub fn new(input: &[u32], block_size: usize) -> Self {
        assert!(block_size % 4 == 0, "block_size must be multiple of 4");

        let svb = StreamVByte::encode(input);
        let mut offsets = Vec::with_capacity((input.len() + block_size - 1) / block_size);
        offsets.push(0);

        for chunk in svb.controls.chunks(block_size / 4) {
            let mut offset = *offsets.last().unwrap();
            for control_byte in chunk {
                for j in 0..4 {
                    let len = ((control_byte >> (6 - j * 2)) & 0b11) as usize + 1;
                    offset += len;
                }
            }

            offsets.push(offset);
        }

        Self {
            svb,
            offsets: offsets.into_boxed_slice(),
            block_size,
        }
    }

    /// Skip the given range and return the number of bytes skipped in the data section, reading only the control bytes.
    pub fn skip(&self, range: std::ops::Range<usize>) -> usize {
        let mut to_skip = range.end - range.start;

        let block_id = range.start / self.block_size;
        let mut offset_in_data = self.offsets[block_id];
        let mut controls_index = range.start / 4;

        // skip till a multiple of 4 using only control bytes
        offset_in_data += self.svb.controls[controls_index..controls_index + (to_skip / 4)]
            .iter()
            .map(|&control_byte| {
                controls_index += 1;
                to_skip -= 4;
                U32_LENGTHS[control_byte as usize] as usize
            })
            .sum::<usize>();

        // skip the remaining values (less than 4)
        for i in 0..to_skip {
            offset_in_data +=
                (((self.svb.controls[controls_index] >> (6 - i * 2)) & 0b11) + 1) as usize;
        }

        offset_in_data
    }

    pub fn get_range(&self, buffer: &mut [u32], range: std::ops::Range<usize>) {
        assert!(
            range.start < range.end && range.end <= self.svb.size,
            "Invalid range",
        );

        assert!(
            buffer.len() >= range.len(),
            "Output buffer is not large enough"
        );

        let block_id = range.start / self.block_size;

        // Skip the beginning of the block to reach the start of the range
        let mut offset_in_data = self.skip(block_id * self.block_size..range.start);

        let length = range.len().min(self.svb.size - range.start);

        // Handle the case where the range does not start at a multiple of 4
        let to_skip = range.start % 4;
        let mut cur_position = 0;
        if to_skip > 0 {
            // Decode the first (partial) control byte
            let mut control_byte = self.svb.controls[range.start / 4] << (to_skip * 2); // shift to ignore the already skipped values
            offset_in_data += decode_less_than_four(
                (4 - to_skip).min(length),
                control_byte,
                &self.svb.data[offset_in_data..],
                buffer,
            ); // min() is to deal with the very special case where both range.start and range.end are in the same control byte. In that case we decode only the required values from the first control byte.

            cur_position = 4 - to_skip;
            if cur_position >= length {
                return;
            }
        }
        let controls_range = range.start.div_ceil(4)..range.end / 4; // end without potentially partial last control byte

        if !controls_range.is_empty() {
            offset_in_data += decode_slice_aligned(
                &mut buffer[cur_position..],
                &self.svb.controls[controls_range.clone()],
                &self.svb.data[offset_in_data..],
            );
            cur_position += controls_range.len() * 4;
        }

        let left_in_last_control_byte = range.end % 4;
        if left_in_last_control_byte > 0 {
            let control_byte = self.svb.controls[controls_range.end];
            let _ = decode_less_than_four(
                left_in_last_control_byte,
                control_byte,
                &self.svb.data[offset_in_data..],
                &mut buffer[cur_position..],
            );
        }
    }
}

/// An implementation of Stream VByte encoding/decoding.
#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq, MemSize, MemDbg)]
pub struct StreamVByte {
    controls: Box<[u8]>,
    data: Box<[u8]>,
    size: usize,
}

impl StreamVByte {
    pub fn encode(data: &[u32]) -> Self {
        if data.is_empty() {
            return Self::default();
        }
        let mut controls = Vec::with_capacity((data.len() + 3) / 4);
        let mut encoded_data = Vec::with_capacity(data.len() * std::mem::size_of::<u32>());
        for chunk in data.chunks(4) {
            let mut control_byte: u8 = 0;
            for &n in chunk.iter() {
                let bytes: [u8; std::mem::size_of::<u32>()] = n.to_be_bytes();
                let length = 1.max(std::mem::size_of::<u32>() - n.leading_zeros() as usize / 8);

                control_byte <<= 2;
                control_byte |= (length - 1) as u8;
                encoded_data.extend_from_slice(&bytes[std::mem::size_of::<u32>() - length..]);
            }

            controls.push(control_byte);
        }

        let remaining = data.len() % 4;
        if remaining != 0 {
            let last_control_byte = controls.last_mut().unwrap();
            *last_control_byte <<= 2 * (4 - remaining);
        }

        // Iterator implementation decode 16 values at a time, for this reason we need to pad
        // with 3 control bytes set to 0 and 15 data bytes set to 0.
        // TODO: Pad with the minimum number of bytes, maybe there are better ways to estimate
        // the required padding
        controls.extend_from_slice(&[0u8; 3]);
        encoded_data.extend_from_slice(&[0u8; 15]);

        Self {
            controls: controls.into_boxed_slice(),
            data: encoded_data.into_boxed_slice(),
            size: data.len(),
        }
    }

    pub fn decode(&self) -> Vec<u32> {
        if self.size == 0 {
            return Vec::new();
        }
        let mut output = vec![0u32; self.size];
        let control_end = self.size / 4; // self.controls.len() cannot be used because of padding 

        let mut encoded_data_index =
            decode_slice_aligned(&mut output, &self.controls[..control_end], &self.data);

        let last_control_byte = if self.size % 4 == 0 {
            0
        } else {
            self.controls[control_end]
        };

        encoded_data_index += stream_vbyte::utils::decode_less_than_four(
            self.size % 4,
            last_control_byte,
            &self.data[encoded_data_index..],
            &mut output[control_end * 4..],
        );

        output
    }

    /// TODO: Make it faster with table lookup
    pub fn control_byte_length(control_byte: u8) -> usize {
        let mut length = 0;
        for i in 0..4 {
            length += ((control_byte >> (6 - i * 2)) & 0b11) as usize + 1;
        }

        assert_eq!(length, U32_LENGTHS[control_byte as usize] as usize);
        length
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn iter(&self) -> StreamVByteIter<'_> {
        StreamVByteIter::new(&self.controls, &self.data, self.size)
    }
}

/// Iterator for StreamVByte that decodes 4 values at a time and keeps them in a buffer.
const BUFFER_SIZE: usize = 16;
pub struct StreamVByteIter<'a> {
    controls: &'a [u8],
    data: &'a [u8],
    size: usize,
    control_index: usize,
    data_index: usize,
    buffer: [u32; BUFFER_SIZE],
    buffer_index: usize,
    position: usize,
}

impl<'a> StreamVByteIter<'a> {
    /// Creates a new iterator for the given `StreamVByte`.
    fn new(controls: &'a [u8], data: &'a [u8], size: usize) -> Self {
        Self {
            controls,
            data,
            size,
            control_index: 0,
            data_index: 0,
            buffer: [0; BUFFER_SIZE],
            buffer_index: BUFFER_SIZE, // Start with empty buffer (index points past last element)
            position: 0,
        }
    }

    /// Fills the buffer with up to BUFFER_SIZE decoded values.
    fn fill_buffer(&mut self) {
        if self.position >= self.size {
            return;
        }

        // self.buffer_index = 0;
        // let mut how_many_controls = BUFFER_SIZE / 4;

        // if self.position + BUFFER_SIZE >= self.size {
        //     how_many_controls = (self.size - self.position + 3) / 4;
        //     self.buffer_index = BUFFER_SIZE - ((self.size - self.position) + 3) / 4 * 4;
        // }

        // self.data_index += crate::stream_vbyte::utils::decode_slice_aligned(
        //     &mut self.buffer[self.buffer_index..],
        //     &self.controls[self.control_index..self.control_index + how_many_controls],
        //     &self.data[self.data_index..],
        // );
        unsafe {
            self.data_index += crate::stream_vbyte::utils::decode_16_aligned(
                &mut self.buffer,
                &self.controls.get_unchecked(self.control_index..),
                &self.data.get_unchecked(self.data_index..),
            );
        }

        self.buffer_index = 0;
        self.control_index += 4;
    }
}

impl<'a> Iterator for StreamVByteIter<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        // If buffer is empty or we've consumed all values in current buffer, fill it
        if self.buffer_index >= BUFFER_SIZE {
            self.fill_buffer();
        }

        // Return the next value from the buffer if available
        if self.position < self.size {
            let value = unsafe { *self.buffer.get_unchecked(self.buffer_index) };
            self.buffer_index += 1;
            self.position += 1;
            return Some(value);
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.size - self.position;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for StreamVByteIter<'a> {
    fn len(&self) -> usize {
        self.size - self.position
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============ BASIC FUNCTIONALITY TESTS ============

    #[test]
    fn test_empty_sequence() {
        let data: Vec<u32> = vec![];
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        assert_eq!(encoded.len(), 0);
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_single_element() {
        let data: Vec<u32> = vec![42];
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        assert_eq!(encoded.len(), 1);
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_basic_sequence() {
        let data: Vec<u32> = vec![1, 2, 3, 4, 5, 100, 1000, 10000];
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(encoded.controls.len(), (data.len() + 3) / 4 + 3); // +3 for padding
        assert_eq!(decoded, data);
        assert_eq!(encoded.len(), data.len());
        assert!(!encoded.is_empty());
    }

    // ============ SIZE VARIATION TESTS ============

    #[test]
    fn test_exact_multiples_of_4() {
        // Test sequences that are exact multiples of 4 elements
        for &size in &[4, 8, 12, 16, 20, 100, 1000] {
            let data: Vec<u32> = (0..size).collect();
            let encoded = StreamVByte::encode(&data);
            let decoded = encoded.decode();

            assert_eq!(decoded, data, "Failed for size {}", size);
            assert_eq!(encoded.len(), data.len());
        }
    }

    #[test]
    fn test_non_multiples_of_4() {
        // Test sequences that are NOT multiples of 4 elements
        for &size in &[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 101, 1001] {
            let data: Vec<u32> = (0..size).collect();
            let encoded = StreamVByte::encode(&data);
            let decoded = encoded.decode();

            assert_eq!(decoded, data, "Failed for size {}", size);
            assert_eq!(encoded.len(), data.len());
        }
    }

    // ============ VALUE RANGE TESTS ============

    #[test]
    fn test_small_values_1_byte() {
        // Test with values that fit in 1 byte (0-255)
        let data: Vec<u32> = (0..256).collect();
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
    }

    #[test]
    fn test_medium_values_2_bytes() {
        // Test with values that need 2 bytes (256-65535)
        let data: Vec<u32> = (256..1024).collect(); // 768 values needing 2 bytes each
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        // Each value should take exactly 2 bytes in data (plus padding)
        assert_eq!(encoded.data.len(), data.len() * 2 + 15);
    }

    #[test]
    fn test_large_values_3_bytes() {
        // Test with values that need 3 bytes (65536-16777215)
        let data: Vec<u32> = vec![65536, 100000, 1000000, 16777215];
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        // Each value should take exactly 3 bytes in data (plus padding)
        assert_eq!(encoded.data.len(), data.len() * 3 + 15);
    }

    #[test]
    fn test_max_values_4_bytes() {
        // Test with values that need 4 bytes (16777216-u32::MAX)
        let data: Vec<u32> = vec![16777216, u32::MAX - 1, u32::MAX];
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        // Each value should take exactly 4 bytes in data (plus padding)
        assert_eq!(encoded.data.len(), data.len() * 4 + 15);
    }

    #[test]
    fn test_mixed_value_sizes() {
        // Test with a mix of 1, 2, 3, and 4 byte values
        let data: Vec<u32> = vec![
            // 1 byte values
            0,
            1,
            127,
            255,
            // 2 byte values
            256,
            1000,
            32767,
            65535,
            // 3 byte values
            65536,
            100000,
            1000000,
            16777215,
            // 4 byte values
            16777216,
            2000000000,
            u32::MAX - 1,
            u32::MAX,
        ];
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        // Total data size should be: 4*1 + 4*2 + 4*3 + 4*4 = 40 bytes
        assert_eq!(encoded.data.len(), 40 + 15); // +15 for padding
    }

    // ============ BOUNDARY VALUE TESTS ============

    #[test]
    fn test_byte_boundaries() {
        // Test specific boundary values for each byte length
        let data: Vec<u32> = vec![
            // 1-byte boundaries
            0,
            1,
            127,
            128,
            254,
            255,
            // 2-byte boundaries
            256,
            257,
            32767,
            32768,
            65534,
            65535,
            // 3-byte boundaries
            65536,
            65537,
            8388607,
            8388608,
            16777214,
            16777215,
            // 4-byte boundaries
            16777216,
            16777217,
            u32::MAX - 1,
            u32::MAX,
        ];
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
    }

    #[test]
    fn test_powers_of_two() {
        // Test all powers of 2 that fit in u32
        let data: Vec<u32> = (0..32).map(|i| 1u32 << i).collect();
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
    }

    #[test]
    fn test_powers_of_two_minus_one() {
        // Test values like 2^n - 1 (all bits set in lower positions)
        let data: Vec<u32> = (1..32).map(|i| (1u32 << i) - 1).collect();
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
    }

    // ============ PATTERN TESTS ============

    #[test]
    fn test_repeated_values() {
        // Test sequences with many repeated values
        let mut data: Vec<u32> = Vec::new();

        // Many small values
        for _ in 0..100 {
            data.push(42);
        }

        // Many medium values
        for _ in 0..100 {
            data.push(12345);
        }

        // Many large values
        for _ in 0..100 {
            data.push(0x12345678);
        }

        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        assert_eq!(encoded.len(), data.len());
    }

    #[test]
    fn test_ascending_sequence() {
        let data: Vec<u32> = (0..1000).collect();
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        assert_eq!(encoded.len(), data.len());
    }

    #[test]
    fn test_descending_sequence() {
        let data: Vec<u32> = (0..1000).rev().collect();
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        assert_eq!(encoded.len(), data.len());
    }

    #[test]
    fn test_alternating_small_large() {
        // Alternating between small and large values
        let mut data: Vec<u32> = Vec::new();
        for i in 0..1000 {
            if i % 2 == 0 {
                data.push(i % 256); // Small value (1 byte)
            } else {
                data.push(0xFF000000 + (i % 256)); // Large value (4 bytes)
            }
        }

        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        assert_eq!(encoded.len(), data.len());
    }

    #[test]
    fn test_fibonacci_sequence() {
        // Test Fibonacci numbers (common in real data)
        let mut data: Vec<u32> = vec![0, 1];
        while data.len() < 50 {
            let next = data[data.len() - 1].saturating_add(data[data.len() - 2]);
            if next == data[data.len() - 1] {
                // Overflow detection
                break;
            }
            data.push(next);
        }

        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        assert_eq!(encoded.len(), data.len());
    }

    // ============ LARGE SEQUENCE TESTS ============

    #[test]
    fn test_large_sequences() {
        // Test with large sequences to ensure scalability
        for &size in &[10000, 50000, 100000] {
            let data: Vec<u32> = (0..size).map(|i| (i * 17 + 13) % 65536).collect();
            let encoded = StreamVByte::encode(&data);
            let decoded = encoded.decode();

            assert_eq!(decoded, data, "Failed for large size {}", size);
            assert_eq!(encoded.len(), data.len());
        }
    }

    #[test]
    fn test_compression_effectiveness() {
        // Test that compression actually reduces size for appropriate data
        let small_values: Vec<u32> = (0..10000).map(|i| i % 256).collect(); // All 1-byte values
        let encoded = StreamVByte::encode(&small_values);
        let original_size = small_values.len() * 4; // 4 bytes per u32
        let compressed_size = encoded.controls.len() + encoded.data.len();

        // Should achieve significant compression for small values
        assert!(compressed_size < original_size);
        assert!(compressed_size < original_size / 2); // At least 2x compression

        let decoded = encoded.decode();
        assert_eq!(decoded, small_values);
    }

    // ============ STRESS TESTS ============

    #[test]
    fn test_all_zeros() {
        let data: Vec<u32> = vec![0; 10000];
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
    }

    #[test]
    fn test_all_max_values() {
        let data: Vec<u32> = vec![u32::MAX; 1000];
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        // All max values should take maximum space (4 bytes each) + 15 for padding
        assert_eq!(encoded.data.len(), data.len() * 4 + 15);
    }

    #[test]
    fn test_random_like_sequence() {
        // Generate a pseudo-random sequence using a simple PRNG
        let mut data: Vec<u32> = Vec::new();
        let mut state = 12345u64; // Simple seed

        for _ in 0..10000 {
            // Simple linear congruential generator
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            data.push((state >> 16) as u32 & 0xFFFF); // Use upper bits, limit to 16-bit values
        }

        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        assert_eq!(encoded.len(), data.len());
    }

    // ============ CONTROL BYTE TESTS ============

    #[test]
    fn test_control_byte_patterns() {
        // Test all possible combinations of 1-4 byte lengths within a single control byte
        let test_cases = vec![
            // All 1-byte values (control = 0b00000000)
            vec![1, 2, 3, 4],
            // All 2-byte values (control = 0b01010101)
            vec![256, 512, 1024, 2048],
            // All 3-byte values (control = 0b10101010)
            vec![65536, 131072, 262144, 524288],
            // All 4-byte values (control = 0b11111111)
            vec![16777216, 33554432, 67108864, 134217728],
            // Mixed: 1,2,3,4 bytes (control = 0b11100100)
            vec![255, 65535, 16777215, u32::MAX],
        ];

        for data in test_cases {
            let encoded = StreamVByte::encode(&data);
            let decoded = encoded.decode();
            assert_eq!(decoded, data, "Failed for pattern: {:?}", data);
        }
    }

    #[test]
    fn test_partial_control_bytes() {
        // Test sequences that don't fill complete control bytes (not multiples of 4)
        let test_cases = vec![
            vec![1],                   // 1 element
            vec![1, 2],                // 2 elements
            vec![1, 2, 3],             // 3 elements
            vec![1, 2, 3, 4, 5],       // 5 elements (1 full + 1 partial control byte)
            vec![1, 2, 3, 4, 5, 6],    // 6 elements
            vec![1, 2, 3, 4, 5, 6, 7], // 7 elements
        ];

        for data in test_cases {
            let encoded = StreamVByte::encode(&data);
            let decoded = encoded.decode();
            assert_eq!(decoded, data, "Failed for partial sequence: {:?}", data);
            assert_eq!(encoded.len(), data.len());
        }
    }

    // ============ EDGE CASE TESTS ============

    #[test]
    fn test_single_large_value() {
        let data: Vec<u32> = vec![u32::MAX];
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        assert_eq!(encoded.len(), 1);
    }

    #[test]
    fn test_alternating_min_max() {
        let data: Vec<u32> = vec![0, u32::MAX, 0, u32::MAX, 0, u32::MAX];
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
    }

    #[test]
    fn test_bit_patterns() {
        // Test various bit patterns that might reveal encoding issues
        let data: Vec<u32> = vec![
            0x00000000, // All zeros
            0x00000001, // Single bit set
            0x80000000, // MSB set
            0x55555555, // Alternating bits
            0xAAAAAAAA, // Alternating bits (inverse)
            0x0F0F0F0F, // Nibble pattern
            0xF0F0F0F0, // Nibble pattern (inverse)
            0x00FF00FF, // Byte pattern
            0xFF00FF00, // Byte pattern (inverse)
            0xFFFFFFFF, // All ones
        ];

        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
    }

    // ============ PERFORMANCE REGRESSION TESTS ============

    #[test]
    fn test_worst_case_compression() {
        // Create data that should compress poorly (all 4-byte values)
        let data: Vec<u32> = (0..1000).map(|i| 0xFF000000 + i).collect();
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        // Should still be correct even if compression is poor
        assert_eq!(encoded.len(), data.len());
    }

    #[test]
    fn test_best_case_compression() {
        // Create data that should compress very well (all small values)
        let data: Vec<u32> = vec![1; 10000]; // All identical small values
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(decoded, data);
        assert_eq!(encoded.len(), data.len());
    }

    // ============ ITERATOR TESTS ============

    #[test]
    fn test_iterator_empty() {
        let data: Vec<u32> = vec![];
        let encoded = StreamVByte::encode(&data);
        let iter_result: Vec<u32> = encoded.iter().collect();

        assert_eq!(iter_result, data);
        assert_eq!(encoded.iter().len(), 0);
        assert_eq!(encoded.iter().size_hint(), (0, Some(0)));
    }

    #[test]
    fn test_iterator_single_element() {
        let data: Vec<u32> = vec![42];
        let encoded = StreamVByte::encode(&data);
        let iter_result: Vec<u32> = encoded.iter().collect();

        assert_eq!(iter_result, data);
        assert_eq!(encoded.iter().len(), 1);
        assert_eq!(encoded.iter().size_hint(), (1, Some(1)));
    }

    #[test]
    fn test_iterator_basic_sequence() {
        let data: Vec<u32> = vec![1, 2, 3, 4, 5, 100, 1000, 10000];
        let encoded = StreamVByte::encode(&data);
        let iter_result: Vec<u32> = encoded.iter().collect();

        assert_eq!(iter_result, data);
        assert_eq!(encoded.iter().len(), data.len());
        assert_eq!(encoded.iter().size_hint(), (data.len(), Some(data.len())));
    }

    #[test]
    fn test_iterator_vs_decode() {
        // Test that iterator produces same results as decode()
        let data: Vec<u32> = (0..1000).map(|i| (i * 17 + 13) % 65536).collect();
        let encoded = StreamVByte::encode(&data);

        let decoded = encoded.decode();
        let iter_result: Vec<u32> = encoded.iter().collect();

        assert_eq!(decoded, data);
        assert_eq!(iter_result, data);
        assert_eq!(decoded, iter_result);
    }

    #[test]
    fn test_iterator_exact_multiples_of_16() {
        // Test sequences that are exact multiples of buffer size (16)
        for &size in &[16, 32, 48, 64, 160, 1600] {
            let data: Vec<u32> = (0..size).collect();
            let encoded = StreamVByte::encode(&data);
            let iter_result: Vec<u32> = encoded.iter().collect();

            assert_eq!(iter_result, data, "Failed for size {}", size);
        }
    }

    #[test]
    fn test_iterator_non_multiples_of_16() {
        // Test sequences that are NOT multiples of buffer size
        for &size in &[1, 7, 15, 17, 31, 33, 47, 49, 63, 65, 159, 161] {
            let data: Vec<u32> = (0..size).collect();
            let encoded = StreamVByte::encode(&data);
            let iter_result: Vec<u32> = encoded.iter().collect();

            assert_eq!(iter_result, data, "Failed for size {}", size);
        }
    }

    #[test]
    fn test_iterator_mixed_value_sizes() {
        let data: Vec<u32> = vec![
            // 1 byte values
            0,
            1,
            127,
            255,
            // 2 byte values
            256,
            1000,
            32767,
            65535,
            // 3 byte values
            65536,
            100000,
            1000000,
            16777215,
            // 4 byte values
            16777216,
            2000000000,
            u32::MAX - 1,
            u32::MAX,
        ];
        let encoded = StreamVByte::encode(&data);
        let iter_result: Vec<u32> = encoded.iter().collect();

        assert_eq!(iter_result, data);
    }

    #[test]
    fn test_iterator_large_sequence() {
        let data: Vec<u32> = (0..10000).map(|i| (i * 17 + 13) % 65536).collect();
        let encoded = StreamVByte::encode(&data);
        let iter_result: Vec<u32> = encoded.iter().collect();

        assert_eq!(iter_result, data);
    }

    #[test]
    fn test_iterator_size_hint_during_iteration() {
        let data: Vec<u32> = (0..100).collect();
        let encoded = StreamVByte::encode(&data);
        let mut iter = encoded.iter();

        // Check size_hint at different points during iteration
        assert_eq!(iter.size_hint(), (100, Some(100)));

        // Consume some elements
        for _ in 0..10 {
            iter.next();
        }
        assert_eq!(iter.size_hint(), (90, Some(90)));

        // Consume more elements
        for _ in 0..50 {
            iter.next();
        }
        assert_eq!(iter.size_hint(), (40, Some(40)));

        // Consume remaining elements
        let remaining: Vec<u32> = iter.collect();
        assert_eq!(remaining.len(), 40);
        assert_eq!(remaining, (60..100).collect::<Vec<u32>>());
    }

    #[test]
    fn test_iterator_exact_size() {
        let data: Vec<u32> = (0..173).collect(); // Non-multiple of 16
        let encoded = StreamVByte::encode(&data);
        let mut iter = encoded.iter();

        assert_eq!(iter.len(), 173);

        // Consume some elements and check len updates
        for _ in 0..50 {
            iter.next();
        }
        assert_eq!(iter.len(), 123);

        // Consume all remaining
        let remaining: Vec<u32> = iter.collect();
        assert_eq!(remaining.len(), 123);
    }

    #[test]
    fn test_iterator_random_like_sequence() {
        let mut data: Vec<u32> = Vec::new();
        let mut state = 12345u64;

        for _ in 0..5000 {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            data.push((state >> 16) as u32 & 0xFFFF);
        }

        let encoded = StreamVByte::encode(&data);
        let iter_result: Vec<u32> = encoded.iter().collect();

        assert_eq!(iter_result, data);
    }

    // ============ RANDOM ACCESS TESTS ============

    #[test]
    fn test_random_access_basic() {
        let data: Vec<u32> = (0..100).collect();
        let ra = StreamVByteRandomAccess::new(&data, 16);

        // Test single range
        let mut buffer = vec![0u32; 10];
        ra.get_range(&mut buffer, 10..20);
        assert_eq!(buffer, (10..20).collect::<Vec<u32>>());
    }

    #[test]
    fn test_random_access_different_block_sizes() {
        let data: Vec<u32> = (0..1000).collect();

        for &block_size in &[4, 8, 16, 32, 64, 128] {
            let ra = StreamVByteRandomAccess::new(&data, block_size);

            // Test multiple ranges
            let test_ranges = vec![0..10, 50..75, 100..150, 500..600, 900..950, 990..1000];

            for range in test_ranges {
                let mut buffer = vec![0u32; range.len()];
                ra.get_range(&mut buffer, range.clone());
                assert_eq!(
                    buffer,
                    range.map(|x| x as u32).collect::<Vec<u32>>(),
                    "Failed for block_size {}",
                    block_size
                );
            }
        }
    }

    #[test]
    fn test_random_access_edge_cases() {
        let data: Vec<u32> = (0..100).collect();
        let ra = StreamVByteRandomAccess::new(&data, 16);

        // Test single element
        let mut buffer = vec![0u32; 1];
        ra.get_range(&mut buffer, 42..43);
        assert_eq!(buffer, vec![42]);

        // Test at beginning
        let mut buffer = vec![0u32; 5];
        ra.get_range(&mut buffer, 0..5);
        assert_eq!(buffer, vec![0, 1, 2, 3, 4]);

        // Test at end
        let mut buffer = vec![0u32; 5];
        ra.get_range(&mut buffer, 95..100);
        assert_eq!(buffer, vec![95, 96, 97, 98, 99]);
    }

    #[test]
    fn test_random_access_mixed_value_sizes() {
        let data: Vec<u32> = vec![
            // 1 byte values
            0,
            1,
            127,
            255,
            // 2 byte values
            256,
            1000,
            32767,
            65535,
            // 3 byte values
            65536,
            100000,
            1000000,
            16777215,
            // 4 byte values
            16777216,
            2000000000,
            u32::MAX - 1,
            u32::MAX,
        ];
        let ra = StreamVByteRandomAccess::new(&data, 8);

        // Test various ranges
        let test_cases = vec![
            (0..4, vec![0, 1, 127, 255]),
            (4..8, vec![256, 1000, 32767, 65535]),
            (8..12, vec![65536, 100000, 1000000, 16777215]),
            (12..16, vec![16777216, 2000000000, u32::MAX - 1, u32::MAX]),
            (2..6, vec![127, 255, 256, 1000]),
        ];

        for (range, expected) in test_cases {
            let mut buffer = vec![0u32; range.len()];
            ra.get_range(&mut buffer, range.clone());
            assert_eq!(buffer, expected, "Failed for range {:?}", range);
        }
    }

    #[test]
    fn test_random_access_vs_full_decode() {
        let mut data: Vec<u32> = Vec::new();
        let mut state = 12345u64;

        for _ in 0..1000 {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            data.push((state >> 16) as u32 & 0xFFFF);
        }

        let ra = StreamVByteRandomAccess::new(&data, 64);

        // Test multiple random ranges
        let test_ranges = vec![0..50, 100..200, 300..350, 500..750, 800..900, 950..1000];

        for range in test_ranges {
            let mut buffer = vec![0u32; range.len()];
            ra.get_range(&mut buffer, range.clone());

            let expected = &data[range.clone()];
            assert_eq!(buffer, expected, "Failed for range {:?}", range);
        }
    }

    #[test]
    fn test_random_access_large_ranges() {
        let data: Vec<u32> = (0..10000).map(|i| (i * 17 + 13) % 65536).collect();
        let ra = StreamVByteRandomAccess::new(&data, 128);

        // Test large ranges
        let test_ranges = vec![0..1000, 2000..3500, 5000..7500, 8000..10000];

        for range in test_ranges {
            let mut buffer = vec![0u32; range.len()];
            ra.get_range(&mut buffer, range.clone());

            let expected = &data[range.clone()];
            assert_eq!(buffer, expected, "Failed for range {:?}", range);
        }
    }

    #[test]
    #[should_panic(expected = "Invalid range")]
    fn test_random_access_invalid_range_start_after_end() {
        let data: Vec<u32> = (0..100).collect();
        let ra = StreamVByteRandomAccess::new(&data, 16);
        let mut buffer = vec![0u32; 10];
        ra.get_range(&mut buffer, 50..40); // start > end
    }

    #[test]
    #[should_panic(expected = "Invalid range")]
    fn test_random_access_invalid_range_out_of_bounds() {
        let data: Vec<u32> = (0..100).collect();
        let ra = StreamVByteRandomAccess::new(&data, 16);
        let mut buffer = vec![0u32; 10];
        ra.get_range(&mut buffer, 90..110); // end > data.len()
    }

    #[test]
    #[should_panic(expected = "Output buffer is not large enough")]
    fn test_random_access_buffer_too_small() {
        let data: Vec<u32> = (0..100).collect();
        let ra = StreamVByteRandomAccess::new(&data, 16);
        let mut buffer = vec![0u32; 5]; // buffer too small for range
        ra.get_range(&mut buffer, 10..20); // range needs 10 elements but buffer has 5
    }
}
