use mem_dbg::*;
use serde::{Deserialize, Serialize};

/// Code here is heavily inspired by [svbyte](https://github.com/bazhenov/svbyte/) and [stream_vbyte_rust](https://bitbucket.org/marshallpierce/stream-vbyte-rust) crates.
use crate::SVBEncodable;
use num::traits::{FromBytes, ToBytes, ops::bytes::NumBytes};

#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq, MemSize, MemDbg)]
pub struct StreamVByteRandomAccess<T: SVBEncodable> {
    svb: StreamVByte<T>,
    offsets: Box<[usize]>,
    block_size: usize, // must be multiple of 4
}

impl<T: SVBEncodable> StreamVByteRandomAccess<T> {
    pub fn new(input: &[T], block_size: usize) -> Self {
        assert!(block_size % 4 == 0, "block_size must be multiple of 4");

        let svb = StreamVByte::encode(input);
        let mut offsets = Vec::with_capacity((input.len() + block_size - 1) / block_size);
        offsets.push(0);

        for chunk in svb.controls.chunks(block_size / 4) {
            let mut offset = *offsets.last().unwrap();
            for control_byte in chunk {
                for j in 0..4 {
                    let len = ((control_byte >> (j * 2)) & 0b11) as usize + 1;
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
}

impl<T> StreamVByteRandomAccess<T>
where
    T: SVBEncodable,
    <T as FromBytes>::Bytes: NumBytes + Default + AsMut<[u8]> + AsRef<[u8]> + Sized,
{
    /// Returns an iterator over a portion of the encoded sequence.
    ///
    /// # Arguments
    /// * `position` - The starting position in the original sequence
    /// * `length` - The number of elements to iterate over
    ///
    /// # Examples
    /// ```
    /// use toolkit::{StreamVByte, StreamVByteRandomAccess};
    ///
    /// let data: Vec<u32> = (0..100).collect();
    /// let svb_ra = StreamVByteRandomAccess::new(&data, 16);
    ///
    /// // Iterate over elements 10-19
    /// let portion: Vec<u32> = svb_ra.iter_range(10, 10).collect();
    /// assert_eq!(portion, (10..20).collect::<Vec<u32>>());
    /// ```
    pub fn iter_range(&self, position: usize, length: usize) -> StreamVByteIter<'_, T> {
        if position >= self.svb.size || length == 0 {
            return StreamVByteIter::new(&[], &[], 0);
        }

        let block_id = position / self.block_size;

        let length = length.min(self.svb.size - position);

        let mut offset_in_data = self.offsets[block_id];
        let mut starting_pos = block_id * self.block_size;

        let mut to_skip = position - starting_pos;
        let mut controls_index = starting_pos / 4;

        while to_skip >= 4 {
            let control_byte = self.svb.controls[controls_index];
            for j in 0..4 {
                let len = ((control_byte >> (j * 2)) & 0b11) as usize + 1;
                offset_in_data += len;
            }
            to_skip -= 4;
            controls_index += 1;
            starting_pos += 4;
        }

        let end = position + length - starting_pos;

        let controls = &self.svb.controls[controls_index..];
        let data = &self.svb.data[offset_in_data..];

        let mut iter = StreamVByteIter::new(controls, data, end);
        for _ in 0..to_skip {
            let _ = iter.next();
        }
        iter
    }
}

#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq, MemSize, MemDbg)]
pub struct StreamVByte<T: SVBEncodable> {
    controls: Box<[u8]>,
    data: Box<[u8]>,
    size: usize,
    phantom: std::marker::PhantomData<T>,
}

impl<T: SVBEncodable> StreamVByte<T> {
    /// Encode a stream of integers using Stream VByte encoding.
    pub fn encode(input: &[T]) -> Self {
        let mut controls = Vec::with_capacity((input.len() + 3) / 4);
        let mut encoded_data = Vec::with_capacity(input.len()); // at least 1 byte per integer

        for chunk in input.chunks(4) {
            Self::encode_upto_4_values(chunk, &mut controls, &mut encoded_data);
        }

        Self {
            controls: controls.into_boxed_slice(),
            data: encoded_data.into_boxed_slice(),
            size: input.len(),
            phantom: std::marker::PhantomData,
        }
    }

    #[inline]
    fn encode_upto_4_values(data: &[T], controls: &mut Vec<u8>, encoded_data: &mut Vec<u8>) {
        debug_assert!(data.len() <= 4);

        let mut control_byte = 0u8;
        for (i, &v) in data.iter().enumerate() {
            let len = Self::encode_value(v, encoded_data) as u8 - 1;

            control_byte |= len << (i * 2);
        }
        controls.push(control_byte);
    }

    #[inline]
    fn encode_value(num: T, vec: &mut Vec<u8>) -> usize {
        // this will calculate 0_u32 as taking 0 bytes, so ensure at least 1 byte
        let len = 1.max(std::mem::size_of::<T>() - num.leading_zeros() as usize / 8);
        let bytes = ToBytes::to_le_bytes(&num);

        for b in bytes.as_ref().iter().take(len) {
            vec.push(*b);
        }
        len
    }

    /// Returns the number of integers stored in the `StreamVByte`.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns `true` if the `StreamVByte` contains no integers.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
}

impl<T> StreamVByte<T>
where
    T: SVBEncodable,
    <T as FromBytes>::Bytes: NumBytes + Default + AsMut<[u8]> + AsRef<[u8]> + Sized,
{
    pub fn decode(&self) -> Vec<T> {
        let mut output = Vec::with_capacity(self.size);
        let mut data_index = 0;

        let mut controls_iter = self.controls.iter();

        for &control_byte in controls_iter.by_ref().take(self.size / 4) {
            self.decode_upto_4_values(control_byte, &mut data_index, &mut output, 4);
        }
        if let Some(&control_byte) = controls_iter.next() {
            self.decode_upto_4_values(control_byte, &mut data_index, &mut output, self.size % 4);
        }

        output
    }

    /// Returns an iterator over the values in the `StreamVByte`.
    pub fn iter(&self) -> StreamVByteIter<'_, T> {
        StreamVByteIter::new(&self.controls, &self.data, self.size)
    }

    #[inline]
    fn decode_upto_4_values(
        &self,
        control_byte: u8,
        data_index: &mut usize,
        output: &mut Vec<T>,
        count: usize,
    ) {
        debug_assert!(count <= 4);

        for i in 0..count {
            let len = ((control_byte >> (i * 2)) & 0b11) as usize + 1;

            let value = Self::decode_value(&self.data[*data_index..*data_index + len]);

            output.push(value);

            *data_index += len;
        }
    }

    #[inline]
    fn decode_value(slice: &[u8]) -> T {
        let mut buf = <T as FromBytes>::Bytes::default();
        let dst = buf.as_mut();

        dst[..slice.len()].copy_from_slice(&slice[..]);
        T::from_le_bytes(&buf)
    }
}

/// Iterator for StreamVByte that decodes 4 values at a time and keeps them in a buffer.
pub struct StreamVByteIter<'a, T: SVBEncodable> {
    controls: &'a [u8],
    data: &'a [u8],
    size: usize,
    control_index: usize,
    data_index: usize,
    buffer: [T; 4],
    buffer_index: usize,
    position: usize,
}

impl<'a, T> StreamVByteIter<'a, T>
where
    T: SVBEncodable,
    <T as FromBytes>::Bytes: NumBytes + Default + AsMut<[u8]> + AsRef<[u8]> + Sized,
{
    /// Creates a new iterator for the given `StreamVByte`.
    fn new(controls: &'a [u8], data: &'a [u8], size: usize) -> Self {
        Self {
            controls,
            data,
            size,
            control_index: 0,
            data_index: 0,
            buffer: [T::default(); 4],
            buffer_index: 4, // Start with empty buffer (index points past last element)
            position: 0,
        }
    }

    /// Fills the buffer with up to 4 decoded values.
    fn fill_buffer(&mut self) {
        if self.position >= self.size {
            return;
        }

        let control_byte = self.controls[self.control_index];
        let values_to_decode = std::cmp::min(4, self.size - self.position);

        // Decode up to 4 values
        for i in 0..values_to_decode {
            let len = ((control_byte >> (i * 2)) & 0b11) as usize + 1;

            let value =
                StreamVByte::decode_value(&self.data[self.data_index..self.data_index + len]);
            self.buffer[i] = value;
            self.data_index += len;
        }

        self.control_index += 1;
        self.buffer_index = 0;
    }
}

impl<'a, T> Iterator for StreamVByteIter<'a, T>
where
    T: SVBEncodable + Copy,
    <T as FromBytes>::Bytes: NumBytes + Default + AsMut<[u8]> + AsRef<[u8]> + Sized,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        // If buffer is empty or we've consumed all values in current buffer, fill it
        if self.buffer_index >= 4 {
            self.fill_buffer();
        }

        // Return the next value from the buffer if available
        if self.position < self.size {
            let value = self.buffer[self.buffer_index];
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

impl<'a, T> ExactSizeIterator for StreamVByteIter<'a, T>
where
    T: SVBEncodable + Copy,
    <T as FromBytes>::Bytes: NumBytes + Default + AsMut<[u8]> + AsRef<[u8]> + Sized,
{
    fn len(&self) -> usize {
        self.size - self.position
    }
}

mod utils;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_vbyte_u32_small() {
        let data: Vec<u32> = vec![1, 2, 3, 4, 5, 100, 1000, 10000];
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(data, decoded);
        assert_eq!(encoded.len(), data.len());
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_stream_vbyte_u32_large() {
        // Test with large u32 vector
        let data: Vec<u32> = (0..10000).map(|i| i * i + 100).collect();
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(data, decoded);
        assert_eq!(encoded.len(), data.len());

        // Verify that compression is effective for small values
        let original_size = data.len() * 4; // 4 bytes per u32
        let compressed_size = encoded.controls.len() + encoded.data.len();
        println!(
            "Original size: {} bytes, Compressed size: {} bytes",
            original_size, compressed_size
        );
        assert!(compressed_size < original_size);
    }

    #[test]
    fn test_stream_vbyte_u16_small() {
        let data: Vec<u16> = vec![1, 2, 3, 4, 5, 100, 1000, 10000];
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(data, decoded);
        assert_eq!(encoded.len(), data.len());
        assert!(!encoded.is_empty());
    }

    #[test]

    fn test_stream_vbyte_u16_large() {
        // Test with large u16 vector
        let data: Vec<u16> = (0..20000).map(|i| (i % 65536) as u16).collect();
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(data, decoded);
        assert_eq!(encoded.len(), data.len());

        // Verify that compression is effective
        let original_size = data.len() * 2; // 2 bytes per u16
        let compressed_size = encoded.controls.len() + encoded.data.len();
        println!(
            "Original size: {} bytes, Compressed size: {} bytes",
            original_size, compressed_size
        );
    }

    #[test]

    fn test_stream_vbyte_u32_mixed_values() {
        // Test with mixed values (small and large)
        let mut data: Vec<u32> = Vec::new();
        for i in 0..1000 {
            data.push(i); // Small values
            data.push(u32::MAX - i); // Large values
            data.push(1 << (i % 20)); // Powers of 2
            data.push((i * 17) % 10000); // Moderate random values
        }

        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(data, decoded);
        assert_eq!(encoded.len(), data.len());
    }

    #[test]

    fn test_stream_vbyte_u16_mixed_values() {
        // Test with mixed values for u16
        let mut data: Vec<u16> = Vec::new();
        for i in 0..1000 {
            data.push(i as u16); // Small values
            data.push(u16::MAX - i as u16); // Large values
            data.push(1 << (i % 14)); // Powers of 2 (max 2^15 for u16)
            data.push(((i * 17) % 1000) as u16); // Moderate random values
        }

        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(data, decoded);
        assert_eq!(encoded.len(), data.len());
    }

    #[test]
    fn test_stream_vbyte_empty() {
        let data: Vec<u32> = vec![];
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(data, decoded);
        assert_eq!(encoded.len(), 0);
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_stream_vbyte_single_value() {
        let data: Vec<u32> = vec![42];
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(data, decoded);
        assert_eq!(encoded.len(), 1);
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_stream_vbyte_compression_efficiency() {
        // Test with many small values to verify compression effectiveness
        let data: Vec<u32> = (0..50000).map(|i| i % 256).collect(); // Values requiring only 1 byte
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(data, decoded);
        assert_eq!(encoded.data.len(), data.len()); // Each value should take exactly 1 byte
    }

    #[test]
    fn test_stream_vbyte_iterator_u32() {
        let data: Vec<u32> = vec![1, 2, 3, 4, 5, 100, 1000, 10000];
        let encoded = StreamVByte::encode(&data);

        // Test iterator vs decode
        let iter_result: Vec<u32> = encoded.iter().collect();
        let decode_result = encoded.decode();

        assert_eq!(iter_result, data);
        assert_eq!(decode_result, data);
        assert_eq!(iter_result, decode_result);

        // Test iterator length
        assert_eq!(encoded.iter().len(), data.len());
    }

    #[test]
    fn test_stream_vbyte_iterator_large() {
        let data: Vec<u32> = (0..10000).map(|i| i * i + 100).collect();
        let encoded = StreamVByte::encode(&data);

        // Test iterator with large data
        let iter_result: Vec<u32> = encoded.iter().collect();
        let decode_result = encoded.decode();

        assert_eq!(data, iter_result);
        assert_eq!(data, decode_result);

        // Test partial iteration
        let partial: Vec<u32> = encoded.iter().take(100).collect();
        assert_eq!(partial, data[0..100]);

        // Test size hint
        let mut iter = encoded.iter();
        assert_eq!(iter.size_hint(), (data.len(), Some(data.len())));

        // Consume a few elements and check size hint
        iter.next();
        iter.next();
        iter.next();
        assert_eq!(iter.size_hint(), (data.len() - 3, Some(data.len() - 3)));
    }

    #[test]
    fn test_stream_vbyte_iterator_u16() {
        let data: Vec<u16> = (0..1000).map(|i| (i % 256) as u16).collect();
        let encoded = StreamVByte::encode(&data);

        // Test iterator
        let iter_result: Vec<u16> = encoded.iter().collect();
        assert_eq!(data, iter_result);

        // Test ExactSizeIterator
        assert_eq!(encoded.iter().len(), data.len());
    }

    #[test]
    fn test_stream_vbyte_iterator_empty() {
        let data: Vec<u32> = vec![];
        let encoded = StreamVByte::encode(&data);

        let iter_result: Vec<u32> = encoded.iter().collect();
        assert_eq!(data, iter_result);
        assert_eq!(encoded.iter().len(), 0);
    }

    #[test]
    fn test_stream_vbyte_iterator_single() {
        let data: Vec<u32> = vec![42];
        let encoded = StreamVByte::encode(&data);

        let iter_result: Vec<u32> = encoded.iter().collect();
        assert_eq!(data, iter_result);
        assert_eq!(encoded.iter().len(), 1);
    }

    #[test]
    fn test_stream_vbyte_iterator_buffer_behavior() {
        // Test that the iterator works correctly with groups of 4
        let data: Vec<u32> = (1..=17).collect(); // 17 elements to test multiple batches
        let encoded = StreamVByte::encode(&data);

        let mut iter = encoded.iter();

        // Verify that the iterator produces correct values one at a time
        for expected in &data {
            assert_eq!(iter.next(), Some(*expected));
        }

        // Verify that the iterator is exhausted
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_stream_vbyte_random_access_basic() {
        let data: Vec<u32> = (0..100).collect();
        let svb_ra = StreamVByteRandomAccess::new(&data, 16);

        // Test basic range iteration
        let portion: Vec<u32> = svb_ra.iter_range(10, 10).collect();
        assert_eq!(portion, (10..20).collect::<Vec<u32>>());

        // Test single element
        let single: Vec<u32> = svb_ra.iter_range(50, 1).collect();
        assert_eq!(single, vec![50]);

        // Test from beginning
        let beginning: Vec<u32> = svb_ra.iter_range(0, 5).collect();
        assert_eq!(beginning, (0..5).collect::<Vec<u32>>());
    }

    #[test]
    fn test_stream_vbyte_random_access_boundary_conditions() {
        let data: Vec<u32> = (0..100).collect();
        let svb_ra = StreamVByteRandomAccess::new(&data, 16);

        // Test empty range
        let empty: Vec<u32> = svb_ra.iter_range(50, 0).collect();
        assert_eq!(empty, Vec::<u32>::new());

        // Test range extending beyond data
        let beyond: Vec<u32> = svb_ra.iter_range(95, 10).collect();
        assert_eq!(beyond, (95..100).collect::<Vec<u32>>());

        // Test starting beyond data
        let beyond_start: Vec<u32> = svb_ra.iter_range(150, 10).collect();
        assert_eq!(beyond_start, Vec::<u32>::new());

        // Test entire sequence
        let all: Vec<u32> = svb_ra.iter_range(0, 100).collect();
        assert_eq!(all, data);

        // Test range larger than data
        let oversized: Vec<u32> = svb_ra.iter_range(0, 200).collect();
        assert_eq!(oversized, data);
    }

    #[test]
    fn test_stream_vbyte_random_access_block_alignment() {
        let data: Vec<u32> = (0..64).collect(); // 64 elements
        let svb_ra = StreamVByteRandomAccess::new(&data, 16); // 16 elements per block

        // Test ranges that align with block boundaries
        let block1: Vec<u32> = svb_ra.iter_range(0, 16).collect();
        assert_eq!(block1, (0..16).collect::<Vec<u32>>());

        let block2: Vec<u32> = svb_ra.iter_range(16, 16).collect();
        assert_eq!(block2, (16..32).collect::<Vec<u32>>());

        // Test ranges that cross block boundaries
        let cross_block: Vec<u32> = svb_ra.iter_range(10, 20).collect();
        assert_eq!(cross_block, (10..30).collect::<Vec<u32>>());

        // Test range spanning multiple blocks
        let multi_block: Vec<u32> = svb_ra.iter_range(5, 50).collect();
        assert_eq!(multi_block, (5..55).collect::<Vec<u32>>());
    }

    #[test]
    fn test_stream_vbyte_random_access_different_block_sizes() {
        let data: Vec<u32> = (0..100).collect();

        // Test with different block sizes
        for &block_size in &[4, 8, 16, 32] {
            let svb_ra = StreamVByteRandomAccess::new(&data, block_size);

            // Test various ranges
            let portion1: Vec<u32> = svb_ra.iter_range(20, 15).collect();
            assert_eq!(portion1, (20..35).collect::<Vec<u32>>());

            let portion2: Vec<u32> = svb_ra.iter_range(0, block_size).collect();
            assert_eq!(portion2, (0..block_size as u32).collect::<Vec<u32>>());

            let portion3: Vec<u32> = svb_ra.iter_range(block_size, block_size).collect();
            assert_eq!(
                portion3,
                ((block_size as u32)..(block_size as u32 * 2)).collect::<Vec<u32>>()
            );
        }
    }

    #[test]
    fn test_stream_vbyte_random_access_mixed_values() {
        // Test with mixed value sizes to ensure proper decoding
        let mut data: Vec<u32> = Vec::new();
        for i in 0..100 {
            data.push(i); // Small values (1 byte each)
            data.push(u32::MAX - i); // Large values (4 bytes each)
            data.push(1 << (i % 20)); // Variable sizes
        }

        let svb_ra = StreamVByteRandomAccess::new(&data, 24); // Block size multiple of 4

        // Test various ranges with mixed data
        let portion1: Vec<u32> = svb_ra.iter_range(0, 10).collect();
        assert_eq!(portion1, data[0..10]);

        let portion2: Vec<u32> = svb_ra.iter_range(50, 20).collect();
        assert_eq!(portion2, data[50..70]);

        let portion3: Vec<u32> = svb_ra.iter_range(100, 50).collect();
        assert_eq!(portion3, data[100..150]);

        // Test range crossing encoding boundaries
        let portion4: Vec<u32> = svb_ra.iter_range(97, 10).collect();
        assert_eq!(portion4, data[97..107]);
    }

    #[test]
    fn test_stream_vbyte_random_access_u16() {
        let data: Vec<u16> = (0..200).map(|i| (i % 65536) as u16).collect();
        let svb_ra = StreamVByteRandomAccess::new(&data, 20);

        // Test basic functionality with u16
        let portion: Vec<u16> = svb_ra.iter_range(50, 25).collect();
        assert_eq!(portion, data[50..75]);

        // Test with larger ranges
        let large_portion: Vec<u16> = svb_ra.iter_range(10, 100).collect();
        assert_eq!(large_portion, data[10..110]);
    }

    #[test]
    fn test_stream_vbyte_random_access_iterator_properties() {
        let data: Vec<u32> = (0..50).collect();
        let svb_ra = StreamVByteRandomAccess::new(&data, 12);

        // Test iterator length and size hint
        let mut iter = svb_ra.iter_range(10, 20);
        assert_eq!(iter.len(), 20);
        assert_eq!(iter.size_hint(), (20, Some(20)));

        // Consume some elements and check remaining length
        iter.next();
        iter.next();
        assert_eq!(iter.len(), 18);
        assert_eq!(iter.size_hint(), (18, Some(18)));

        // Test with partial ranges
        let partial_iter = svb_ra.iter_range(45, 10); // Should only get 5 elements (45-49)
        assert_eq!(partial_iter.len(), 5);

        let collected: Vec<u32> = partial_iter.collect();
        assert_eq!(collected, (45..50).collect::<Vec<u32>>());
    }

    #[test]
    fn test_stream_vbyte_random_access_edge_positions() {
        let data: Vec<u32> = (0..100).collect();
        let svb_ra = StreamVByteRandomAccess::new(&data, 16);

        // Test positions that are not aligned with encoding boundaries
        for start_pos in [1, 3, 5, 7, 9, 11, 13, 15] {
            let portion: Vec<u32> = svb_ra.iter_range(start_pos, 8).collect();
            assert_eq!(portion, data[start_pos..start_pos + 8]);
        }

        // Test various lengths
        for length in [1, 2, 3, 5, 7, 11, 13, 17, 19] {
            let portion: Vec<u32> = svb_ra.iter_range(20, length).collect();
            assert_eq!(portion, data[20..20 + length]);
        }
    }

    #[test]
    fn test_stream_vbyte_random_access_performance_patterns() {
        // Test patterns that might reveal performance issues
        let data: Vec<u32> = (0..1000).collect();
        let svb_ra = StreamVByteRandomAccess::new(&data, 32);

        // Sequential small reads
        for i in (0..900).step_by(10) {
            let portion: Vec<u32> = svb_ra.iter_range(i, 5).collect();
            assert_eq!(portion, data[i..i + 5]);
        }

        // Large reads at various positions
        for &start in &[0, 100, 200, 300, 400, 500, 600, 700, 800] {
            let portion: Vec<u32> = svb_ra.iter_range(start, 50).collect();
            assert_eq!(portion, data[start..start + 50]);
        }

        // Single element reads at various positions
        for &pos in &[0, 1, 99, 100, 199, 200, 299, 300, 499, 500, 999] {
            let portion: Vec<u32> = svb_ra.iter_range(pos, 1).collect();
            assert_eq!(portion, vec![data[pos]]);
        }
    }
}
