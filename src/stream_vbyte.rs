use mem_dbg::*;
use serde::{Deserialize, Serialize};

/// Code here is heavily inspired by [svbyte](https://github.com/bazhenov/svbyte/) and [stream_vbyte_rust](https://bitbucket.org/marshallpierce/stream-vbyte-rust) crates.
use crate::SVBEncodable;
use num::traits::{FromBytes, ToBytes, ops::bytes::NumBytes};

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
            dbg!(encoded_data.len());
            dbg!(len + 1);
            dbg!(v);
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
            dbg!(len, *data_index, self.data.len());

            let value = Self::decode_value(&self.data[*data_index..*data_index + len]);

            output.push(value);

            *data_index += len;
        }
    }

    #[inline]
    fn decode_value(slice: &[u8]) -> T {
        let mut buf = <T as FromBytes>::Bytes::default();
        let dst = buf.as_mut();
        dbg!(slice.len(), dst.len());
        dst[..slice.len()].copy_from_slice(&slice[..]);
        T::from_le_bytes(&buf)
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
        // Test con vettore grande di u32
        let data: Vec<u32> = (0..10000).map(|i| i * i + 100).collect();
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(data, decoded);
        assert_eq!(encoded.len(), data.len());

        // Verifica che la compressione sia efficace per valori piccoli
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
        // Test con vettore grande di u16
        let data: Vec<u16> = (0..20000).map(|i| (i % 65536) as u16).collect();
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(data, decoded);
        assert_eq!(encoded.len(), data.len());

        // Verifica che la compressione sia efficace
        let original_size = data.len() * 2; // 2 bytes per u16
        let compressed_size = encoded.controls.len() + encoded.data.len();
        println!(
            "Original size: {} bytes, Compressed size: {} bytes",
            original_size, compressed_size
        );
    }

    #[test]

    fn test_stream_vbyte_u32_mixed_values() {
        // Test con valori misti (piccoli e grandi)
        let mut data: Vec<u32> = Vec::new();
        for i in 0..1000 {
            data.push(i); // Valori piccoli
            data.push(u32::MAX - i); // Valori grandi
            data.push(1 << (i % 20)); // Potenze di 2
            data.push((i * 17) % 10000); // Valori casuali moderati
        }

        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(data, decoded);
        assert_eq!(encoded.len(), data.len());
    }

    #[test]

    fn test_stream_vbyte_u16_mixed_values() {
        // Test con valori misti per u16
        let mut data: Vec<u16> = Vec::new();
        for i in 0..1000 {
            data.push(i as u16); // Valori piccoli
            data.push(u16::MAX - i as u16); // Valori grandi
            data.push(1 << (i % 14)); // Potenze di 2 (max 2^15 per u16)
            data.push(((i * 17) % 1000) as u16); // Valori casuali moderati
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
        // Test con molti valori piccoli per verificare l'efficacia della compressione
        let data: Vec<u32> = (0..50000).map(|i| i % 256).collect(); // Valori che richiedono solo 1 byte
        let encoded = StreamVByte::encode(&data);
        let decoded = encoded.decode();

        assert_eq!(data, decoded);
        assert_eq!(encoded.data.len(), data.len()); // Each value should take exactly 1 byte
    }
}
