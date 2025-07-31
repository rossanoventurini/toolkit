use crate::BitVec;
use crate::bitvector::{BitBoxed, BitVector};
use crate::utils::compute_mask;

use mem_dbg::*;
use serde::{Deserialize, Serialize};

/// A resizable, growable, and mutable bit field vector.
pub type BitFieldVec = BitField<Vec<u64>>;
/// Bit operations on a slice of u64, immutable or mutable but not growable bit field field vector.
pub type BitFieldSlice<'a> = BitField<&'a [u64]>;
/// Bit operations on a boxed slice of u64, immutable or mutable but not growable bit field vector.
pub type BitFieldBoxed = BitField<Box<[u64]>>;

/// A bit field structure that stores values using a fixed number of bits per value.
/// This is useful for storing sequences of integers that can be represented with
/// fewer than 64 bits each, providing space efficiency.
///
///
/// TODO: add methods to modify bits or append new values, an many more.
/// TODO: add iterators for iterating over the values
#[derive(Debug, Default, Clone, Serialize, Deserialize, Eq, PartialEq, MemSize, MemDbg)]
pub struct BitField<V: AsRef<[u64]>> {
    /// The underlying bit vector storing the packed values
    bitvector: BitVector<V>,
    /// Number of bits used to represent each value (0-64)
    field_width: u8,
    /// Mask with the last field_width bits set to 1, used for extracting values
    mask: u64,
    /// Number of fields stored in the bit field
    length: usize,
}

impl<V: AsRef<[u64]> + Default> BitField<V> {
    /// Creates a new BitField with the specified number of bits per value.
    ///
    /// # Arguments
    /// * `field_width` - Number of bits per value (must be between 0 and 64)
    ///
    /// # Panics
    /// Panics if field_width is greater than 64.
    pub fn new(field_width: u8) -> Self {
        assert!(field_width <= 64, "field_width must be between 0 and 64");

        let mask = compute_mask(field_width as usize);

        Self {
            bitvector: BitVector::<V>::default(),
            field_width,
            mask,
            length: 0,
        }
    }

    /// Returns the number of bits used per value.
    pub fn field_width(&self) -> u8 {
        self.field_width
    }

    /// Returns the mask used for extracting values.
    pub fn mask(&self) -> u64 {
        self.mask
    }

    /// Returns the number of values stored in the bit field.
    pub fn len(&self) -> usize {
        self.length
    }

    /// Returns true if the bit field is empty.
    pub fn is_empty(&self) -> bool {
        self.bitvector.is_empty()
    }

    /// Gets the value at the specified index.
    ///
    /// # Arguments
    /// * `index` - The index of the value to retrieve
    ///
    /// # Returns
    /// The value at the specified index, or None if the index is out of bounds.
    pub fn get(&self, index: usize) -> Option<u64> {
        if index >= self.len() {
            return None;
        }

        Some(unsafe { self.get_unchecked(index) })
    }

    /// Gets the value at the specified index without bounds checking.
    ///
    /// # Safety
    /// The caller must ensure that the index is within bounds.
    pub unsafe fn get_unchecked(&self, index: usize) -> u64 {
        let bit_index = index * self.field_width as usize;
        unsafe {
            self.bitvector
                .get_bits_unchecked(bit_index, self.field_width as usize)
        }
    }
}

impl BitField<Vec<u64>> {
    #[must_use]
    pub fn with_capacity(n_vals: usize, width: u8) -> Self {
        assert!(width <= 64, "width must be between 0 and 64");

        let capacity = n_vals * width as usize;
        Self {
            bitvector: BitVec::with_capacity(capacity),
            field_width: width,
            mask: compute_mask(width as usize),
            length: 0,
        }
    }

    pub fn push(&mut self, value: u64) {
        if self.field_width == 0 {
            // Special case: if field_width is 0, we only store zeros
            assert!(
                value == 0,
                "Cannot push non-zero value when field_width is 0"
            );

            if self.len() == 0 {
                // If this is the first value, initialize the bitvector
                self.bitvector = BitVec::with_zeros(1);
            }

            self.length += 1;
            return;
        }

        assert!(
            value <= self.mask,
            "Value exceeds the maximum representable value"
        );

        self.bitvector.append_bits(value, self.field_width as usize);
        self.length += 1;
    }
}

impl<D: AsRef<[u64]>, V: AsRef<[u64]> + Default + From<Vec<u64>>> From<D> for BitField<V> {
    /// Creates a BitField from a slice of u64 values.
    ///
    /// This implementation calculates the minimum number of bits required to store
    /// the largest value in the input data, then packs all values using that
    /// number of bits.
    ///
    /// # Arguments
    /// * `data` - A reference to a slice of u64 values
    ///
    /// # Returns
    /// A BitField containing all the values from the input data.
    ///
    /// # Examples
    /// ```
    /// use toolkit::BitFieldBoxed;
    ///
    /// let data = vec![1, 7, 15, 3];
    /// let bf = BitFieldBoxed::from(data);
    ///
    /// assert_eq!(bf.field_width(), 4); // 15 requires 4 bits
    /// assert_eq!(bf.get(0), Some(1));
    /// assert_eq!(bf.get(1), Some(7));
    /// assert_eq!(bf.get(2), Some(15));
    /// assert_eq!(bf.get(3), Some(3));
    /// ```
    fn from(data: D) -> Self {
        let data = data.as_ref();

        if data.is_empty() {
            return Self::new(1);
        }

        // Find the maximum value to determine the minimum number of bits needed
        let max_value = *data.iter().max().unwrap();

        // Calculate the minimum number of bits needed
        // For max_value = 0, we need 0 bits (special case handled below)
        // For max_value > 0, we need ceil(log2(max_value + 1)) bits
        let field_width = if max_value == 0 {
            0
        } else {
            64 - max_value.leading_zeros() as u8
        };

        // Create the mask
        let mask = compute_mask(field_width as usize);

        // For the special case where all values are 0 (field_width = 0),
        // we create a BitVector with only one bit set to 0
        let total_bits = if field_width == 0 {
            1 // Use 1 bit for all the values, set to 0
        } else {
            data.len() * field_width as usize
        };

        let mut bitvector = BitBoxed::with_zeros(total_bits);

        // Pack the values into the bit vector
        // Special case: if field_width = 0, all values are 0 and the bitvector is already all zeros
        if field_width > 0 {
            for (i, &value) in data.iter().enumerate() {
                let bit_index = i * field_width as usize;
                bitvector.set_bits(bit_index, field_width as usize, value & mask);
            }
        }

        Self {
            bitvector: bitvector.convert_into(),
            field_width,
            mask,
            length: data.len(),
        }
    }
}

impl<V> BitField<V>
where
    V: AsRef<[u64]>,
{
    /// Converts the `BitField` into a new `BitField` with a different data type.
    ///
    /// We do not implement `From<BitField<S>> for BitField<D>` because it would conflict with the blanket
    ///  implementation `impl<T> From<T> for T>` provided by the standard library when `V == D`.
    ///  Instead, we expose a `convert_into` method to handle the conversion explicitly without ambiguity.

    pub fn convert_into<D>(&self) -> BitField<D>
    where
        D: AsRef<[u64]> + From<Vec<u64>>,
    {
        BitField::<D> {
            bitvector: self.bitvector.convert_into(),
            field_width: self.field_width,
            mask: self.mask,
            length: self.length,
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_push_basic() {
        let mut bf = BitFieldVec::new(4);
        bf.push(1);
        bf.push(7);
        bf.push(15);
        bf.push(3);
        assert_eq!(bf.len(), 4);
        assert_eq!(bf.get(0), Some(1));
        assert_eq!(bf.get(1), Some(7));
        assert_eq!(bf.get(2), Some(15));
        assert_eq!(bf.get(3), Some(3));
    }

    #[test]
    fn test_push_zero_width() {
        let mut bf = BitFieldVec::new(0);
        bf.push(0);
        bf.push(0);
        assert_eq!(bf.len(), 2);
        assert_eq!(bf.get(0), Some(0));
        assert_eq!(bf.get(1), Some(0));
    }

    #[test]
    #[should_panic]
    fn test_push_zero_width_nonzero_value() {
        let mut bf = BitFieldVec::new(0);
        bf.push(1); // panic
    }

    #[test]
    #[should_panic]
    fn test_push_value_too_large() {
        let mut bf = BitFieldVec::new(3);
        bf.push(8); // 8 non rappresentabile con 3 bit
    }
    use super::*;

    #[test]
    fn test_new_bitfield() {
        let bf = BitFieldVec::new(8);
        assert_eq!(bf.field_width(), 8);
        assert_eq!(bf.mask(), 0xFF);
        assert!(bf.is_empty());
    }

    #[test]
    fn test_from_empty_data() {
        let data: Vec<u64> = vec![];
        let bf = BitFieldVec::from(data);
        assert_eq!(bf.field_width(), 1);
        assert!(bf.is_empty());
    }

    #[test]
    fn test_from_data_with_small_values() {
        let data = vec![1, 7, 15, 3];
        let bf = BitFieldVec::from(data);

        assert_eq!(bf.field_width(), 4); // 15 = 0b1111 requires 4 bits
        assert_eq!(bf.len(), 4);
        assert_eq!(bf.get(0), Some(1));
        assert_eq!(bf.get(1), Some(7));
        assert_eq!(bf.get(2), Some(15));
        assert_eq!(bf.get(3), Some(3));
    }

    #[test]
    fn test_from_data_with_zero() {
        let data = vec![0, 1, 2];
        let bf = BitFieldVec::from(data);

        assert_eq!(bf.field_width(), 2); // 2 = 0b10 requires 2 bits
        assert_eq!(bf.get(0), Some(0));
        assert_eq!(bf.get(1), Some(1));
        assert_eq!(bf.get(2), Some(2));
    }

    #[test]
    fn test_from_data_only_zeros() {
        let data = vec![0, 0, 0];
        let bf = BitFieldVec::from(data);

        assert_eq!(bf.field_width(), 0); // 0 requires 0 bits (special case)
        assert_eq!(bf.len(), 3);
        assert_eq!(bf.get(0), Some(0));
        assert_eq!(bf.get(1), Some(0));
        assert_eq!(bf.get(2), Some(0));
    }

    #[test]
    fn test_from_single_zero() {
        let data = vec![0];
        let bf = BitFieldVec::from(data);

        assert_eq!(bf.field_width(), 0); // Single 0 requires 0 bits
        assert_eq!(bf.len(), 1);
        assert_eq!(bf.get(0), Some(0));
        assert_eq!(bf.get(1), None); // Out of bounds
    }

    #[test]
    fn test_out_of_bounds_access() {
        let data = vec![1, 2, 3];
        let bf = BitFieldVec::from(data);

        assert_eq!(bf.get(3), None);
        assert_eq!(bf.get(100), None);
    }

    #[test]
    fn test_len_method() {
        // Test with different data sizes
        let data1 = vec![1, 2, 3, 4, 5];
        let bf1 = BitFieldVec::from(data1);
        assert_eq!(bf1.len(), 5);

        let data2 = vec![0, 0, 0, 0];
        let bf2 = BitFieldVec::from(data2);
        assert_eq!(bf2.len(), 4);

        // Empty BitField
        let bf3 = BitFieldVec::new(8);
        assert_eq!(bf3.len(), 0);
    }

    #[test]
    fn test_new_with_zero_bits() {
        // Now field_width = 0 is allowed for the case where all values are zero
        let bf = BitFieldVec::new(0);
        assert_eq!(bf.field_width(), 0);
        assert_eq!(bf.mask(), 0);
        assert!(bf.is_empty());
    }

    #[test]
    #[should_panic]
    fn test_new_with_too_many_bits() {
        BitFieldVec::new(65);
    }
}
