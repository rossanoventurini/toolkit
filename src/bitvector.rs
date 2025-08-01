//! # BitVector Module
//!
//! This module provides flexible and efficient implementations of mutable, immutable, and growable bit vectors.
//! It supports single-bit access, appending, slicing with offsets, iteration over set or unset bits,
//! and conversion to and from compact or growable representations.
//!
//! ## Features
//!
//! - `BitVec`: A growable and mutable bit vector (`Vec<u64>`-backed).
//! - `BitBoxed`: A compact, fixed-size version of a bit vector.
//! - `BitSlice`: A read-only view over existing memory.
//! - Efficient iteration over `1` and `0` bit positions.
//! - Fast `get_bits` and `set_bits` (up to 64 bits).
//! - Optional offset support via `BitSliceWithOffset`.
//!
//! ## Examples
//!
//! ### Basic usage
//!
//! ```rust
//! use toolkit::{BitVec, AccessBin};
//!
//! let mut bv = BitVec::new();
//! bv.push(true);
//! bv.push(false);
//! bv.push(true);
//!
//! assert_eq!(bv.len(), 3);
//! assert_eq!(bv.get(0), Some(true));
//! assert_eq!(bv.get(1), Some(false));
//!
//! bv.set(1, true);
//! assert_eq!(bv.get(1), Some(true));
//! ```
//!
//! ### Creating a compact `BitBoxed` bit vector
//!
//! ```rust
//! use toolkit::BitBoxed;
//! use toolkit::AccessBin;
//!
//! let bb = BitBoxed::with_ones(10);
//! assert_eq!(bb.count_ones(), 10);
//! assert_eq!(bb.get(9), Some(true));
//! ```
//!
//! ### Iterating over positions of ones
//!
//! ```rust
//! use toolkit::BitVec;
//! use toolkit::AccessBin;
//!
//! let vv = vec![0, 3, 5];
//! let bv: BitVec = vv.iter().copied().collect();
//!
//! let ones: Vec<usize> = bv.ones().collect();
//! assert_eq!(ones, vv);
//! ```
//!
//! ### Using `BitSliceWithOffset`
//!
//! ```rust
//! use toolkit::{BitVec, BitSliceWithOffset};
//!
//! let mut bv = BitVec::new();
//! bv.append_bits(0b1111_0000, 8);
//! let slice = BitSliceWithOffset::new(&bv, 4);
//!
//! assert_eq!(slice.get_bits(0, 4), Some(0b1111));
//! ```

// TODO:
// - add CacheLine-based bit vectors
// - create a BitBoxed with fixed size (with_zeros() or with_ones())
// - add a function to get a BitSlice from a starting word of a given bitlength

use crate::AccessBin;
use crate::utils::compute_mask;

use mem_dbg::*;
use serde::{Deserialize, Serialize};

/// A resizable, growable, and mutable bit vector.
pub type BitVec = BitVector<Vec<u64>>;
/// Bit operations on a slice of u64, immutable or mutable but not growable bit vector.
pub type BitSlice<'a> = BitVector<&'a [u64]>;
/// Bit operations on a boxed slice of u64, immutable or mutable but not growable bit vector.
pub type BitBoxed = BitVector<Box<[u64]>>;

/// Implementation of an immutable bit vector.
#[derive(Default, Clone, Serialize, Deserialize, Eq, PartialEq, MemSize, MemDbg)]
pub struct BitVector<V: AsRef<[u64]>> {
    data: V,
    n_bits: usize,
}

impl<V: AsRef<[u64]>> BitVector<V> {
    /// Creates a `BitVector` from raw parts.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it does not perform bounds checking.
    /// It is the caller's responsibility to ensure that the provided `data` and `n_bits`
    /// are valid and consistent.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::{BitSlice, BitVec};
    ///
    /// let data = vec![0, 2, 3, 4, 5];
    /// let n_bits = data.len() * 64;
    /// let bv = unsafe { BitVec::from_raw_parts(data, n_bits) };
    ///
    /// assert_eq!(bv.get_bits(64, 64), Some(2));
    ///
    /// let data = vec![0, 2, 3, 4, 5];
    /// let n_bits = data.len() * 64;
    /// let bs = unsafe { BitSlice::from_raw_parts(&data[1..], n_bits-64) };
    ///
    /// assert_eq!(bs.get_bits(0, 64), Some(2));
    ///
    /// ```
    pub unsafe fn from_raw_parts(data: V, n_bits: usize) -> Self {
        Self { data, n_bits }
    }

    /// Accesses `len` bits, with 0 <= `len` <= 64, starting at position `index`.
    ///
    /// Returns [`None`] if `index`+`len` is out of bounds or if `len` is greater than 64.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::{BitVec, BitSlice, AccessBin};
    ///
    /// let v = vec![0,2,3,4,5];
    /// let bv: BitVec = v.into_iter().collect();
    /// assert_eq!(bv.get(1), Some(false));
    ///
    /// assert_eq!(bv.get_bits(1, 3), Some(0b110)); // Accesses bits from index 1 to 3
    ///
    /// // Accessing bits from index 1 to 8, which is out of bounds
    /// assert_eq!(bv.get_bits(1, 8), None);
    ///
    /// // Accessing more than 64 bits
    /// assert_eq!(bv.get_bits(0, 65), None);
    ///
    /// // Accessing 0 bits
    /// assert_eq!(bv.get_bits(2, 0), Some(0));
    ///
    /// // Accessing last bit
    /// assert_eq!(bv.get_bits(bv.len()-1, 1), Some(1));
    /// ```
    #[must_use]
    #[inline]
    pub fn get_bits(&self, index: usize, len: usize) -> Option<u64> {
        if (len > 64) | (index + len > self.n_bits) {
            return None;
        }
        // SAFETY: safe access due to the above checks
        Some(unsafe { self.get_bits_unchecked(index, len) })
    }

    /// Accesses `len` bits, starting at position `index`, without performing bounds checking.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it does not perform bounds checking.
    /// It is the caller's responsibility to ensure that the provided `index` and `len`
    /// are within the bounds of the bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::{BitVec};
    ///
    /// let v = vec![0,2,3,4,5];
    /// let bv: BitVec = v.into_iter().collect();
    ///
    /// assert_eq!(unsafe{bv.get_bits_unchecked(0, 4)}, 0b1101);
    /// assert_eq!(unsafe{bv.get_bits_unchecked(0, 0)}, 0);
    /// ```
    #[must_use]
    #[inline]
    pub unsafe fn get_bits_unchecked(&self, index: usize, len: usize) -> u64 {
        debug_assert!(len <= 64 && index + len <= self.n_bits);

        unsafe { Self::get_bits_slice(self.data.as_ref(), index, len) }
    }

    // TODO: make the two functions a trait and implement for &[u64] together with set_bit and set_bits for &mut [T]. This way we can have a generic type T which implements those traits for &[T] and &mut [T].

    // Private function to decode bits at a given index on a slice.
    // The function does not check bounds while accessing data and does not clear bits in position larger than len.
    #[inline]
    #[must_use]
    unsafe fn get_bits_unmasked_slice(data: &[u64], index: usize, len: usize) -> u64 {
        let (block, shift) = (index >> 6, index & 63);

        let w = unsafe { *data.get_unchecked(block) } >> shift;

        if shift + len <= 64 {
            w
        } else {
            w | (unsafe { *data.get_unchecked(block + 1) } << (64 - shift))
        }
    }

    /// Returns the position of the next 1 bit in the bit vector starting from position `index`.
    /// Returns [`None`] if `index` is out of bounds or if there is no one after index.
    ///
    /// # Examples
    /// ```
    /// use toolkit::{BitVec, AccessBin};
    ///
    /// let v = vec![0, 2, 3, 5];
    /// let bv: BitVec = v.into_iter().collect();
    ///
    /// assert_eq!(bv.next_one(0), Some(0)); // First 1 at position 0
    /// assert_eq!(bv.next_one(1), Some(2)); // Next 1 after position 1 is at position 2
    /// assert_eq!(bv.next_one(4), Some(5)); // Next 1 after position 4 is at position 5
    /// assert_eq!(bv.next_one(6), None);    // No 1 after position 6
    /// ```
    #[inline]
    #[must_use]
    pub fn next_one(&self, index: usize) -> Option<usize> {
        if index >= self.n_bits {
            return None;
        }

        // SAFETY: index is ok due to the above check
        unsafe { self.next_one_unchecked(index) }.filter(|&res| res < self.n_bits)
    }

    /// Returns the position of the next 1 bit in the bit vector starting from position `index`.
    ///
    /// If there is no bit after that position, the function returns a value larger than or
    /// equal to the number of bits in the bit vector. The function does not check bounds.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index` is within bounds of the bit vector.
    /// Invoking this function with an out-of-bounds index is undefined behavior.
    #[inline]
    #[must_use]
    /// Returns the position of the next 1 bit in the bit vector starting from position `index` without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index` is within bounds of the bit vector and that the offset is valid.
    /// Invoking this function with an out-of-bounds index or invalid offset is undefined behavior.
    pub unsafe fn next_one_unchecked(&self, index: usize) -> Option<usize> {
        unsafe { Self::next_bit_slice_unchecked::<true>(self.data.as_ref(), index, self.n_bits) }
    }

    /// Returns the position of the next 0 bit in the bit vector starting from position `index`.
    /// Returns [`None`] if `index` is out of bounds or if there is no zero after index.
    ///
    /// # Examples
    /// ```
    /// use toolkit::{BitVec};
    ///
    /// let v = vec![0, 2, 3, 5];
    /// let bv: BitVec = v.into_iter().collect();
    ///
    /// assert_eq!(bv.next_zero(0), Some(1)); // First 0 after position 0 is at position 1
    /// assert_eq!(bv.next_zero(2), Some(4)); // Next 0 after position 2 is at position 4
    /// assert_eq!(bv.next_zero(5), None);    // No 0 after position 5 (end of bitvector)
    /// ```
    #[inline]
    #[must_use]
    pub fn next_zero(&self, index: usize) -> Option<usize> {
        if index >= self.n_bits {
            return None;
        }

        // SAFETY: index is ok due to the above check
        unsafe { self.next_zero_unchecked(index) }.filter(|&res| res < self.n_bits)
    }

    /// Returns the position of the next 0 bit in the bit vector starting from position `index`.
    ///
    /// If there is no bit after that position, the function returns a value larger than or equal
    /// to the number of bits in the bit vector. The function does not check bounds.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index` is within bounds of the bit vector.
    /// Invoking this function with an out-of-bounds index is undefined behavior.
    #[inline]
    #[must_use]
    /// Returns the position of the next 0 bit in the bit vector starting from position `index` without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `index` is within bounds of the bit vector and that the offset is valid.
    /// Invoking this function with an out-of-bounds index or invalid offset is undefined behavior.
    pub unsafe fn next_zero_unchecked(&self, index: usize) -> Option<usize> {
        unsafe { Self::next_bit_slice_unchecked::<false>(self.data.as_ref(), index, self.n_bits) }
    }

    // Private function that returns the position of the next 1 bit in the bit vector starting
    // from position `index``. If such bit does not exist, the function returns a value larger
    // than or euqal to the number of bits in the bit vector.
    //
    // UB: if `index` is out of bounds.
    #[inline]
    #[must_use]
    unsafe fn next_bit_slice_unchecked<const BIT: bool>(
        data: &[u64],
        index: usize,
        n_bits: usize,
    ) -> Option<usize> {
        let mut block = index >> 6;
        let shift = index & 63;
        let last_block = (n_bits - 1) >> 6; //  block of the last bit

        let w = if BIT {
            unsafe { *data.get_unchecked(block) }
        } else {
            unsafe { !*data.get_unchecked(block) }
        } >> shift;

        if w != 0 {
            return Some(index + w.trailing_zeros() as usize);
        }

        block += 1;

        while block <= last_block {
            let w = if BIT {
                unsafe { *data.get_unchecked(block) }
            } else {
                unsafe { !*data.get_unchecked(block) }
            };

            if w != 0 {
                return Some(((block) << 6) + w.trailing_zeros() as usize);
            }
            block += 1;
        }
        None
    }

    // Private function to decode bits at a given index on a slice.
    // The function does not check bounds while accessing data.
    #[inline]
    #[must_use]
    unsafe fn get_bits_slice(data: &[u64], index: usize, len: usize) -> u64 {
        (unsafe { Self::get_bits_unmasked_slice(data, index, len) }) & (compute_mask(len))
    }

    // Private function to decode a bit at a given index on a slice. The function does not
    // check bounds.
    #[inline]
    #[must_use]
    unsafe fn get_bit_slice(data: &[u64], index: usize) -> bool {
        let word = index >> 6;
        let pos_in_word = index & 63;

        (unsafe { *data.get_unchecked(word) } >> pos_in_word) & 1 != 0
    }

    /// Gets a whole 64-bit word from the bit vector at index `i` in the underlying vector of u64.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::BitVec;
    ///
    /// let v = vec![0,2,3,4,5];
    /// let bv: BitVec = v.into_iter().collect();
    ///
    /// // Get the 64-bit word at index 0
    /// let word = bv.get_word(0);
    /// assert_eq!(word, 0b111101);
    /// ```
    #[must_use]
    #[inline]
    pub fn get_word(&self, i: usize) -> u64 {
        self.data.as_ref()[i]
    }

    /// Returns the 64-bit word at the given index without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `i` is a valid index into the data array.
    /// Invoking this function with an out-of-bounds index is undefined behavior.
    #[must_use]
    #[inline]
    pub unsafe fn get_word_unchecked(&self, i: usize) -> u64 {
        unsafe { *self.data.as_ref().get_unchecked(i) }
    }

    /// Returns a non-consuming iterator over positions of bits set to 1 in the bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::BitVec;
    ///
    /// let vv: Vec<usize> = vec![0, 63, 128, 129, 254, 1026];
    /// let bv: BitVec = vv.iter().copied().collect();
    ///
    /// let v: Vec<usize> = bv.ones().collect();
    /// assert_eq!(v, vv);
    /// ```
    #[must_use]
    pub fn ones(&self) -> BitVectorBitPositionsIter<true> {
        let bs = unsafe { BitSliceWithOffset::from_raw_parts(self.data.as_ref(), self.n_bits, 0) };

        BitVectorBitPositionsIter::new(bs)
    }

    /// Returns a non-consuming iterator over positions of bits set to 1 in the bit vector, starting at a specified bit position.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::BitVec;
    ///
    /// let vv: Vec<usize> = vec![0, 63, 128, 129, 254, 1026];
    /// let bv: BitVec = vv.iter().copied().collect();
    ///
    /// let v: Vec<usize> = bv.ones_with_pos(2).collect();
    /// assert_eq!(v, vec![63, 128, 129, 254, 1026]);
    /// ```
    #[must_use]
    pub fn ones_with_pos(&self, pos: usize) -> BitVectorBitPositionsIter<true> {
        let bs = unsafe { BitSliceWithOffset::from_raw_parts(self.data.as_ref(), self.n_bits, 0) };

        BitVectorBitPositionsIter::with_pos(bs, pos)
    }

    /// Returns a non-consuming iterator over positions of bits set to 0 in the bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::BitVec;
    /// use toolkit::gen_sequences::negate_vector;
    ///
    /// let vv: Vec<usize> = vec![0, 63, 128, 129, 254, 1026];
    /// let bv: BitVec = vv.iter().copied().collect();
    ///
    /// let v: Vec<usize> = bv.zeros().collect();
    /// assert_eq!(v, negate_vector(&vv));
    /// ```
    #[must_use]
    pub fn zeros(&self) -> BitVectorBitPositionsIter<false> {
        let bs = unsafe { BitSliceWithOffset::from_raw_parts(self.data.as_ref(), self.n_bits, 0) };

        BitVectorBitPositionsIter::new(bs)
    }

    /// Returns a non-consuming iterator over positions of bits set to 0 in the bit vector, starting at a specified bit position.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::BitVec;
    /// use toolkit::gen_sequences::negate_vector;
    ///
    /// let vv: Vec<usize> = vec![0, 63, 128, 129, 254, 1026];
    /// let bv: BitVec = vv.iter().copied().collect();
    ///
    /// let v: Vec<usize> = bv.zeros_with_pos(100).collect();
    /// let expected: Vec<usize> = negate_vector(&vv).into_iter().filter(|&x| x >= 100).collect();
    /// assert_eq!(v, expected);
    /// ```
    #[must_use]
    pub fn zeros_with_pos(&self, pos: usize) -> BitVectorBitPositionsIter<false> {
        let bs = unsafe { BitSliceWithOffset::from_raw_parts(self.data.as_ref(), self.n_bits, 0) };

        BitVectorBitPositionsIter::with_pos(bs, pos)
    }

    /// Returns a non-consuming iterator over bits of the bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::BitVec;
    ///
    /// let v = vec![0,2,3,5];
    /// let bv: BitVec = v.into_iter().collect();
    ///
    /// let mut iter = bv.iter();
    /// assert_eq!(iter.next(), Some(true)); // First bit is true
    /// assert_eq!(iter.next(), Some(false)); // Second bit is false
    /// assert_eq!(iter.next(), Some(true)); // Third bit is true
    /// assert_eq!(iter.next(), Some(true)); // Fourth bit is true
    /// assert_eq!(iter.next(), Some(false)); // Fifth bit is false
    /// assert_eq!(iter.next(), Some(true)); // Sixth bit is true
    /// assert_eq!(iter.next(), None); // End of the iterator
    /// ```
    pub fn iter(&self) -> BitVectorIter<V, &Self> {
        BitVectorIter {
            bv: self,
            i: 0,
            n_bits: self.n_bits,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Checks if the bit vector is empty.
    ///
    /// # Returns
    ///
    /// Returns `true` if the bit vector is empty, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::BitVec;
    ///
    /// let v = vec![0,2,3,4,5];
    /// let bv: BitVec = v.into_iter().collect();
    ///
    /// assert!(!bv.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n_bits == 0
    }

    /// Returns the number of bits in the bit vector.
    ///
    /// # Returns
    ///
    /// The number of bits in the bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::BitVec;
    ///
    /// let v = vec![0,2,3,4,5];
    /// let bv: BitVec = v.into_iter().collect();
    ///
    /// assert_eq!(bv.len(), 6);
    /// ```
    pub fn len(&self) -> usize {
        self.n_bits
    }

    /// Counts the number of ones (bits set to 1) in the bit vector.
    /// This is an expensive operation, as it requires iterating over the entire bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::BitVec;
    ///
    /// let v = vec![0,2,3,4,5];
    /// let bv: BitVec = v.into_iter().collect();
    ///
    /// assert_eq!(bv.count_ones(), 5);
    /// ```
    pub fn count_ones(&self) -> usize {
        self.data
            .as_ref()
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// Counts the number of zeros (bits set to 0) in the bit vector.
    /// This is an expensive operation, as it requires iterating over the entire bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::BitVec;
    ///
    /// let v = vec![0,2,3,4,5];
    /// let bv: BitVec = v.into_iter().collect();
    ///
    /// assert_eq!(bv.count_zeros(), 1);
    /// ```
    #[inline]
    #[must_use]
    pub fn count_zeros(&self) -> usize {
        self.len() - self.count_ones()
    }
}

impl<V: AsRef<[u64]>> AccessBin for BitVector<V> {
    /// Returns the bit at the given position `index`,
    /// or [`None`] if `index` is out of bounds.
    ///
    /// # Examples
    /// ```
    /// use toolkit::{BitVec, AccessBin};
    ///
    /// let v = vec![0,2,3,4,5];
    /// let bv: BitVec = v.into_iter().collect();
    ///
    /// assert_eq!(bv.get(5), Some(true));
    /// assert_eq!(bv.get(1), Some(false));
    /// assert_eq!(bv.get(10), None);
    /// ```
    #[inline]
    fn get(&self, index: usize) -> Option<bool> {
        if index >= self.len() {
            return None;
        }
        Some(unsafe { self.get_unchecked(index) })
    }

    /// Returns the bit at position `index`.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    ///
    /// # Examples
    /// ```
    /// use toolkit::{BitVec, AccessBin};
    ///
    /// let v = vec![0,2,3,4,5];
    /// let bv: BitVec = v.into_iter().collect();
    ///
    /// assert_eq!(unsafe{bv.get_unchecked(5)}, true);
    /// ```
    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> bool {
        unsafe { Self::get_bit_slice(self.data.as_ref(), index) }
    }
}

impl<V> BitVector<V>
where
    V: AsRef<[u64]>,
{
    /// Converts the `BitVector` into a new `BitVector` with a different data type.
    ///
    /// We do not implement `From<BitVector<S>> for BitVector<D>` because it would conflict with the blanket
    /// implementation `impl<T> From<T> for T>` provided by the standard library when `V == D`.
    /// Instead, we expose a `convert_into` method to handle the conversion explicitly without ambiguity.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::{BitVec, BitBoxed, AccessBin};
    ///
    /// let mut bv = BitVec::new();
    /// bv.push(true);
    /// bv.push(false);
    /// bv.push(true);
    ///
    /// // Convert from growable BitVec to fixed-size BitBoxed
    /// let bb: BitBoxed = bv.convert_into();
    ///
    /// assert_eq!(bb.len(), 3);
    /// assert_eq!(bb.get(0), Some(true));
    /// assert_eq!(bb.get(1), Some(false));
    /// ```
    pub fn convert_into<D>(&self) -> BitVector<D>
    where
        D: AsRef<[u64]> + From<Vec<u64>>,
    {
        let data = self.data.as_ref().to_vec().into();
        let n_bits = self.n_bits;
        BitVector { data, n_bits }
    }
}

impl<V: AsRef<[u64]> + AsMut<[u64]>> BitVector<V> {
    /// Sets the bit at the given position `index` to `bit`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::{BitVec, BitBoxed, AccessBin};
    ///
    /// let mut bv = BitVec::with_capacity(2);
    /// bv.push(true);
    /// bv.push(false);
    ///
    /// bv.set(1, true);
    /// assert_eq!(bv.get(1), Some(true));
    ///
    /// // This will panic because index is out of bounds
    /// // bv.set(10, false);
    ///
    /// let mut bb = BitBoxed::from(bv);
    /// bb.set(0, false);
    /// assert_eq!(bb.get(0), Some(false));
    ///
    /// ```
    #[inline]
    pub fn set(&mut self, index: usize, bit: bool) {
        assert!(index < self.n_bits);

        let word = index >> 6;
        let pos_in_word = index & 63;
        self.data.as_mut()[word] &= !(1_u64 << pos_in_word);
        self.data.as_mut()[word] |= (bit as u64) << pos_in_word;
    }

    /// Sets `len` bits, with 1 <= `len` <= 64,
    /// starting at position `index` to the `len` least
    /// significant bits in `bits`.
    ///
    /// # Panics
    ///
    /// Panics if `index`+`len` is out of bounds,
    /// `len` is greater than 64, or if the most significant bit in `bits`
    /// is at a position larger than or equal to `len`.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::{BitVec, BitBoxed};
    ///
    /// let mut bv = BitVec::with_zeros(5);
    /// bv.set_bits(0, 3, 0b101); // Sets bits 0 to 2 to 101
    /// assert_eq!(bv.get_bits(0, 3), Some(0b101));
    ///
    /// let mut bb = BitBoxed::from(bv);
    /// bb.set_bits(0, 3, 0b100); // Sets bits 0 to 2 to 100
    /// assert_eq!(bb.get_bits(0, 3), Some(0b100))
    /// ```
    #[inline]
    pub fn set_bits(&mut self, index: usize, len: usize, bits: u64) {
        assert!(index + len <= self.n_bits);
        // check there are no spurious bits
        assert!(len == 64 || (bits >> len) == 0);
        assert!(len <= 64);

        if len == 0 {
            return;
        }

        // SAFETY: len <= 64 checked above
        let mask = compute_mask(len);
        let word = index >> 6;
        let pos_in_word = index & 63;

        self.data.as_mut()[word] &= !(mask << pos_in_word);
        self.data.as_mut()[word] |= bits << pos_in_word;

        let stored = 64 - pos_in_word;
        if stored < len {
            self.data.as_mut()[word + 1] &= !(mask >> stored);
            self.data.as_mut()[word + 1] |= bits >> stored;
        }
    }
}

impl BitVector<Vec<u64>> {
    /// Creates a new empty growable bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::BitVec;
    ///
    /// let bv = BitVec::new();
    /// assert_eq!(bv.len(), 0);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty bit vector with at least a capacity of `n_bits`.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::BitVec;
    ///
    /// let bv = BitVec::new();
    /// assert_eq!(bv.len(), 0);
    /// ```
    #[must_use]
    pub fn with_capacity(n_bits: usize) -> Self {
        let capacity = n_bits.div_ceil(64);
        Self {
            data: Vec::with_capacity(capacity),
            ..Self::default()
        }
    }

    /// Pushes a `bit` at the end of the bit vector.
    ///
    /// # Panics
    ///
    /// Panics if the size of the bit vector exceeds `usize::MAX` bits.
    ///
    /// # Example
    ///
    /// ```
    /// use toolkit::{BitVec, AccessBin};
    ///
    /// let mut bv = BitVec::new();
    /// bv.push(true);
    /// bv.push(false);
    /// bv.push(true);
    ///
    /// assert_eq!(bv.len(), 3);
    /// assert_eq!(bv.get(0), Some(true));
    /// assert_eq!(bv.count_ones(), 2);
    /// ```
    #[inline]
    pub fn push(&mut self, bit: bool) {
        let pos_in_word = self.n_bits % 64;
        if pos_in_word == 0 {
            self.data.push(0);
        }

        // push a 1
        if let Some(last) = self.data.last_mut() {
            *last |= (bit as u64) << pos_in_word;
        };

        self.n_bits += 1;
    }

    /// Appends `len` bits at the end of the bit vector by taking
    /// the least significant `len` bits in the u64 value `bits`.
    ///
    /// # Panics
    ///
    /// Panics if `len` is larger than 64 or if a bit of position
    /// larger than `len` is set in `bits`.
    ///
    /// Panics if the size of the bit vector exceeds `usize::MAX` bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::BitVec;
    ///
    /// let mut bv = BitVec::with_capacity(7);
    /// bv.append_bits(0b101, 3);  // appends 101
    /// bv.append_bits(0b0110, 4); // appends 0110  
    ///
    ///         
    /// assert_eq!(bv.len(), 7);
    /// assert_eq!(bv.get_bits(0, 3), Some(5));
    /// ```
    #[inline]
    pub fn append_bits(&mut self, bits: u64, len: usize) {
        assert!(len == 64 || (bits >> len) == 0);
        assert!(len <= 64);

        if len == 0 {
            return;
        }

        let pos_in_word: usize = self.n_bits & 63;
        self.n_bits += len;

        if pos_in_word == 0 {
            self.data.push(bits);
        } else if let Some(last) = self.data.last_mut() {
            *last |= bits << pos_in_word;
            if len > 64 - pos_in_word {
                self.data.push(bits >> (64 - pos_in_word));
            }
        }
    }

    /// Appends the bits of a given bit vector at the end of the current bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::{BitVec, AccessBin};
    ///
    /// let mut bv1 = BitVec::new();
    /// bv1.append_bits(0b101, 3);  // bv1 = [1,0,1]
    ///
    /// let mut bv2 = BitVec::new();
    /// bv2.append_bits(0b110, 3);  // bv2 = [0,1,1]
    ///
    /// bv1.concat(&bv2);           // bv1 = [1,0,1,0,1,1]
    ///
    /// assert_eq!(bv1.len(), 6);
    /// assert_eq!(bv1.get(0), Some(true));  // First bit from bv1
    /// assert_eq!(bv1.get(3), Some(false)); // First bit from bv2
    /// assert_eq!(bv1.get(5), Some(true));  // Last bit from bv2
    /// ```
    pub fn concat<W: AsRef<[u64]>>(&mut self, rhs: impl AsRef<BitVector<W>>) {
        let rhs = rhs.as_ref();

        if rhs.is_empty() {
            return;
        }

        let shift = self.n_bits % 64;
        let n_bits = self.n_bits + rhs.n_bits;
        let n_words = n_bits.div_ceil(64);

        if shift == 0 {
            // word-aligned, easy case
            self.data.extend(rhs.data.as_ref().iter());
        } else {
            for w in rhs.data.as_ref().iter().take(self.data.len() - 1) {
                let cur_word = self.data.last_mut().unwrap();
                *cur_word |= w << shift;
                self.data.push(w >> (64 - shift));
            }
            let cur_word = self.data.last_mut().unwrap();
            *cur_word |= *rhs.data.as_ref().last().unwrap() << shift;
            if self.data.len() < n_words {
                self.data
                    .push(*rhs.data.as_ref().last().unwrap() >> (64 - shift));
            }
        }

        self.n_bits = n_bits;
    }

    /// Extends the bit vector by adding `n` bits set to 0.
    ///
    /// # Panics
    ///
    /// Panics if the size of the bit vector exceeds `usize::MAX` bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::{BitVec, AccessBin};
    ///
    /// let mut bv = BitVec::with_capacity(10);
    /// bv.extend_with_zeros(10);
    /// assert_eq!(bv.len(), 10);
    /// assert_eq!(bv.get(8), Some(false));
    /// ```
    pub fn extend_with_zeros(&mut self, n: usize) {
        let new_size = (self.n_bits + n).div_ceil(64);
        self.data.resize_with(new_size, Default::default);
        self.n_bits += n;
    }

    /// Extends the bit vector by adding `n` bits set to 1.
    ///
    /// # Panics
    ///
    /// Panics if the size of the bit vector exceeds `usize::MAX` bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::{BitVec, AccessBin};
    ///
    /// let mut bv = BitVec::with_capacity(100);
    /// bv.extend_with_ones(100);
    /// assert_eq!(bv.len(), 100);
    /// assert_eq!(bv.get(8), Some(true));
    /// assert_eq!(bv.get(99), Some(true));
    /// ```
    pub fn extend_with_ones(&mut self, n: usize) {
        let new_size = (self.n_bits + n).div_ceil(64);
        self.data.resize_with(new_size, || u64::MAX);

        let last = n % 64;
        if last > 0 {
            *self.data.last_mut().unwrap() = u64::MAX >> (64 - last);
        }
        self.n_bits += n;
    }

    /// Shrinks the underlying vector of 64-bit words to fit the actual size of the bit vector.
    /// This can free up memory if the bit vector has a large capacity compared to its length.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::BitVec;
    ///
    /// let mut bv = BitVec::with_capacity(1000);
    /// bv.push(true);
    /// bv.push(false);
    ///
    /// // The vector may have reserved space for more bits than needed
    /// bv.shrink_to_fit(); // Free unused capacity
    ///
    /// assert_eq!(bv.len(), 2);
    /// ```
    pub fn shrink_to_fit(&mut self) {
        self.data.shrink_to_fit();
    }
}

impl<V: AsRef<[u64]> + From<Vec<u64>>> BitVector<V> {
    /// Creates a bit vector with `n_bits` set to 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::BitBoxed;
    ///
    /// let bb = BitBoxed::with_zeros(5);
    /// assert_eq!(bb.len(), 5);
    /// assert_eq!(bb.count_ones(), 0);
    /// ```
    #[must_use]
    pub fn with_zeros(n_bits: usize) -> Self {
        let n_words = n_bits.div_ceil(64);
        let data = vec![0_u64; n_words];

        BitVector {
            data: data.into(),
            n_bits,
        }
    }

    /// Creates a bit vector with `n_bits` set to 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::BitBoxed;
    ///
    /// let bb = BitBoxed::with_ones(5);
    /// assert_eq!(bb.len(), 5);
    /// assert_eq!(bb.count_ones(), 5);
    ///
    /// let bb = BitBoxed::with_ones(123);
    /// assert_eq!(bb.len(), 123);
    /// assert_eq!(bb.count_ones(), 123);
    ///
    /// let bb = BitBoxed::with_ones(128);
    /// assert_eq!(bb.len(), 128);
    /// assert_eq!(bb.count_ones(), 128);
    /// ```
    #[must_use]
    pub fn with_ones(n_bits: usize) -> Self {
        let n_words = n_bits.div_ceil(64);
        let last_word = n_bits & 63;
        let mut data = vec![u64::MAX; n_words - 1];
        data.push(if last_word == 0 {
            u64::MAX
        } else {
            (1_u64 << last_word) - 1
        });

        BitVector {
            data: data.into(),
            n_bits,
        }
    }
}

impl Extend<bool> for BitVector<Vec<u64>> {
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = bool>,
    {
        for bit in iter {
            self.push(bit);
        }
    }

    /* Nigthly
        fn extend_one(&mut self, item: bool) {
            self.push(item);
        }
        fn extend_reserve(&mut self, additional: usize) {
            self.data.reserve
        }
    */
}

/// Extends a `BitVector` with an iterator over `usize` values.
///
/// # Examples
///
/// ```
/// use toolkit::{BitVec, AccessBin};
///
/// let mut bv = BitVec::new();
///
/// // Extending the bit vector with a range of positions
/// bv.extend(0..5);
/// assert_eq!(bv.len(), 5);
/// assert_eq!(bv.get(3), Some(true));
/// ```
impl Extend<usize> for BitVector<Vec<u64>> {
    fn extend<T>(&mut self, iter: T)
    where
        T: IntoIterator<Item = usize>,
    {
        for pos in iter {
            if pos >= self.n_bits {
                self.extend_with_zeros(pos + 1 - self.n_bits);
            }
            self.set(pos, true);
        }
    }
}

// impl SpaceUsage for BitVector {
//     /// Returns the space usage in bytes.
//     #[must_use]
//     fn space_usage_byte(&self) -> usize {
//         self.data.space_usage_byte() + 8
//     }
// }

/// Creates a `BitVector` from an iterator over `bool` values.
///
/// # Examples
///
/// ```
/// use toolkit::{AccessBin, BitVec};
///
/// // Create a bit vector from an iterator over bool values
/// let bv: BitVec = vec![true, false, true].into_iter().collect();
///
/// assert_eq!(bv.len(), 3);
/// assert_eq!(bv.get(1), Some(false));
/// ```
impl FromIterator<bool> for BitVector<Vec<u64>> {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = bool>,
    {
        let mut bv = BitVec::default();
        bv.extend(iter);

        bv
    }
}

impl FromIterator<bool> for BitVector<Box<[u64]>> {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = bool>,
    {
        BitVector::<Vec<u64>>::from_iter(iter).into()
    }
}

// it contains all the type of num_traits::int::PrimInt without bool
pub trait MyPrimInt: TryInto<usize> {}

macro_rules! impl_my_prim_int {
    ($($t:ty),*) => {
        $(impl MyPrimInt for $t {
        })*
    }
}

impl_my_prim_int![
    i8, u8, i16, u16, i32, u32, i64, u64, isize, usize, u128, i128
];

/// Creates a `BitVector` from an iterator over non-negative integer values.
///
/// # Panics
/// Panics if any value of the sequence cannot be converted to usize.
///
/// # Examples
///
/// ```
/// use toolkit::{AccessBin, BitVec};
///
/// // Create a bit vector from an iterator over usize values
/// let bv: BitVec = vec![0, 1, 3, 5].into_iter().collect();
///
/// assert_eq!(bv.len(), 6);
/// assert_eq!(bv.get(3), Some(true));
/// ```
impl<V> FromIterator<V> for BitVector<Vec<u64>>
where
    V: MyPrimInt,
    <V as TryInto<usize>>::Error: std::fmt::Debug,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = V>,
        <V as TryInto<usize>>::Error: std::fmt::Debug,
    {
        let mut bv = BitVector::<Vec<u64>>::default();
        bv.extend(
            iter.into_iter()
                .map(|x| x.try_into().expect("Cannot a value convert to usize")),
        );

        bv
    }
}

impl<V> FromIterator<V> for BitVector<Box<[u64]>>
where
    V: MyPrimInt,
    <V as TryInto<usize>>::Error: std::fmt::Debug,
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = V>,
        <V as TryInto<usize>>::Error: std::fmt::Debug,
    {
        BitVector::<Vec<u64>>::from_iter(iter).convert_into()
    }
}

/// Implements conversion from mutable `BitVector` to an immutable one.
///
/// This conversion consumes the original mutable `BitVector` and creates an
/// immutable version.
///
/// # Examples
///
/// ```
/// use toolkit::{BitVec,BitBoxed, AccessBin};
///
/// let mut bvm = BitVec::new();
/// bvm.push(true);
/// bvm.push(false);
///
/// // Convert mutable BitVector to immutable BitVector
/// let bv: BitBoxed = bvm.into();
///
/// assert_eq!(bv.get(0), Some(true));
/// ```
impl From<BitVector<Vec<u64>>> for BitVector<Box<[u64]>> {
    fn from(bvm: BitVector<Vec<u64>>) -> Self {
        Self {
            data: bvm.data.into_boxed_slice(),
            n_bits: bvm.n_bits,
        }
    }
}

/// Implements conversion from an immutable `BitVector` to a mutable one.
///
/// This conversion takes ownership of the original `BitVector` and creates a mutable version.
///
/// # Examples
///
/// ```
/// use toolkit::{BitVec, BitBoxed, AccessBin};
///
/// let v = vec![0,2,3,4,5];
/// let mut bv: BitBoxed = v.into_iter().collect();
///
/// let mut bvm: BitVec = bv.into();
///
/// assert_eq!(bvm.get(0), Some(true));
/// assert_eq!(bvm.len(), 6);
/// bvm.push(true);
/// assert_eq!(bvm.len(), 7);
/// ```
impl From<BitVector<Box<[u64]>>> for BitVector<Vec<u64>> {
    fn from(bv: BitVector<Box<[u64]>>) -> Self {
        Self {
            data: bv.data.into(),
            n_bits: bv.n_bits,
        }
    }
}

impl From<BitVector<&[u64]>> for BitVector<Vec<u64>> {
    fn from(bv: BitVector<&[u64]>) -> Self {
        Self {
            data: bv.data.into(),
            n_bits: bv.n_bits,
        }
    }
}

impl<V: AsRef<[u64]>> AsRef<BitVector<V>> for BitVector<V> {
    fn as_ref(&self) -> &BitVector<V> {
        self
    }
}

#[derive(Debug)]
pub struct BitVectorBitPositionsIter<'a, const BIT: bool> {
    bs: BitSliceWithOffset<'a>,
    cur_position: usize, // Current position in the bit vector
}

impl<'a, const BIT: bool> BitVectorBitPositionsIter<'a, BIT> {
    #[must_use]
    #[inline]
    pub fn new(bs: BitSliceWithOffset<'a>) -> Self {
        Self::with_pos(bs, 0)
    }

    #[must_use]
    #[inline]
    pub fn with_pos(bs: BitSliceWithOffset<'a>, pos: usize) -> Self {
        Self {
            bs,
            cur_position: pos,
        }
    }
}

impl<'a, const BIT: bool> BitVectorBitPositionsIter<'a, BIT> {
    /// If bits == 0, return 0
    #[must_use]
    #[inline]
    pub fn get_bits(&mut self, bits: usize) -> Option<u64> {
        if bits > 64 || self.cur_position + bits > self.bs.n_bits {
            return None;
        }

        // SAFETY: the check self.cur_position + bits <= self.n_bits guarntees
        // that cur_word_pos is in bounds while filling the buffer in unsafe get_bits_unchecked

        Some(unsafe { self.get_bits_unchecked(bits) })
    }

    /// Returns `len` bits from the current position without bounds checking and advances the position.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `len` is valid and that the current position plus `len` does not exceed the bounds of the bit slice.
    /// Invoking this function with invalid arguments is undefined behavior.
    #[must_use]
    #[inline]
    pub unsafe fn get_bits_unchecked(&mut self, len: usize) -> u64 {
        let v = unsafe { self.bs.get_bits_unchecked(self.cur_position, len) };
        self.cur_position += len;

        v
    }
}

/// Iterator over the positions of bits set to BIT (false for zeros,
/// true for ones) in the bit vector.
impl<'a, const BIT: bool> Iterator for BitVectorBitPositionsIter<'a, BIT> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_position >= self.bs.n_bits {
            return None;
        }

        let p = if BIT {
            unsafe { self.bs.next_one_unchecked(self.cur_position) }
        } else {
            unsafe { self.bs.next_zero_unchecked(self.cur_position) }
        };

        if let Some(pos) = p {
            if pos < self.bs.n_bits {
                self.cur_position = pos + 1;
                return Some(pos);
            }
        }

        None
    }
}

pub struct BitVectorIter<V: AsRef<[u64]>, T: AsRef<BitVector<V>>> {
    bv: T,
    n_bits: usize,
    i: usize,
    _phantom: std::marker::PhantomData<V>,
}

impl<V: AsRef<[u64]>, T: AsRef<BitVector<V>>> ExactSizeIterator for BitVectorIter<V, T> {
    fn len(&self) -> usize {
        self.bv.as_ref().n_bits - self.i
    }
}

impl<V: AsRef<[u64]>, T: AsRef<BitVector<V>>> Iterator for BitVectorIter<V, T> {
    type Item = bool;
    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.n_bits {
            self.i += 1;
            Some(unsafe { self.bv.as_ref().get_unchecked(self.i - 1) })
        } else {
            None
        }
    }
}

impl<V: AsRef<[u64]>> IntoIterator for BitVector<V> {
    type IntoIter = BitVectorIter<V, BitVector<V>>;
    type Item = bool;

    fn into_iter(self) -> Self::IntoIter {
        let n_bits = self.as_ref().n_bits;
        BitVectorIter {
            bv: self,
            i: 0,
            n_bits,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, V: AsRef<[u64]>> IntoIterator for &'a BitVector<V> {
    type IntoIter = BitVectorIter<V, &'a BitVector<V>>;
    type Item = bool;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<V: AsRef<[u64]>> std::fmt::Debug for BitVector<V> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        let data_str: Vec<String> = self
            .data
            .as_ref()
            .iter()
            .map(|x| format!("{:b}", x))
            .collect();
        write!(
            fmt,
            "BitVector {{ n_bits:{:?}, data:{:?}}}",
            self.n_bits, data_str
        )
    }
}

// The bit vector may start at an offset (in bits) in the first word (i.e., the first word may contain some bits that are not part of the bit vector). This is useful for the implementation of the [`BitVecCollection`] where we concatenate several binary vectors and we want to avoid padding.
#[derive(Debug, Default, Clone, Eq, PartialEq)]
pub struct BitSliceWithOffset<'a> {
    data: &'a [u64],
    n_bits: usize,
    offset: usize,
}

impl<'a> BitSliceWithOffset<'a> {
    /// `offset` is any bit position in the bit vector (i.e., offset < n_bits).
    /// # Examples
    ///
    /// ```
    /// use toolkit::BitVec;
    /// use toolkit::BitSliceWithOffset;
    ///
    /// let v = vec![0b000001010, 0b01010111000000, u64::MAX];
    ///
    /// // Bitslice with offset that excludes the first 64 + 5 bits
    /// let offset = 5;
    /// let bswo = unsafe{ BitSliceWithOffset::from_raw_parts(&v[1..], 59+64, offset)};
    ///
    /// assert_eq!(bswo.len(), 59+64);
    /// assert_eq!(bswo.get_bits(0, 4), Some(0b1110));
    /// ```
    pub fn new<V: AsRef<[u64]>>(bv: &'a BitVector<V>, offset: usize) -> Self {
        if offset > bv.n_bits {
            return BitSliceWithOffset::default();
        }

        let p = offset / 64;
        let data = &bv.data.as_ref()[p..];
        let n_bits = bv.n_bits - offset;
        let offset = offset % 64;

        Self {
            data,
            n_bits,
            offset,
        }
    }

    /// Creates a new BitSliceWithOffset from raw parts without checking validity.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the provided data slice is valid for the given `n_bits` and `offset`.
    /// Invoking this function with invalid arguments is undefined behavior.
    #[inline]
    pub unsafe fn from_raw_parts(data: &'a [u64], n_bits: usize, offset: usize) -> Self {
        Self {
            data,
            n_bits,
            offset,
        }
    }

    /// Accesses `len` bits, with 0 <= `len` <= 64, starting at position `index`.
    ///
    /// Returns [`None`] if `index`+`len` is out of bounds or if `len` is greater than 64.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::BitVec;
    /// use toolkit::BitSliceWithOffset;
    ///
    /// let v = vec![0b000001010, 0b01010111000000, u64::MAX];
    ///
    /// // Bitslice with offset that excludes the first 64 + 5 bits
    /// let offset = 5;
    /// let bswo = unsafe{ BitSliceWithOffset::from_raw_parts(&v[1..], 59+64, offset)};
    ///
    /// assert_eq!(bswo.len(), 59+64);
    /// assert_eq!(bswo.get_bits(0, 4), Some(0b1110));
    /// assert_eq!(bswo.get_bits(bswo.len()-2, 1), Some(1));
    /// assert_eq!(bswo.get_bits(bswo.len()-2, 0), Some(0));
    ///
    /// ```
    #[must_use]
    #[inline]
    pub fn get_bits(&self, index: usize, len: usize) -> Option<u64> {
        if (len > 64) | (index + len > self.n_bits) {
            return None;
        }
        // SAFETY: safe access due to the above checks
        Some(unsafe { self.get_bits_unchecked(index, len) })
    }

    /// Accesses `len` bits, starting at position `index`, without performing bounds checking.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it does not perform bounds checking.
    /// It is the caller's responsibility to ensure that the provided `index` and `len`
    /// are within the bounds of the bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::BitVec;
    /// use toolkit::BitSliceWithOffset;
    ///
    /// let v = vec![0b000001010, 0b01010111000000, u64::MAX];
    ///
    /// // Bitslice with offset that excludes the first 64 + 5 bits
    /// let offset = 5;
    /// let bswo = unsafe{ BitSliceWithOffset::from_raw_parts(&v[1..], 59+64, offset)};
    ///
    /// assert_eq!(unsafe{bswo.get_bits_unchecked(0, 4)}, 0b1110);
    /// assert_eq!(unsafe{bswo.get_bits_unchecked(0, 0)}, 0);
    ///
    /// ```
    #[must_use]
    #[inline]
    pub unsafe fn get_bits_unchecked(&self, index: usize, len: usize) -> u64 {
        debug_assert!(index + len <= self.n_bits, "Index out of bounds");
        unsafe { BitVector::<&[u64]>::get_bits_slice(self.data, index + self.offset, len) }
    }

    pub fn next_one(&self, index: usize) -> Option<usize> {
        if index >= self.n_bits {
            return None;
        }
        // SAFETY: safe access due to the above checks
        let p = unsafe { self.next_one_unchecked(index) };
        if let Some(pos) = p {
            if pos < self.n_bits {
                return Some(pos);
            }
        }
        None
    }

    pub unsafe fn next_one_unchecked(&self, index: usize) -> Option<usize> {
        if let Some(pos) = unsafe {
            BitVector::<&[u64]>::next_bit_slice_unchecked::<true>(
                self.data,
                index + self.offset,
                self.n_bits + self.offset,
            )
        } {
            return Some(pos - self.offset);
        }

        None
    }

    pub fn next_zero(&self, index: usize) -> Option<usize> {
        if index >= self.n_bits {
            return None;
        }
        // SAFETY: safe access due to the above checks
        let p = unsafe { self.next_zero_unchecked(index) };

        if let Some(pos) = p {
            if pos < self.n_bits {
                return Some(pos);
            }
        }
        None
    }

    pub unsafe fn next_zero_unchecked(&self, index: usize) -> Option<usize> {
        if let Some(pos) = unsafe {
            BitVector::<&[u64]>::next_bit_slice_unchecked::<false>(
                self.data,
                index + self.offset,
                self.n_bits + self.offset,
            )
        } {
            return Some(pos - self.offset);
        }
        None
    }
    /// Returns a non-consuming iterator over positions of bits set to 1 in the bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::BitVec;
    /// use toolkit::BitSliceWithOffset;
    ///
    /// let v = vec![0b000001010, 0b01010111000000, u64::MAX];
    ///
    /// // Bitslice with offset that excludes the first 64 + 5 bits
    /// let offset = 5;
    /// let bswo = unsafe{ BitSliceWithOffset::from_raw_parts(&v[1..], 59+64, offset)};
    /// let mut v = vec![1, 2, 3, 5, 7];
    /// v.extend(59..(59+64));
    /// assert_eq!(bswo.ones().collect::<Vec<_>>(), v);
    /// ```
    #[must_use]
    pub fn ones(&self) -> BitVectorBitPositionsIter<true> {
        BitVectorBitPositionsIter::with_pos(self.clone(), 0)
    }

    /// Returns a non-consuming iterator over positions of bits set to 1 in the bit vector, starting at a specified bit position.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::BitVec;
    /// use toolkit::BitSliceWithOffset;
    ///
    /// let v = vec![0b000001010, 0b01010111000000, u64::MAX];
    ///
    /// // Bit slice with offset that excludes the first 64 + 5 bits
    /// let offset = 5;
    /// let bswo = unsafe{ BitSliceWithOffset::from_raw_parts(&v[1..], 59+64, offset)};
    /// let mut v = vec![5, 7];
    /// v.extend(59..(59+64));
    /// assert_eq!(bswo.ones_with_pos(5).collect::<Vec<_>>(), v);
    /// ```
    #[must_use]
    pub fn ones_with_pos(&self, pos: usize) -> BitVectorBitPositionsIter<true> {
        BitVectorBitPositionsIter::with_pos(self.clone(), pos)
    }

    /// Returns a non-consuming iterator over positions of bits set to 0 in the bit vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::BitVec;
    /// use toolkit::gen_sequences::negate_vector;
    ///
    /// let vv: Vec<usize> = vec![0, 63, 128, 129, 254, 1026];
    /// let bv: BitVec = vv.iter().copied().collect();
    ///
    /// let v: Vec<usize> = bv.zeros().collect();
    /// assert_eq!(v, negate_vector(&vv));
    /// ```
    #[must_use]
    pub fn zeros(&self) -> BitVectorBitPositionsIter<false> {
        BitVectorBitPositionsIter::with_pos(self.clone(), 0)
    }

    /// Returns a non-consuming iterator over positions of bits set to 0 in the bit vector, starting at a specified bit position.
    #[must_use]
    pub fn zeros_with_pos(&self, pos: usize) -> BitVectorBitPositionsIter<false> {
        BitVectorBitPositionsIter::with_pos(self.clone(), pos)
    }

    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.n_bits
    }

    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl AccessBin for BitSliceWithOffset<'_> {
    #[inline]
    fn get(&self, index: usize) -> Option<bool> {
        if index >= self.n_bits {
            return None;
        }
        Some(unsafe { self.get_unchecked(index) })
    }

    unsafe fn get_unchecked(&self, index: usize) -> bool {
        debug_assert!(index < self.n_bits, "Index out of bounds");
        unsafe { BitVector::<&[u64]>::get_bit_slice(self.data, index + self.offset) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen_sequences::{gen_strictly_increasing_sequence, negate_vector};

    #[test]
    fn test_is_empty() {
        let bv = BitVec::default();
        assert!(bv.is_empty());
    }

    // Build a bit vector of size n with even positions set to one
    // and odd ones to zero
    fn build_alternate(n: usize) -> BitVec {
        let mut bv = BitVec::with_capacity(n);
        for i in 0..n {
            bv.push(i % 2 == 0);
        }
        bv
    }

    #[test]
    fn test_get() {
        let n = 1024 + 13;
        let bv = build_alternate(n);

        for i in 0..n {
            assert_eq!(bv.get(i).unwrap(), i % 2 == 0);
        }
    }

    #[test]
    fn test_iter() {
        let n = 1024 + 13;
        let bv: BitVec = build_alternate(n).into();

        for (i, bit) in bv.into_iter().enumerate() {
            assert_eq!(bit, i % 2 == 0);
        }
    }

    #[test]
    fn test_get_set_bits() {
        let n = 1024 + 13;
        let mut bv = BitVec::new();
        bv.extend_with_zeros(n);

        assert_eq!(bv.get_bits(61, 35).unwrap(), 0);
        assert_eq!(bv.get_bits(0, 42).unwrap(), 0);
        assert_eq!(bv.get_bits(n - 42 - 1, 42).unwrap(), 0);
        assert_eq!(bv.get_bits(n - 42, 42).unwrap(), 0);
        assert_eq!(bv.get_bits(n - 1, 1).unwrap(), 0);
        assert_eq!(bv.get_bits(n - 42, 43), None);
        bv.set_bits(0, 6, 42);
        assert_eq!(bv.get_bits(0, 6).unwrap(), 42);
        bv.set_bits(n - 61 - 1, 61, 42);
        assert_eq!(bv.get_bits(n - 61 - 1, 61).unwrap(), 42);
        bv.set_bits(n - 67 - 1, 33, 42);
        assert_eq!(bv.get_bits(n - 67 - 1, 33).unwrap(), 42);
    }

    #[test]
    fn test_from_iter() {
        let n = 1024 + 13;
        let bv = build_alternate(n);

        let bv2: BitVec = (0..n).map(|x| x % 2 == 0).collect();

        assert_eq!(bv, bv2);

        /* Note: if last bits are zero, the bit vector may differ
        because we are inserting only position of ones */
        let bv2: BitVec = (0..n).filter(|x| x % 2 == 0).collect();

        assert_eq!(bv, bv2);
    }

    #[test]
    fn test_next_one_and_zero() {
        let n = 1024 + 13;
        let bv = BitVec::with_ones(n);

        for i in 0..n {
            assert_eq!(bv.next_one(i).unwrap(), i);
        }
        assert_eq!(bv.next_one(n), None);

        let v = vec![
            1, 129, 193, 257, 321, 385, 449, 513, 577, 641, 705, 769, 833, 897, 961,
        ];

        let bv = BitVec::from_iter(v.iter().copied());

        let mut prev_pos = 0;
        for &p in v.iter() {
            assert_eq!(bv.next_one(prev_pos).unwrap(), p);
            prev_pos = p + 1;
        }
        assert_eq!(bv.next_one(*v.last().unwrap() + 1), None);

        for offset in [10, 64, 123, 961] {
            let bswo = BitSliceWithOffset::new(&bv, offset);

            assert_eq!(bswo.len(), bv.len() - offset);

            let mut prev_pos = 0;
            for p in v.iter().filter(|&&x| x >= offset).map(|&x| x - offset) {
                assert_eq!(bswo.next_one(prev_pos).unwrap(), p);
                prev_pos = p + 1;
            }
            assert_eq!(bswo.next_one(*v.last().unwrap() - offset + 1), None);
        }

        let bv = BitVec::from_iter(v.iter().copied());

        let v = negate_vector(&v);

        let mut prev_pos = 0;
        for &p in v.iter() {
            assert_eq!(bv.next_zero(prev_pos).unwrap(), p);
            prev_pos = p + 1;
        }
        assert_eq!(bv.next_zero(*v.last().unwrap() + 1), None);
    }

    #[test]
    fn test_get_bits_iter() {
        for n_bits in 3..4 {
            let mut bv = BitVec::new();
            let max = 1 << n_bits;

            for i in 0..1024 {
                bv.append_bits(i % max, n_bits);
            }

            let mut iter = bv.ones();
            for i in 0..1024 {
                assert_eq!(iter.get_bits(n_bits), Some(i % max));
            }
        }
    }

    #[test]
    fn test_get_bits_iter_2() {
        let bv = BitVec::from_iter(vec![0, 63, 128, 129, 254, 1026]);

        for n_bits in 1..64 {
            let mut iter = bv.ones();
            for position in (0..bv.len() - n_bits).step_by(n_bits) {
                assert_eq!(iter.get_bits(n_bits), bv.get_bits(position, n_bits));
            }
        }
    }

    #[test]
    fn test_iter_zeros() {
        let bv = BitVec::default();
        let v: Vec<usize> = bv.zeros().collect();
        assert!(v.is_empty());

        let vv: Vec<usize> = vec![0, 63, 128, 129, 254, 1026];
        let bv: BitVec = vv.iter().copied().collect();

        let v: Vec<usize> = bv.zeros().collect();
        assert_eq!(v, negate_vector(&vv));

        let v: Vec<usize> = bv.zeros_with_pos(63).collect();
        assert_eq!(v[0], 64);
        assert_eq!(*v.last().unwrap(), 1025);
    }

    #[test]
    fn test_iter_ones() {
        let bv = BitVec::default();
        let v: Vec<usize> = bv.ones().collect();
        assert!(v.is_empty());

        let vv: Vec<usize> = vec![0, 63, 128, 129, 254, 1026];
        let bv: BitVec = vv.iter().copied().collect();

        let v: Vec<usize> = bv.ones().collect();
        assert_eq!(v, vv);

        let v: Vec<usize> = bv.ones_with_pos(127).collect();
        assert_eq!(v, vec![128, 129, 254, 1026]);

        let v: Vec<usize> = bv.ones_with_pos(129).collect();
        assert_eq!(v, vec![129, 254, 1026]);

        let v: Vec<usize> = bv.ones_with_pos(130).collect();
        assert_eq!(v, vec![254, 1026]);

        let v: Vec<usize> = bv.ones_with_pos(1027).collect();
        assert_eq!(v, vec![]);

        let vv: Vec<usize> = (0..1024).collect();
        let bv: BitVec = vv.iter().copied().collect();
        let v: Vec<usize> = bv.ones().collect();
        assert_eq!(v, vv);

        let vv = gen_strictly_increasing_sequence(1024 * 4, 1 << 20);

        let bv: BitVec = vv.iter().copied().collect();
        let v: Vec<usize> = bv.ones().collect();
        assert_eq!(v, vv);
    }

    #[test]
    fn test_concat() {
        let mut bv1 = BitVec::new();
        bv1.push(true);
        bv1.push(false);

        let mut bv2 = BitVec::new();
        bv2.push(true);
        bv2.push(true);

        bv1.concat(&bv2);

        assert_eq!(bv1.len(), 4);
        assert_eq!(bv1.get(2), Some(true));

        let vv: Vec<usize> = vec![0, 63, 128, 129, 254, 1026];
        let mut bv1: BitVec = vv.iter().copied().collect();
        let bv2: BitVec = vv.iter().copied().collect();
        bv1.concat(bv2);
        assert_eq!(bv1.len(), 2054);
        assert_eq!(bv1.get(1026), Some(true));
        assert_eq!(bv1.get(2053), Some(true));
        assert_eq!(bv1.get(2054), None);
    }
}
