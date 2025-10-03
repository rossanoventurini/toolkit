#![feature(unchecked_shifts)]
#![feature(generic_const_exprs)]
//! This module provides a collection of bit manipulation utilities, including
//! bit vectors, bit fields, and various utility functions for working with bits.
//! This is a preliminary version and may undergo changes in the next future.

pub mod bitvector;
// pub use bitvector::bitvector_collection::{BitBoxedCollection, BitVecCollection};
pub use bitvector::{BitBoxed, BitSlice, BitSliceWithOffset, BitVec};

pub mod bitfield;
pub use bitfield::{BitFieldBoxed, BitFieldSlice, BitFieldVec};

pub mod darray;
pub use darray::DArray;

pub mod elias_fano;
pub use elias_fano::EliasFano;
pub use elias_fano::EliasFanoBuilder;

pub mod stream_vbyte;
pub use stream_vbyte::StreamVByte;

pub use stream_vbyte::StreamVByteIter;
pub use stream_vbyte::StreamVByteRandomAccess;

pub mod gen_sequences;

pub mod algorithms;

pub mod utils;

/// A trait for the support of `get` query over the binary alphabet.
pub trait AccessBin {
    /// Returns the bit at the given position `i`,
    /// or [`None`] if ```i``` is out of bounds.
    fn get(&self, i: usize) -> Option<bool>;

    /// Returns the bit at the given position `i`.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    unsafe fn get_unchecked(&self, i: usize) -> bool;
}

/// A trait for the support of `rank` query over the binary alphabet.
pub trait RankBin {
    /// Returns the number of zeros in the indexed sequence up to
    /// position `i` excluded.
    #[inline]
    fn rank0(&self, i: usize) -> Option<usize> {
        if let Some(k) = self.rank1(i) {
            return Some(i - k);
        }

        None
    }

    /// Returns the number of ones in the indexed sequence up to
    /// position `i` excluded.
    fn rank1(&self, i: usize) -> Option<usize>;

    /// Returns the number of ones in the indexed sequence up to
    /// position `i` excluded. `None` if the position is out of bound.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    unsafe fn rank1_unchecked(&self, i: usize) -> usize;

    /// Returns the number of zeros in the indexed sequence up to
    /// position `i` excluded.
    ///
    /// # Safety
    /// Calling this method with an out-of-bounds index is undefined behavior.
    #[inline]
    unsafe fn rank0_unchecked(&self, i: usize) -> usize {
        i - unsafe { self.rank1_unchecked(i) }
    }

    fn n_zeros(&self) -> usize;
}

/// A trait for the support of `select` query over the binary alphabet.
pub trait SelectBin {
    /// Returns the position of the `i+1`-th occurrence of a bit set to `1`.
    /// Returns `None` if there is no such position.
    fn select1(&self, i: usize) -> Option<usize>;

    /// Returns the position of the `i+1`-th occurrence of a bit set to `1`.
    ///
    /// # Safety
    /// This method doesn't check that such element exists
    /// Calling this method with an i >= maximum rank1 is undefined behaviour.
    unsafe fn select1_unchecked(&self, i: usize) -> usize;

    /// Returns the position of the `i+1`-th occurrence of a bit set to `0`.
    /// Returns `None` if there is no such position.
    fn select0(&self, i: usize) -> Option<usize>;

    /// Returns the position of the `i+1`-th occurrence of a bit set to  `0`.
    ///
    /// # Safety
    /// This method doesnt check that such element exists
    /// Calling this method with an `i >= maximum rank0` is undefined behaviour.
    unsafe fn select0_unchecked(&self, i: usize) -> usize;
}
