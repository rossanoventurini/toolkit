#![feature(unchecked_shifts)]
//! This module provides a collection of bit manipulation utilities, including
//! bit vectors, bit fields, and various utility functions for working with bits.
//! This is a preliminary version and may undergo changes in the next future.

pub mod bitvector;
// pub use bitvector::bitvector_collection::{BitBoxedCollection, BitVecCollection};
pub use bitvector::{BitBoxed, BitSlice, BitSliceWithOffset, BitVec};

pub mod bitfield;
pub use bitfield::{BitField, BitFieldBoxed, BitFieldSlice};

pub mod gen_sequences;

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
