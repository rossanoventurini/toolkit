#![feature(unchecked_shifts)]

pub mod bitvector;
// pub use bitvector::bitvector_collection::{BitBoxedCollection, BitVecCollection};
pub use bitvector::{BitBoxed, BitSlice, BitSliceWithOffset, BitVec};

pub mod gen_sequences;

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
