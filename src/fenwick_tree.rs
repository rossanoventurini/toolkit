use num::Zero;
use std::ops::{AddAssign, Bound, RangeBounds, SubAssign};

use serde::{Deserialize, Serialize};
use std::iter::FusedIterator;

/// A Fenwick Tree (also known as Binary Indexed Tree or BIT) for efficient prefix sum queries
/// and point updates on a mutable sequence.
///
/// # Overview
///
/// A Fenwick Tree maintains a sequence of values supporting two main operations:
/// - **Point update**: Add or subtract a value at a specific index in Θ(log n) time
/// - **Prefix sum**: Compute the sum of elements in a range in Θ(log n) time
///
/// This implementation also provides Θ(n) iteration over the prefix sums or the original elements using
/// a stack-based algorithm that incrementally updates the Fenwick tree path.
///
/// # Generic Parameters
///
/// - `T`: The type of elements stored. Must implement `Copy`, `Zero`, `AddAssign`, and `SubAssign`.
/// - `HOLES`: A const boolean flag controlling the internal layout:
///   - `true` (default): Uses a "holes" layout with extra space (n + n/2^14) to improve cache locality
///   - `false`: Uses a compact layout with exactly n elements
///
/// # Complexity
///
/// - **Construction**: Θ(n) using `from()` or `FromIterator`
/// - **Point update** (`add_at`, `sub_at`): Θ(log n)
/// - **Range sum** (`sum`): Θ(log n)
/// - **Iteration** (`iter`, `prefix_sums`): Θ(n) total (amortized O(1) per element)
///
/// # Examples
///
/// ```
/// use toolkit::FenwickTree;
///
/// // Create from a vector
/// let mut ft = FenwickTree::<i32, false>::from(vec![1, 2, 3, 4, 5]);
///
/// // Query prefix sums
/// assert_eq!(ft.sum(0..3), 6);  // sum of [1, 2, 3]
/// assert_eq!(ft.sum(1..=3), 9); // sum of [2, 3, 4]
///
/// // Update values
/// ft.add_at(2, 10); // Add 10 to index 2
/// assert_eq!(ft.sum(0..3), 16); // Now sum is 1 + 2 + 13 = 16
///
/// // Iterate over elements
/// let elements: Vec<i32> = ft.iter().collect();
/// assert_eq!(elements, vec![1, 2, 13, 4, 5]);
///
/// // Iterate over prefix sums
/// let prefix_sums: Vec<i32> = ft.prefix_sums().collect();
/// assert_eq!(prefix_sums, vec![1, 3, 16, 20, 25]);
/// ```
///
/// # Implementation Notes
///
/// This Fenwick Tree uses 0-based indexing externally, but internally converts to 1-based
/// indexing for the tree operations. The tree structure is based on the binary representation
/// of indices, where each node stores the sum of a range determined by the lowest set bit.
///
/// When `HOLES = true`, the index mapping `index(i) = i + (i >> 14)` introduces gaps in the
/// array to improve cache performance on modern CPUs by reducing cache line conflicts.
///
/// # Performance Benchmarks
///
/// ## Point Update and Range Query Operations
///
/// Performance of `add_at()` and `sum()` operations on Intel Core Ultra 7 265K @ 5.6 GHz:
///
/// | Elements       | `add_at()` (HOLES=true) | `add_at()` (HOLES=false) | `sum()` (HOLES=true) | `sum()` (HOLES=false) |
/// |----------------|-------------------------|--------------------------|----------------------|-----------------------|
/// | 1,000          | 6 ns                    | 6 ns                     | 7 ns                 | 6 ns                  |
/// | 100,000        | 12 ns                   | 12 ns                    | 9 ns                 | 8 ns                  |
/// | 10,000,000     | 49 ns                   | 59 ns                    | 42 ns                | 38 ns                 |
/// | 1,000,000,000  | 133 ns                  | 174 ns                   | 92 ns                | 105 ns                |
///
/// HOLES=true provides better performance for very large datasets due to improved cache locality.
///
/// ## Iterator Performance
///
/// Time per element for `iter()` and `prefix_sums()` on Intel Core Ultra 7 265K @ 5.6 GHz:
///
/// | Elements       | `iter()` (HOLES=true) | `iter()` (HOLES=false) | `prefix_sums()` (HOLES=true) | `prefix_sums()` (HOLES=false) |
/// |----------------|-----------------------|------------------------|------------------------------|-------------------------------|
/// | 1,000          | 2 ns                  | >1 ns                  | 2 ns                         | 2 ns                          |
/// | 100,000        | >1 ns                 | >1 ns                  | 2 ns                         | >1 ns                         |
/// | 10,000,000     | 5 ns                  | 5 ns                   | 5 ns                         | 5 ns                          |
/// | 1,000,000,000  | 5 ns                  | 5 ns                   | 5 ns                         | 5 ns                          |
///
/// **Test Configuration:**
/// - CPU: Intel Core Ultra 7 265K (20 cores, max 5.6 GHz)
/// - Cache: L1d: 704 KiB, L1i: 1.1 MiB, L2: 36 MiB, L3: 30 MiB
/// - Compiler: rustc with `RUSTFLAGS='-C target-cpu=native'`
///
/// The Θ(n) total complexity is confirmed: iteration time scales linearly with the number of elements,
/// with amortized constant time per element ranging from <1 ns to ~5 ns depending on dataset size and
/// cache effects.
#[derive(Default, Clone, Eq, PartialEq, Debug, Serialize, Deserialize)]
pub struct FenwickTree<T, const HOLES: bool = true>
where
    T: Copy + Zero + AddAssign + SubAssign,
{
    tree: Box<[T]>,
    n: usize,
}

impl<T, const HOLES: bool> FenwickTree<T, HOLES>
where
    T: Copy + Zero + AddAssign + SubAssign,
{
    /// Creates a new Fenwick Tree with `n` elements, all initialized to zero.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of elements in the sequence
    ///
    /// # Panics
    ///
    /// Panics if `n` is too large to allocate the required memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::FenwickTree;
    ///
    /// let ft = FenwickTree::<i32, false>::with_len(100);
    /// assert_eq!(ft.len(), 100);
    /// assert_eq!(ft.sum(0..100), 0); // All elements are zero
    /// ```
    #[must_use]
    pub fn with_len(n: usize) -> Self {
        Self {
            tree: vec![T::zero(); Self::size(n)].into_boxed_slice(),
            n,
        }
    }

    /// Subtracts a value from the element at index `i`.
    ///
    /// This operation updates the tree structure in Θ(log n) time.
    ///
    /// # Arguments
    ///
    /// * `i` - The index to update (0-based)
    /// * `v` - The value to subtract
    ///
    /// # Panics
    ///
    /// Panics if `i >= n`.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::FenwickTree;
    ///
    /// let mut ft = FenwickTree::<i32, false>::from(vec![10, 20, 30]);
    /// ft.sub_at(1, 5);
    /// assert_eq!(ft.sum(0..=1), 25); // 10 + (20 - 5)
    /// ```
    #[inline(always)]
    pub fn sub_at(&mut self, i: usize, v: T) {

        assert!(i < self.n);

        let mut i = i;
        while i < self.n {
            let pos = if HOLES { Self::index(i) } else { i };
            self.tree[pos] -= v;
            i = Self::next(i);
        }
    }

    /// Returns the index of the parent node in the Fenwick tree (1-indexed).
    ///
    /// This is computed by clearing the lowest set bit: `i & (i - 1)`.
    #[inline(always)]
    const fn prev(i: usize) -> usize {
        i & (i - 1)
    }

    /// Returns the index of the next node to update in the Fenwick tree (0-indexed).
    ///
    /// This is computed by setting the lowest unset bit: `i | (i + 1)`.
    #[inline(always)]
    const fn next(i: usize) -> usize {
        i | (i + 1)
    }

    /// Maps a logical index to a physical index in the tree when `HOLES = true`.
    ///
    /// The mapping `i + (i >> 14)` introduces gaps to improve cache locality.
    #[inline(always)]
    const fn index(i: usize) -> usize {
        i + (i >> 14)
    }

    /// Computes the size of the internal array needed to accommodate `n` elements.
    ///
    /// - If `HOLES = true`: Returns `n + index(n)` to account for gaps
    /// - If `HOLES = false`: Returns `n` (compact layout)
    const fn size(n: usize) -> usize {
        if HOLES {
            n + Self::index(n)
        } else {
            n
        }
    }


    /// Computes the sum of elements in the specified range.
    ///
    /// Supports any range type (`a..b`, `a..=b`, `..b`, `a..`, etc.) and computes
    /// the sum in Θ(log n) time using the Fenwick tree structure.
    ///
    /// # Arguments
    ///
    /// * `bounds` - A range specifying the elements to sum (0-based indexing)
    ///
    /// # Returns
    ///
    /// The sum of elements in the specified range. Returns `T::zero()` for empty ranges.
    ///
    /// # Panics
    ///
    /// Panics if the bounds are out of range (>= n).
    ///
    /// # Complexity
    ///
    /// Θ(log n) where n is the tree size.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::FenwickTree;
    ///
    /// let ft = FenwickTree::<i32, false>::from(vec![1, 2, 3, 4, 5]);
    ///
    /// assert_eq!(ft.sum(0..3), 6);    // [1, 2, 3]
    /// assert_eq!(ft.sum(1..=3), 9);   // [2, 3, 4]
    /// assert_eq!(ft.sum(..), 15);     // all elements
    /// assert_eq!(ft.sum(2..), 12);    // [3, 4, 5]
    /// assert_eq!(ft.sum(2..2), 0);    // empty range
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn sum<B>(&self, bounds: B) -> T
    where
        B: RangeBounds<usize>,
    {
        let mut start = match bounds.start_bound() {
            Bound::Excluded(&usize::MAX) => panic!("Bound is out of range"),
            Bound::Excluded(x) => *x + 1,
            Bound::Included(x) => *x,
            Bound::Unbounded => usize::MIN,
        };

        let mut end = match bounds.end_bound() {
            Bound::Included(&usize::MAX) => usize::MAX,
            Bound::Included(x) => *x + 1,
            Bound::Excluded(x) => *x,
            Bound::Unbounded => self.n,
        };

        assert!(end <= self.n);
        assert!(start < self.n);

        let mut sum = T::zero();

        // TODO: instead of stopping at 0 we can stop at (start & end)
        // TODO: do we need get_unchecked?
        while end > start {
            let pos = if HOLES { Self::index(end - 1) } else { end - 1 };
            sum += self.tree[pos];
            end = Self::prev(end);
        }

        while start > end {
            let pos = if HOLES {
                Self::index(start - 1)
            } else {
                start - 1
            };
            sum -= self.tree[pos];
            start = Self::prev(start);
        }

        sum
    }

    /// Adds a value to the element at index `i`.
    ///
    /// This operation updates the tree structure in Θ(log n) time.
    ///
    /// # Arguments
    ///
    /// * `i` - The index to update (0-based)
    /// * `v` - The value to add
    ///
    /// # Panics
    ///
    /// Panics if `i >= n`.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::FenwickTree;
    ///
    /// let mut ft = FenwickTree::<i32, false>::from(vec![1, 2, 3, 4, 5]);
    /// ft.add_at(2, 10);
    /// assert_eq!(ft.sum(0..=2), 16); // 1 + 2 + (3 + 10)
    ///
    /// // Iterate to see the updated values
    /// let elements: Vec<i32> = ft.iter().collect();
    /// assert_eq!(elements, vec![1, 2, 13, 4, 5]);
    /// ```
    #[inline(always)]
    pub fn add_at(&mut self, i: usize, v: T) {
        assert!(i < self.n);

        let mut i = i;
        while i < self.n {
            let pos = if HOLES { Self::index(i) } else { i };
            self.tree[pos] += v;
            i = Self::next(i);
        }
    }

    /// Returns the number of elements in the Fenwick tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::FenwickTree;
    ///
    /// let ft = FenwickTree::<i32, false>::with_len(100);
    /// assert_eq!(ft.len(), 100);
    /// ```
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns `true` if the Fenwick tree contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::FenwickTree;
    ///
    /// let ft = FenwickTree::<i32, false>::with_len(0);
    /// assert!(ft.is_empty());
    ///
    /// let ft2 = FenwickTree::<i32, false>::with_len(10);
    /// assert!(!ft2.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Constructs a Fenwick tree from a vector of values.
    ///
    /// This is more efficient than creating an empty tree and calling `add_at`
    /// repeatedly, as it builds the tree structure in Θ(n) time.
    ///
    /// # Arguments
    ///
    /// * `v` - A vector of initial values
    ///
    /// # Returns
    ///
    /// A Fenwick tree containing the elements from the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::FenwickTree;
    ///
    /// let ft = FenwickTree::<i32, false>::from(vec![1, 2, 3, 4, 5]);
    /// assert_eq!(ft.len(), 5);
    /// assert_eq!(ft.sum(0..5), 15);
    /// ```
    pub fn from(mut v: Vec<T>) -> Self {
        let n = v.len();

        if HOLES {
            v.resize(Self::index(n), T::zero());
        }

        for i in 0..n {
            let parent = Self::next(i);
            if parent < n {
                let child = v[i];
                v[parent] += child;
            }
        }

        FenwickTree {
            tree: v.into_boxed_slice(),
            n,
        }
    }
}

/// Iterator over the original elements of a Fenwick tree.
///
/// This is the base iterator that traverses the Fenwick tree structure using a stack-based
/// algorithm. The stack maintains the path from the root to the current position, allowing
/// Θ(n) total iteration time (amortized Θ(1) per element).
///
/// # Algorithm
///
/// The iterator maintains:
/// - A stack of (position, value) pairs representing nodes in the current Fenwick path
/// - A cumulative sum of values in the stack (the prefix sum up to the current position)
///
/// When moving to the next position:
/// 1. Find the parent node in the Fenwick tree
/// 2. Pop nodes from the stack until the parent is found (or stack is empty)
/// 3. Push the current node onto the stack
/// 4. Compute the element as the difference between consecutive prefix sums
///
/// This approach ensures each tree node is visited at most a constant number of times
/// across the entire iteration, achieving Θ(n) total complexity.
///
/// # Examples
///
/// ```
/// use toolkit::FenwickTree;
///
/// let ft = FenwickTree::<i32, false>::from(vec![1, 2, 3, 4, 5]);
///
/// // Iterate over elements
/// let elements: Vec<i32> = ft.iter().collect();
/// assert_eq!(elements, vec![1, 2, 3, 4, 5]);
///
/// // Use iterator methods
/// let sum: i32 = ft.iter().sum();
/// assert_eq!(sum, 15);
///
/// let first_three: Vec<i32> = ft.iter().take(3).collect();
/// assert_eq!(first_three, vec![1, 2, 3]);
/// ```
pub struct FenwickIter<'a, T, const HOLES: bool>
where
    T: Copy + Zero + AddAssign + SubAssign,
{
    ft: &'a FenwickTree<T, HOLES>,
    i: usize,
    /// Stack of (1-indexed position, tree value) pairs representing the current path
    stack: Vec<(usize, T)>,
    /// Cumulative sum from the stack (prefix sum of current position)
    sum: T,
}

/// Iterator over prefix sums of a Fenwick tree.
///
/// This iterator is built on top of `FenwickIter` and accumulates the elements
/// to produce cumulative sums.
///
/// # Complexity
///
/// - Total: Θ(n) for all prefix sums
/// - Per element: Θ(1) amortized
///
/// # Examples
///
/// ```
/// use toolkit::FenwickTree;
///
/// let ft = FenwickTree::<i32, false>::from(vec![1, 2, 3, 4, 5]);
/// let prefix_sums: Vec<i32> = ft.prefix_sums().collect();
///
/// // prefix_sums[i] = sum of elements from 0 to i (inclusive)
/// assert_eq!(prefix_sums[0], 1);      // 1
/// assert_eq!(prefix_sums[1], 3);      // 1 + 2
/// assert_eq!(prefix_sums[2], 6);      // 1 + 2 + 3
/// assert_eq!(prefix_sums[3], 10);     // 1 + 2 + 3 + 4
/// assert_eq!(prefix_sums[4], 15);     // 1 + 2 + 3 + 4 + 5
/// ```
pub struct FenwickPrefixSumIter<'a, T, const HOLES: bool>
where
    T: Copy + Zero + AddAssign + SubAssign,
{
    iter: FenwickIter<'a, T, HOLES>,
    cumsum: T,
}

impl<T, const HOLES: bool> FenwickTree<T, HOLES>
where
    T: Copy + Zero + AddAssign + SubAssign,
{
    /// Returns an iterator over the original elements (in index order 0..n).
    ///
    /// The iterator reconstructs the original values by traversing the Fenwick tree
    /// structure using an efficient stack-based algorithm with Θ(n) total complexity.
    ///
    /// # Returns
    ///
    /// A `FenwickIter` that yields each element in the original sequence.
    ///
    /// # Complexity
    ///
    /// - Total: Θ(n) for iterating all elements
    /// - Per element: Θ(1) amortized
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::FenwickTree;
    ///
    /// let mut ft = FenwickTree::<i32, false>::from(vec![1, 2, 3, 4, 5]);
    /// ft.add_at(2, 10); // Modify element at index 2
    ///
    /// let elements: Vec<i32> = ft.iter().collect();
    /// assert_eq!(elements, vec![1, 2, 13, 4, 5]);
    /// ```
    #[inline]
    pub fn iter(&self) -> FenwickIter<'_, T, HOLES> {
        FenwickIter {
            ft: self,
            i: 0,
            stack: Vec::new(),
            sum: T::zero(),
        }
    }

    /// Returns an iterator over prefix sums (cumulative sums from index 0 to each position).
    ///
    /// Each element yielded is the sum of all elements from index 0 up to and including
    /// the current position. This is equivalent to calling `sum(0..=i)` for each index i,
    /// but much more efficient.
    ///
    /// # Returns
    ///
    /// A `FenwickPrefixSumIter` that yields prefix sums.
    ///
    /// # Complexity
    ///
    /// - Total: Θ(n) for iterating all prefix sums
    /// - Per element: Θ(1) amortized
    ///
    /// # Examples
    ///
    /// ```
    /// use toolkit::FenwickTree;
    ///
    /// let ft = FenwickTree::<i32, false>::from(vec![1, 2, 3, 4, 5]);
    ///
    /// let prefix_sums: Vec<i32> = ft.prefix_sums().collect();
    /// assert_eq!(prefix_sums, vec![1, 3, 6, 10, 15]);
    ///
    /// // Verify against sum queries
    /// for (i, &ps) in prefix_sums.iter().enumerate() {
    ///     assert_eq!(ps, ft.sum(0..=i));
    /// }
    /// ```
    #[inline]
    pub fn prefix_sums(&self) -> FenwickPrefixSumIter<'_, T, HOLES> {
        FenwickPrefixSumIter {
            iter: self.iter(),
            cumsum: T::zero(),
        }
    }
}

impl<'a, T, const HOLES: bool> Iterator for FenwickIter<'a, T, HOLES>
where
    T: Copy + Zero + AddAssign + SubAssign,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.ft.n {
            return None;
        }

        let idx = self.i;
        self.i += 1;

        let prev_sum = self.sum;
        let pos = idx + 1;
        let parent = FenwickTree::<T, HOLES>::prev(pos);

        // Pop from stack until we find the parent or stack is empty
        while let Some(&(top_pos, val)) = self.stack.last() {
            if top_pos == parent {
                // Found the parent, we can use it
                break;
            } else {
                // This node is not in our path, pop it
                self.stack.pop();
                self.sum -= val;
            }
        }

        // Now push current position onto the stack
        let tree_idx = if HOLES {
            FenwickTree::<T, HOLES>::index(idx)
        } else {
            idx
        };
        let val = self.ft.tree[tree_idx];
        self.stack.push((pos, val));
        self.sum += val;

        // Element at idx is: prefix_sum[idx] - prefix_sum[idx-1]
        let element = {
            let mut e = self.sum;
            e -= prev_sum;
            e
        };

        Some(element)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.ft.n - self.i;
        (remaining, Some(remaining))
    }
}

impl<'a, T, const HOLES: bool> ExactSizeIterator for FenwickIter<'a, T, HOLES>
where
    T: Copy + Zero + AddAssign + SubAssign,
{
    #[inline]
    fn len(&self) -> usize {
        self.ft.n - self.i
    }
}

impl<'a, T, const HOLES: bool> FusedIterator for FenwickIter<'a, T, HOLES>
where
    T: Copy + Zero + AddAssign + SubAssign,
{
}

impl<'a, T, const HOLES: bool> Iterator for FenwickPrefixSumIter<'a, T, HOLES>
where
    T: Copy + Zero + AddAssign + SubAssign,
{
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|element| {
            self.cumsum += element;
            self.cumsum
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T, const HOLES: bool> ExactSizeIterator for FenwickPrefixSumIter<'a, T, HOLES>
where
    T: Copy + Zero + AddAssign + SubAssign,
{
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<'a, T, const HOLES: bool> FusedIterator for FenwickPrefixSumIter<'a, T, HOLES>
where
    T: Copy + Zero + AddAssign + SubAssign,
{
}

/// Construct a Fenwick tree from a slice.
///
/// This is a convenience implementation that converts the slice to a `Vec` and
/// delegates to the `From<Vec<T>>` implementation.
///
/// # Complexity
///
/// Θ(n) where n is the length of the slice.
///
/// # Examples
///
/// ```
/// use toolkit::FenwickTree;
///
/// let data = [1, 2, 3, 4, 5];
/// let ft = FenwickTree::<i32, false>::from(&data[..]);
///
/// assert_eq!(ft.sum(..), 15);
/// assert_eq!(ft.iter().collect::<Vec<_>>(), vec![1, 2, 3, 4, 5]);
/// ```
impl<T, const HOLES: bool> From<&[T]> for FenwickTree<T, HOLES>
where
    T: Copy + Zero + AddAssign + SubAssign,
{
    fn from(v: &[T]) -> Self {
        FenwickTree::from(v.to_vec())
    }
}

/// Construct a Fenwick tree from an iterator.
///
/// This implementation collects all elements from the iterator into a `Vec`
/// and then builds the Fenwick tree using the `From<Vec<T>>` implementation.
///
/// # Complexity
///
/// Θ(n) where n is the number of elements in the iterator.
///
/// # Examples
///
/// ```
/// use toolkit::FenwickTree;
///
/// let ft: FenwickTree<i32, false> = (1..=5).collect();
///
/// assert_eq!(ft.sum(..), 15);
/// assert_eq!(ft.len(), 5);
/// ```
///
/// Can be used with any iterator:
///
/// ```
/// use toolkit::FenwickTree;
///
/// let data = vec![10, 20, 30];
/// let ft: FenwickTree<i32, false> = data.iter().copied().collect();
///
/// assert_eq!(ft.sum(..), 60);
/// ```
impl<T, const HOLES: bool> FromIterator<T> for FenwickTree<T, HOLES>
where
    T: Copy + Zero + AddAssign + SubAssign,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Self::from(iter.into_iter().collect::<Vec<_>>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fenwick_tree() {
        let n = 1000;

        let ft: FenwickTree<_, false> = (0..n).into_iter().collect();

        let mut s = 0;
        for i in 0..n {
            assert_eq!(ft.sum(0..i), s);
            s += i;
        }
    }

    #[test]
    fn test_fenwick_iter() {
        // Test with simple sequence
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let ft: FenwickTree<i32, false> = data.clone().into_iter().collect();
        
        let collected: Vec<i32> = ft.iter().collect();
        assert_eq!(collected, data, "Iterator should return original elements");
    }

    #[test]
    fn test_fenwick_iter_holes() {
        // Test with HOLES = true
        let data = vec![10, 20, 30, 40, 50];
        let ft: FenwickTree<i32, true> = data.clone().into_iter().collect();
        
        let collected: Vec<i32> = ft.iter().collect();
        assert_eq!(collected, data, "Iterator with HOLES should return original elements");
    }

    #[test]
    fn test_fenwick_iter_large() {
        // Test with larger sequence
        let n = 1000;
        let data: Vec<usize> = (0..n).collect();
        let ft: FenwickTree<usize, false> = data.clone().into_iter().collect();
        
        let collected: Vec<usize> = ft.iter().collect();
        assert_eq!(collected, data, "Iterator should work for large sequences");
    }

    #[test]
    fn test_fenwick_iter_after_updates() {
        // Test iterator after modifications
        let mut ft: FenwickTree<i32, false> = vec![1, 2, 3, 4, 5].into_iter().collect();
        
        // Add some values
        ft.add_at(2, 10); // element at index 2 becomes 13
        ft.add_at(4, 5);  // element at index 4 becomes 10
        
        let collected: Vec<i32> = ft.iter().collect();
        assert_eq!(collected, vec![1, 2, 13, 4, 10], "Iterator should reflect updates");
    }

    #[test]
    fn test_fenwick_iter_empty() {
        // Test with empty tree
        let ft: FenwickTree<i32, false> = vec![].into_iter().collect();
        let collected: Vec<i32> = ft.iter().collect();
        assert_eq!(collected, vec![], "Iterator should handle empty tree");
    }

    #[test]
    fn test_fenwick_iter_single_element() {
        // Test with single element
        let data = vec![42];
        let ft: FenwickTree<i32, false> = data.clone().into_iter().collect();
        let collected: Vec<i32> = ft.iter().collect();
        assert_eq!(collected, data, "Iterator should handle single element");
    }

    #[test]
    fn test_fenwick_iter_two_elements() {
        // Test with two elements
        let data = vec![10, 20];
        let ft: FenwickTree<i32, false> = data.clone().into_iter().collect();
        let collected: Vec<i32> = ft.iter().collect();
        assert_eq!(collected, data, "Iterator should handle two elements");
    }

    #[test]
    fn test_fenwick_iter_power_of_two() {
        // Test with power of 2 sizes (different tree structures)
        for &size in &[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            let data: Vec<i32> = (0..size).map(|i| i as i32).collect();
            let ft: FenwickTree<i32, false> = data.clone().into_iter().collect();
            let collected: Vec<i32> = ft.iter().collect();
            assert_eq!(collected, data, "Iterator should work for size {}", size);
        }
    }

    #[test]
    fn test_fenwick_iter_non_power_of_two() {
        // Test with non-power of 2 sizes
        for size in [3, 5, 7, 11, 13, 17, 31, 63, 127, 255, 1000] {
            let data: Vec<i32> = (0..size).map(|i| i as i32).collect();
            let ft: FenwickTree<i32, false> = data.clone().into_iter().collect();
            let collected: Vec<i32> = ft.iter().collect();
            assert_eq!(collected, data, "Iterator should work for size {}", size);
        }
    }

    #[test]
    fn test_fenwick_iter_negative_values() {
        // Test with negative values
        let data = vec![-5, -3, -1, 0, 1, 3, 5, 10, -20, 15];
        let ft: FenwickTree<i32, false> = data.clone().into_iter().collect();
        let collected: Vec<i32> = ft.iter().collect();
        assert_eq!(collected, data, "Iterator should handle negative values");
    }

    #[test]
    fn test_fenwick_iter_zeros() {
        // Test with all zeros
        let data = vec![0, 0, 0, 0, 0, 0, 0, 0];
        let ft: FenwickTree<i32, false> = data.clone().into_iter().collect();
        let collected: Vec<i32> = ft.iter().collect();
        assert_eq!(collected, data, "Iterator should handle all zeros");
    }

    #[test]
    fn test_fenwick_iter_same_values() {
        // Test with all same values
        let data = vec![7; 100];
        let ft: FenwickTree<i32, false> = data.clone().into_iter().collect();
        let collected: Vec<i32> = ft.iter().collect();
        assert_eq!(collected, data, "Iterator should handle same values");
    }

    #[test]
    fn test_fenwick_iter_alternating() {
        // Test with alternating pattern
        let data: Vec<i32> = (0..100).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let ft: FenwickTree<i32, false> = data.clone().into_iter().collect();
        let collected: Vec<i32> = ft.iter().collect();
        assert_eq!(collected, data, "Iterator should handle alternating values");
    }

    #[test]
    fn test_fenwick_iter_after_subtract() {
        // Test iterator after subtractions
        let mut ft: FenwickTree<i32, false> = vec![10, 20, 30, 40, 50].into_iter().collect();
        
        ft.sub_at(1, 5);  // element at index 1 becomes 15
        ft.sub_at(3, 10); // element at index 3 becomes 30
        
        let collected: Vec<i32> = ft.iter().collect();
        assert_eq!(collected, vec![10, 15, 30, 30, 50], "Iterator should reflect subtractions");
    }

    #[test]
    fn test_fenwick_iter_mixed_operations() {
        // Test iterator after mixed add/subtract operations
        let mut ft: FenwickTree<i32, false> = vec![1, 2, 3, 4, 5, 6, 7, 8].into_iter().collect();
        
        ft.add_at(0, 10);
        ft.sub_at(2, 1);
        ft.add_at(4, 5);
        ft.sub_at(6, 3);
        
        let collected: Vec<i32> = ft.iter().collect();
        assert_eq!(collected, vec![11, 2, 2, 4, 10, 6, 4, 8], "Iterator should reflect mixed operations");
    }

    #[test]
    fn test_fenwick_iter_holes_vs_no_holes() {
        // Test that HOLES=true and HOLES=false produce same iteration results
        let data: Vec<i32> = (1..=100).collect();
        
        let ft_holes: FenwickTree<i32, true> = data.clone().into_iter().collect();
        let ft_no_holes: FenwickTree<i32, false> = data.clone().into_iter().collect();
        
        let collected_holes: Vec<i32> = ft_holes.iter().collect();
        let collected_no_holes: Vec<i32> = ft_no_holes.iter().collect();
        
        assert_eq!(collected_holes, data, "HOLES=true should match original");
        assert_eq!(collected_no_holes, data, "HOLES=false should match original");
        assert_eq!(collected_holes, collected_no_holes, "Both should produce same results");
    }

    #[test]
    fn test_fenwick_iter_consistency_with_sum() {
        // Verify that iterator values match sum queries
        let data: Vec<i32> = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
        let ft: FenwickTree<i32, false> = data.clone().into_iter().collect();
        
        let collected: Vec<i32> = ft.iter().collect();
        
        // Check each position
        for i in 0..data.len() {
            assert_eq!(collected[i], data[i], "Element {} should match", i);
            
            // Also verify prefix sum is correct
            let prefix_sum: i32 = collected[0..=i].iter().sum();
            assert_eq!(ft.sum(0..=i), prefix_sum, "Prefix sum at {} should match", i);
        }
    }

    #[test]
    fn test_fenwick_iter_multiple_iterations() {
        // Test that multiple iterations produce same results
        let data: Vec<i32> = (1..=50).collect();
        let ft: FenwickTree<i32, false> = data.clone().into_iter().collect();
        
        let iter1: Vec<i32> = ft.iter().collect();
        let iter2: Vec<i32> = ft.iter().collect();
        let iter3: Vec<i32> = ft.iter().collect();
        
        assert_eq!(iter1, data, "First iteration should match");
        assert_eq!(iter2, data, "Second iteration should match");
        assert_eq!(iter3, data, "Third iteration should match");
        assert_eq!(iter1, iter2, "Iterations should be consistent");
        assert_eq!(iter2, iter3, "Iterations should be consistent");
    }

    #[test]
    fn test_fenwick_iter_partial_iteration() {
        // Test partial iteration with take
        let data: Vec<i32> = (0..100).collect();
        let ft: FenwickTree<i32, false> = data.clone().into_iter().collect();
        
        let first_10: Vec<i32> = ft.iter().take(10).collect();
        assert_eq!(first_10, &data[0..10], "First 10 elements should match");
        
        let first_50: Vec<i32> = ft.iter().take(50).collect();
        assert_eq!(first_50, &data[0..50], "First 50 elements should match");
    }

    #[test]
    fn test_fenwick_iter_skip() {
        // Test iteration with skip
        let data: Vec<i32> = (0..100).collect();
        let ft: FenwickTree<i32, false> = data.clone().into_iter().collect();
        
        let skipped: Vec<i32> = ft.iter().skip(50).collect();
        assert_eq!(skipped, &data[50..], "Skipped elements should match");
    }

    #[test]
    fn test_fenwick_iter_enumerate() {
        // Test iteration with enumerate
        let data: Vec<i32> = vec![10, 20, 30, 40, 50];
        let ft: FenwickTree<i32, false> = data.clone().into_iter().collect();
        
        for (i, val) in ft.iter().enumerate() {
            assert_eq!(val, data[i], "Element at index {} should match", i);
        }
    }

    #[test]
    fn test_fenwick_iter_zip() {
        // Test iteration with zip
        let data: Vec<i32> = (1..=20).collect();
        let ft: FenwickTree<i32, false> = data.clone().into_iter().collect();
        
        let zipped: Vec<(i32, i32)> = ft.iter().zip(data.iter().copied()).collect();
        
        for (from_iter, from_vec) in zipped {
            assert_eq!(from_iter, from_vec, "Zipped values should match");
        }
    }

    #[test]
    fn test_fenwick_iter_large_values() {
        // Test with large i64 values
        let data: Vec<i64> = vec![
            1_000_000_000,
            -500_000_000,
            2_000_000_000,
            -1_000_000_000,
            3_000_000_000,
        ];
        let ft: FenwickTree<i64, false> = data.clone().into_iter().collect();
        let collected: Vec<i64> = ft.iter().collect();
        assert_eq!(collected, data, "Iterator should handle large values");
    }

    #[test]
    fn test_fenwick_iter_stress_test() {
        // Stress test with large tree and random-like pattern
        let size = 10000;
        let data: Vec<i32> = (0..size).map(|i| (i * 7919 + 104729) % 1000 - 500).collect();
        let ft: FenwickTree<i32, false> = data.clone().into_iter().collect();
        let collected: Vec<i32> = ft.iter().collect();
        assert_eq!(collected, data, "Iterator should handle large stress test");
    }

    #[test]
    fn test_fenwick_prefix_sums() {
        // Test prefix sums iterator
        let data = vec![1, 2, 3, 4, 5];
        let ft: FenwickTree<i32, false> = data.clone().into_iter().collect();
        
        let prefix_sums: Vec<i32> = ft.prefix_sums().collect();
        assert_eq!(prefix_sums, vec![1, 3, 6, 10, 15], "Prefix sums should be cumulative");
    }

    #[test]
    fn test_fenwick_prefix_sums_negative() {
        // Test prefix sums with negative values
        let data = vec![5, -3, 2, -1, 4];
        let ft: FenwickTree<i32, false> = data.clone().into_iter().collect();
        
        let prefix_sums: Vec<i32> = ft.prefix_sums().collect();
        assert_eq!(prefix_sums, vec![5, 2, 4, 3, 7], "Prefix sums should handle negatives");
    }

    #[test]
    fn test_fenwick_prefix_sums_consistency() {
        // Verify prefix sums match sum queries
        let data: Vec<i32> = vec![3, 1, 4, 1, 5, 9, 2, 6];
        let ft: FenwickTree<i32, false> = data.clone().into_iter().collect();
        
        let prefix_sums: Vec<i32> = ft.prefix_sums().collect();
        
        for (i, &ps) in prefix_sums.iter().enumerate() {
            assert_eq!(ps, ft.sum(0..=i), "Prefix sum at {} should match sum query", i);
        }
    }

    #[test]
    fn test_fenwick_prefix_sums_empty() {
        // Test prefix sums with empty tree
        let ft: FenwickTree<i32, false> = vec![].into_iter().collect();
        let prefix_sums: Vec<i32> = ft.prefix_sums().collect();
        assert_eq!(prefix_sums, vec![], "Empty tree should produce empty prefix sums");
    }

    #[test]
    fn test_fenwick_iter_exact_size() {
        // Test ExactSizeIterator implementation
        let data: Vec<i32> = (0..100).collect();
        let ft: FenwickTree<i32, false> = data.clone().into_iter().collect();
        
        let mut iter = ft.iter();
        assert_eq!(iter.len(), 100, "Initial length should be 100");
        
        iter.next();
        assert_eq!(iter.len(), 99, "Length after one next should be 99");
        
        for _ in 0..50 {
            iter.next();
        }
        assert_eq!(iter.len(), 49, "Length after 51 elements should be 49");
        
        let remaining: Vec<i32> = iter.collect();
        assert_eq!(remaining.len(), 49, "Should have 49 elements remaining");
    }

    #[test]
    fn test_fenwick_prefix_sums_exact_size() {
        // Test ExactSizeIterator for prefix sums
        let data: Vec<i32> = (1..=50).collect();
        let ft: FenwickTree<i32, false> = data.clone().into_iter().collect();
        
        let mut iter = ft.prefix_sums();
        assert_eq!(iter.len(), 50, "Initial length should be 50");
        
        iter.next();
        assert_eq!(iter.len(), 49, "Length after one next should be 49");
        
        let _ = iter.by_ref().take(20).collect::<Vec<_>>();
        assert_eq!(iter.len(), 29, "Length after taking 20 should be 29");
    }

    #[test]
    fn test_fenwick_iter_size_hint() {
        // Test size_hint
        let data: Vec<i32> = (0..100).collect();
        let ft: FenwickTree<i32, false> = data.clone().into_iter().collect();
        
        let iter = ft.iter();
        assert_eq!(iter.size_hint(), (100, Some(100)), "Size hint should be exact");
        
        let iter = ft.iter().skip(30);
        assert_eq!(iter.size_hint(), (70, Some(70)), "Size hint after skip should be 70");
    }

    #[test]
    fn test_fenwick_iter_fused() {
        // Test FusedIterator - calling next after exhaustion should keep returning None
        let data = vec![1, 2, 3];
        let ft: FenwickTree<i32, false> = data.into_iter().collect();
        
        let mut iter = ft.iter();
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_fenwick_prefix_sums_large() {
        // Test prefix sums with larger dataset
        let size = 1000;
        let data: Vec<i32> = (1..=size).collect();
        let ft: FenwickTree<i32, false> = data.clone().into_iter().collect();
        
        let prefix_sums: Vec<i32> = ft.prefix_sums().collect();
        
        // Verify a few specific values
        assert_eq!(prefix_sums[0], 1);
        assert_eq!(prefix_sums[9], 55); // sum of 1..=10
        assert_eq!(prefix_sums[99], 5050); // sum of 1..=100
        assert_eq!(prefix_sums[999], (size * (size + 1)) / 2); // triangular number formula
    }

    #[test]
    fn test_fenwick_iter_collect_twice() {
        // Test that we can create multiple independent iterators
        let data: Vec<i32> = (1..=10).collect();
        let ft: FenwickTree<i32, false> = data.clone().into_iter().collect();
        
        let collected1: Vec<i32> = ft.iter().collect();
        let collected2: Vec<i32> = ft.iter().collect();
        
        assert_eq!(collected1, data);
        assert_eq!(collected2, data);
        assert_eq!(collected1, collected2);
    }

    #[test]
    fn test_fenwick_prefix_sums_after_updates() {
        // Test prefix sums after tree modifications
        let mut ft: FenwickTree<i32, false> = vec![1, 2, 3, 4, 5].into_iter().collect();
        
        ft.add_at(2, 10); // Element at index 2 becomes 13
        
        let prefix_sums: Vec<i32> = ft.prefix_sums().collect();
        // [1, 2, 13, 4, 5] -> prefix sums: [1, 3, 16, 20, 25]
        assert_eq!(prefix_sums, vec![1, 3, 16, 20, 25]);
    }
}