///! Utility functions for Stream VByte encoding/decoding.
///! Many functions and documentation here are from the nice implementation
///! by [Denis Bazhenov](https://github.com/bazhenov/svbyte/blob/master/src/lib.rs).
use std::{
    arch::x86_64::{_mm_loadu_si128, _mm_shuffle_epi8, _mm_storeu_si128},
    debug_assert,
};

#[allow(non_camel_case_types)]
type u32x4 = [u32; 4];

#[inline]
pub(super) fn decode_16_aligned(buffer: &mut [u32; 16], controls: &[u8], data: &[u8]) -> usize {
    let mut buffer_ptr = buffer.as_mut_ptr().cast::<u32x4>();
    let mut control_words = controls.as_ptr();
    let mut data_stream = data.as_ptr();

    for _ in 0..4 {
        let encoded_len =
            unsafe { simd_decode(&*data_stream.cast(), *control_words, &mut *buffer_ptr) };

        control_words = control_words.wrapping_add(1);
        buffer_ptr = buffer_ptr.wrapping_add(1);

        data_stream = data_stream.wrapping_add(encoded_len);
    }
    data_stream as usize - data.as_ptr() as usize
}

/// Decode all the u32 values given a slice of control bytes and data bytes. Decoded values are written to `buffer` slice, which must be large enough to hold `length` values.
/// The function returns the number of bytes read from the data slice.
#[inline]
pub(super) fn decode_slice_aligned(buffer: &mut [u32], controls: &[u8], data: &[u8]) -> usize {
    assert!(
        buffer.len() >= controls.len() * 4,
        "output slice is not large enough"
    );

    let mut iterations = controls.len();

    let mut buffer: *mut u32x4 = buffer.as_mut_ptr().cast();
    let mut control_words = controls.as_ptr();
    let mut data_stream = data.as_ptr();
    let data_stream_end = (data.last().unwrap() as *const u8).wrapping_add(1);
    let mut data_stream_offset = 0usize;

    // Without manual loop unroll, performance is quite bad on Intel Xeon E3
    // Decode loop unrolling
    const UNROLL_FACTOR: usize = 4;
    let n_unrolled_iterations = iterations / UNROLL_FACTOR;

    for _ in 0..n_unrolled_iterations {
        for _ in 0..UNROLL_FACTOR {
            debug_assert!(
                data_stream.wrapping_add(16) <= data_stream_end,
                "At least 16 bytes should be available in the data stream"
            );
            let encoded_len =
                unsafe { simd_decode(&*data_stream.cast(), *control_words, &mut *buffer) };

            control_words = control_words.wrapping_add(1);
            buffer = buffer.wrapping_add(1);

            data_stream = data_stream.wrapping_add(encoded_len);
            data_stream_offset += encoded_len;
        }

        iterations -= UNROLL_FACTOR;
    }

    // Tail decode
    for _ in 0..iterations {
        debug_assert!(
            data_stream.wrapping_add(16) <= data_stream_end,
            "At least 16 bytes should be available in the data stream"
        );
        let encoded_len =
            unsafe { simd_decode(&*data_stream.cast(), *control_words, &mut *buffer) };

        control_words = control_words.wrapping_add(1);
        buffer = buffer.wrapping_add(1);

        data_stream = data_stream.wrapping_add(encoded_len);
        data_stream_offset += encoded_len;
    }

    data_stream_offset
}

/// Shuffle masks and correspinding length of encoded numbers
///
/// For more information see documentation to [`u32_shuffle_masks`]
///
/// [`u32_shuffle_masks`]: u32_shuffle_masks
const MASKS: [(u32x4, usize); 256] = u32_shuffle_masks();

pub(super) const U32_LENGTHS: [u8; 256] = u32_lengths();

const fn u32_lengths() -> [u8; 256] {
    let mut lengths = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        lengths[i] =
            ((i & 0b11) + ((i >> 2) & 0b11) + ((i >> 4) & 0b11) + ((i >> 6) & 0b11)) as u8 + 4;
        i += 1;
    }
    lengths
}

/// Decoding SIMD kernel using SSSE3 intrinsics
///
/// Types of this function tries to implement safety guardrails as much as possible. Namely:
/// `output` - is a reference to the buffer of 4 u32 values;
/// `input` - is a reference to u8 array of unspecified length (`control_word` speciefies how much will be decoded);
//
/// Technically the encoded length can be calculated from control word directly using horizontal 2-bit sum
/// ```rust,ignore
/// let result = *control_word;
/// let result = ((result & 0b11001100) >> 2) + (result & 0b00110011);
/// let result = (result >> 4) + (result & 0b1111) + 4;
/// ```
/// Unfortunatley, this approach is slower then memoized length. There is a mention of this approach can be faster
/// when using `u32` control words, which implies decoding a batch of size 16[^1].
///
/// [^1]: [Bit hacking versus memoization: a Stream VByte example](https://lemire.me/blog/2017/11/28/bit-hacking-versus-memoization-a-stream-vbyte-example/)
#[inline]
fn simd_decode(input: &[u8; 16], control_word: u8, output: &mut u32x4) -> usize {
    let (ref mask, encoded_len) = MASKS[control_word as usize];
    unsafe {
        let mask = _mm_loadu_si128(mask.as_ptr().cast());
        let input = _mm_loadu_si128(input.as_ptr().cast());
        let answer = _mm_shuffle_epi8(input, mask);
        _mm_storeu_si128(output.as_mut_ptr().cast(), answer);
    }

    encoded_len
}

/**
Prepares shuffle mask for decoding a single `u32` using `pshufb` instruction

`len` parameter is describing the length of decoded `u32` in the input register (1-4). `offset` parameter is
describing the base offset in the register. It is the sum of all previous number lengths loaded in the input register.
*/
const fn u32_shuffle_mask(len: usize, offset: usize) -> u32 {
    const PZ: u8 = 0b10000000;
    assert!(offset < 16, "Offset should be <16");
    let offset = offset as u8;
    let p1 = offset;
    let p2 = offset + 1;
    let p3 = offset + 2;
    let p4 = offset + 3;
    match len {
        1 => u32::from_be_bytes([PZ, PZ, PZ, p1]),
        2 => u32::from_be_bytes([PZ, PZ, p1, p2]),
        3 => u32::from_be_bytes([PZ, p1, p2, p3]),
        4 => u32::from_be_bytes([p1, p2, p3, p4]),
        _ => panic!("Length of u32 is 1..=4 bytes"),
    }
}

/**
Preparing shuffling masks for `pshufb` SSE instructions

`pshufb` (`_mm_shuffle_epi8()`) allows to shuffle bytes around in a `__mm128` register. Shuffle mask consist of 16
bytes. Each byte describe byte index in input register which should be copied to corresponding byte in the output
register. For addressing 16 bytes we need log(16) = 4 bits. So bits 0:3 of each byte are storing input register byte
index. MSB of each byte indicating if corresponding byte in output register should be zeroed out. 4 least significant
bits are non effective if MSB is set.

`pshufb` SSE instruction visualization.

```graph
  Byte offsets:           0        1        2        3        4
                  ┌────────┬────────┬────────┬────────┬────────┬───┐
Input Register:   │   0x03 │   0x15 │   0x22 │   0x19 │   0x08 │...│
                  └────▲───┴────────┴────▲───┴────▲───┴────▲───┴───┘
                       │        ┌────────┘        │        │
                       │        │        ┌─────────────────┘
                       │        │        │        │
                       └───────────────────────────────────┐
                                │        │        │        │
                  ┌────────┬────┴───┬────┴───┬────┴───┬────┴───┬───┐
  Mask Register:  │   0x80 │   0x02 │   0x04 │   0x03 │   0x00 │...│
                  ├────────┼────────┼────────┼────────┼────────┼───┤
Output Register:  │   0x00 │   0x22 │   0x08 │   0x19 │   0x03 │...│
                  └────────┴────────┴────────┴────────┴────────┴───┘
```

See [`_mm_shuffle_epi8()`][_mm_shuffle_epi8] documentation.

[_mm_shuffle_epi8]: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=shuffle_epi8&ig_expand=6097
*/
const fn u32_shuffle_masks() -> [(u32x4, usize); 256] {
    let mut masks = [([0u32; 4], 0usize); 256];

    let mut a = 1;
    while a <= 4 {
        let mut b = 1;
        while b <= 4 {
            let mut c = 1;
            while c <= 4 {
                let mut d = 1;
                while d <= 4 {
                    // Loading in reverse order because Intel is Little Endian Machine
                    let mask = [
                        u32_shuffle_mask(a, 0),
                        u32_shuffle_mask(b, a),
                        u32_shuffle_mask(c, a + b),
                        u32_shuffle_mask(d, a + b + c),
                    ];

                    // counting in the index must be 0 based (eg. length of 1 is `00`, not `01`), hence `a - 1`
                    let idx = (a - 1) << 6 | (b - 1) << 4 | (c - 1) << 2 | (d - 1);
                    assert!(a + b + c + d <= 16);
                    masks[idx] = (mask, a + b + c + d);
                    d += 1;
                }
                c += 1;
            }
            b += 1;
        }
        a += 1;
    }
    masks
}
