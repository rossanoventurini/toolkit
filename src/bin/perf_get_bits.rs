use mem_dbg::*;

use std::hint::black_box;
use toolkit::AccessBin;
use toolkit::gen_sequences::gen_strictly_increasing_sequence;
const N_RUNS: usize = 500;

fn main() {
    let n_bits = 1026;
    let v = gen_strictly_increasing_sequence(n_bits / 2, n_bits);

    let bv = toolkit::bitvector::BitVec::from_iter(v);

    let _ = bv.mem_dbg(DbgFlags::empty());

    let mut timings = toolkit::utils::TimingQueries::new(N_RUNS, n_bits);

    for _ in 0..N_RUNS {
        timings.start();
        for i in 0..bv.len() {
            let _ = unsafe { black_box(bv.get_unchecked(i)) };
        }

        timings.stop();
    }
    let (_, _, avg) = timings.get_float();
    println!("One bit: avg: {:.2} ns", avg);

    let mut t = [0.0; 64 + 1];

    for bit_len in 1..=64 {
        let mut timings = toolkit::utils::TimingQueries::new(N_RUNS, n_bits - bit_len);

        for _ in 0..N_RUNS {
            timings.start();
            for _ in 0..bit_len {
                for pos in (0..bv.len() - bit_len).step_by(bit_len) {
                    let _ = black_box(unsafe { bv.get_bits_unchecked(pos, bit_len) });
                }
            }
            timings.stop();
        }

        let (_, _, avg) = timings.get_float();
        //println!("{bit_len:>2} bits: {:.2} ns", avg);
        t[bit_len] = avg;
    }

    let mut t_i = [0.0; 64 + 1];
    for bit_len in 1..=64 {
        let mut timings = toolkit::utils::TimingQueries::new(N_RUNS, n_bits - bit_len);

        for _ in 0..N_RUNS {
            timings.start();
            for _ in 0..bit_len {
                let mut iter = bv.ones();
                for _ in (0..bv.len() - bit_len).step_by(bit_len) {
                    let _ = black_box(unsafe { iter.get_bits_unchecked(bit_len) });
                }
            }
            timings.stop();
        }
        let (_, _, avg) = timings.get_float();
        t_i[bit_len] = avg;
    }

    for bit_len in 1..=64 {
        println!(
            "Sequential: {bit_len:>2} bits: {:.2} ns, Sequential with iterator: {:.2} ns",
            t[bit_len], t_i[bit_len]
        );
    }
}
