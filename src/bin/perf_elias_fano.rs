use toolkit::elias_fano::EliasFano;
use toolkit::gen_sequences::{gen_queries, gen_strictly_increasing_sequence};
use toolkit::utils::{TimingQueries, type_of};

use mem_dbg::*;

fn main() {
    let n_queries = 100_000;
    let n_runs = 3;
    let u = 1 << 32;

    for logn in 26..29 {
        let n = 1 << logn;
        let seq = gen_strictly_increasing_sequence(n, u);
        let queries = gen_queries(n_queries, n);
        let ef = EliasFano::from(&seq);
        let mut res = 0;
        let mut t = TimingQueries::new(n_runs, n_queries);
        for _ in 0..n_runs {
            t.start();
            for &q in queries.iter() {
                res += ef.select(q).unwrap();
            }
            t.stop();
        }
        let (t_min, t_max, t_avg) = t.get();
        println!(
            "SELECT1: [ds_name: {}, n: {}, bitsize: {:?}, min_time (ns): {}, max_time (ns): {}, avg_time (ns): {}]",
            type_of(&ef),
            n,
            u,
            t_min,
            t_max,
            t_avg
        );
        println!();
        let _ = ef.mem_dbg(DbgFlags::empty());
        println!("-------------------------------------\n");
        println!("Fake: {res}");
    }
}
