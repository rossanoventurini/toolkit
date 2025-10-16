use toolkit::utils::{type_of, TimingQueries};
use toolkit::gen_sequences::gen_queries;
use toolkit::FenwickTree;

use num::Zero;
use std::ops::{AddAssign, SubAssign};
use std::time::Instant;

const N_RUNS: usize = 5;
const DEFAULT_N_QUERIES: usize = 10_000_000;

use clap::Parser;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    n: usize,
    #[arg(default_value_t = DEFAULT_N_QUERIES)]
    n_queries: usize,
    #[arg(short, long)]
    update: bool,
    #[arg(short, long)]
    sum: bool,
    #[arg(short, long)]
    iter: bool,
}

fn test_update<T, const HOLES: bool>(ds: &mut FenwickTree<T, HOLES>, queries: &[usize], v: T)
where
    T: Copy + Zero + AddAssign+ SubAssign,
{
    let n = ds.len();

    let mut t = TimingQueries::new(N_RUNS, queries.len());
    for _ in 0..N_RUNS {
        t.start();
        for &i in queries.iter() {
            ds.add_at(i, v);
        }
        t.stop();
    }

    let (t_min, t_max, t_avg) = t.get();
    println!("UPDATE: [ds_name: {}, n: {}, min_time (ns): {}, max_time (ns): {}, avg_time (ns): {}, num_queries: {}, num_runs: {}]", type_of(&ds), n, t_min, t_max, t_avg, queries.len(), N_RUNS);
}

fn test_sum<T, const HOLES: bool>(ds: &FenwickTree<T, HOLES>, queries: &[usize])
where
    T: Copy + Zero + AddAssign + SubAssign + std::fmt::Display,
{
    let mut r = T::zero();
    let mut t = TimingQueries::new(N_RUNS, queries.len());
    let n = ds.len();

    for _ in 0..N_RUNS {
        t.start();
        for &i in queries.iter() {
            r += ds.sum(..i);
        }
        t.stop();
    }

    let (t_min, t_max, t_avg) = t.get();
    println!("SUM: [ds_name: {}, n: {}, min_time (ns): {}, max_time (ns): {}, avg_time (ns): {}, num_queries: {}, num_runs: {}]", type_of(&ds), n, t_min, t_max, t_avg, queries.len(), N_RUNS);

    println!("IGNORE: {r}");
}

fn test_iter<T, const HOLES: bool>(ds: &FenwickTree<T, HOLES>)
where
    T: Copy + Zero + AddAssign + SubAssign + std::fmt::Display,
{
    let n = ds.len();
    let mut t = TimingQueries::new(N_RUNS, n);
    let mut r = T::zero();

    for _ in 0..N_RUNS {
        t.start();
        for val in ds.iter() {
            r += val;
        }
        t.stop();
    }

    let (t_min, t_max, t_avg) = t.get();

    println!("ITER: [ds_name: {}, n: {}, min_time (ns): {}, max_time (ns): {}, avg_time (ns): {}, num_queries: {}, num_runs: {}]", type_of(&ds), n, t_min, t_max, t_avg, n, N_RUNS);
    
    println!("IGNORE: {r}");
}

fn test_prefix_sums<T, const HOLES: bool>(ds: &FenwickTree<T, HOLES>)
where
    T: Copy + Zero + AddAssign + SubAssign + std::fmt::Display,
{
    let n = ds.len();
    let mut min_time = u64::MAX;
    let mut max_time = 0u64;
    let mut total_time = 0u64;
    let mut r = T::zero();

    for _ in 0..N_RUNS {
        let start = Instant::now();
        for val in ds.prefix_sums() {
            r += val;
        }
        let elapsed = start.elapsed().as_nanos() as u64;
        
        min_time = min_time.min(elapsed);
        max_time = max_time.max(elapsed);
        total_time += elapsed;
    }

    let avg_time = total_time / N_RUNS as u64;
    let time_per_element = avg_time / n as u64;

    println!("PREFIX_SUMS: [ds_name: {}, n: {}, min_time (ns): {}, max_time (ns): {}, avg_time (ns): {}, time_per_element (ns): {}, num_runs: {}]", 
        type_of(&ds), n, min_time, max_time, avg_time, time_per_element, N_RUNS);
    
    println!("IGNORE: {r}");
}

fn main() {
    let args = Args::parse();

    let n = args.n;
    let update_queries = gen_queries(args.n_queries, n);
    let sum_queries = gen_queries(args.n_queries, n);
    println!("n = {}, n_queries = {}", n, args.n_queries);

    println!("\n=== FenwickTree with HOLES=true ===");
    let mut ft = FenwickTree::<i64, true>::with_len(n);
    
    // Populate with some values for more realistic iteration
    for i in (0..n).step_by(n / 100 + 1) {
        ft.add_at(i, (i as i64 % 1000) + 1);
    }
    
    if args.update {
        test_update(&mut ft, &update_queries, 42);
    }
    if args.sum {
        test_sum(&ft, &sum_queries);
    }
    if args.iter {
        test_iter(&ft);
        test_prefix_sums(&ft);
    }

    println!("\n=== FenwickTree with HOLES=false ===");
    let mut ft = FenwickTree::<i64, false>::with_len(n);
    
    // Populate with some values for more realistic iteration
    for i in (0..n).step_by(n / 100 + 1) {
        ft.add_at(i, (i as i64 % 1000) + 1);
    }
    
    if args.update {
        test_update(&mut ft, &update_queries, 42);
    }
    if args.sum {
        test_sum(&ft, &sum_queries);
    }
    if args.iter {
        test_iter(&ft);
        test_prefix_sums(&ft);
    }
}