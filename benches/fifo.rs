use cachekit::policy::fifo::FifoCache;
use cachekit::traits::CoreCache;
use criterion::{Criterion, criterion_group, criterion_main};

fn bench_fifo_insert_get(c: &mut Criterion) {
    c.bench_function("fifo_insert_get", |b| {
        b.iter(|| {
            let mut c = FifoCache::new(1024);
            for i in 0..1024 {
                c.insert(i, i);
            }
            for i in 0..1024 {
                let _ = c.get(&i);
            }
        })
    });
}

criterion_group!(benches, bench_fifo_insert_get);
criterion_main!(benches);
