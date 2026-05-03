use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_encode(c: &mut Criterion) {
    let data = include_str!("../README.md").repeat(800);
    c.bench_function("encode_throughput", |b| {
        b.iter(|| {
            let tokens = ringtail_ai::encode(black_box(&data));
            black_box(tokens);
        })
    });
}

criterion_group!(benches, bench_encode);
criterion_main!(benches);
