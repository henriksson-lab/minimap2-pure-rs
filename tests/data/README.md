# Test Fixtures

`chr11_bug_window.fa` and `chr11_bug_query.fq` are a reduced regression case
for a z-drop split alignment bug observed on chromosome 11 with the
`map-hifi` preset. The reference is a 16 kb window extracted from the original
chr11 reference around the mapped loci, and the query is the single read that
triggered the discrepancy.

The fixture is intentionally small enough to keep in the repository while still
exercising the long right-extension and split behavior that previously caused
Rust output to diverge from C minimap2.
