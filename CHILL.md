# Chill Language Primer

Chill is a C-like language with strong influences from Rust. It's primary
objective is to make C semantics more consistent, remove implicit behaviors,
and add a distinction between nullable and non-nullable pointers.

```
fn add_two_ints(lhs: Int, rhs: Int): Int {
   return lhs + rhs; 
}
```
