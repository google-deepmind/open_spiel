# Cached Constants

Cached constant-builders return NumPy, not JAX.

Functions decorated with `@functools.cache` or `@functools.lru_cache` must not
build or return `jax.numpy` arrays. If the first call happens inside a JIT trace,
caching a JAX array can store a tracer and leak it into later calls. Return
`numpy.ndarray` values instead. JAX bakes NumPy arrays into XLA constants when
traced, so this avoids tracer leaks without a performance cost.

Use `np.array`, `np.asarray`, `np.arange`, and NumPy dtypes inside cached
constant-builders. Convert to JAX arrays at the point of traced computation only
when the surrounding code truly needs that conversion.

The lint guard walks the whole cached function body, including nested helper
definitions. If a nested helper needs `jnp.*`, hoist it out of the cached
function so the cached builder stays purely NumPy.
