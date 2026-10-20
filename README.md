# `future-slot`
A no-std no-alloc box-like datastructure `FutureSlot` that allows any `impl
Future` to be written into pinned memory, yielding a concretely typed handle
similar to `Pin<Box<dyn Future>>`.

This way, many different future types, including the anonymous futures returned
by `async` blocks and functions, can be placed into a non-generic nameable
concrete type.

Also includes a macro for wrapping a function call in such a slot, statically
allocating the correct amount of memory automatically.