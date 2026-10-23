# `future-slot`
A no-std no-alloc way to box any `impl Future` into pinned memory, yielding a
concretely typed handle analogous to `Pin<Box<dyn Future>>`.

This way, many different future types, including the anonymous futures returned
by `async` blocks and functions, can be placed into a non-generic nameable
concrete type.

Includes a macro for wrapping a function call in such a slot, statically
allocating the correct amount of memory for the implied return type.

<img alt="Pinned" src="https://github.com/user-attachments/assets/f9533b16-a8da-4b2f-b0b5-6126b6ceea8c" width="100px" />
