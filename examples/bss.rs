//! Test to ensure that static `AtomicFutureSlot` (and thus `FutureSlot`) reside
//! in the `.bss` segment. Change the size and align of [`_ATOMIC_FUTURE_SLOT`]
//! below, and observe the change in the segment map:
//!
//! ```bash
//! cargo build --release --example bss
//!     && readelf target/release/examples/bss -S | grep -A 1  "\.bss"
//! ```
//! (or use `rust-size`)
//!
//! You should see the size of the `.bss` segment change with the capacity of
//! the slot.
use future_slot::atomic::AtomicFutureSlot;

#[unsafe(no_mangle)]
static _ATOMIC_FUTURE_SLOT: AtomicFutureSlot<65536, 1024> = AtomicFutureSlot::new();

fn main() {
    println!("{:#?}", &raw const _ATOMIC_FUTURE_SLOT);
}
