//! Internally synchronized [`FutureSlot`]. Only available on targets with
//! [`AtomicBool`] support.

use core::{
    cell::UnsafeCell,
    pin::Pin,
    sync::atomic::{AtomicBool, Ordering},
    task::{Context, Poll},
};

use super::{Aligned, AlignmentValid, FutureSlot, OwnedFuture};

/// Like [`FutureSlot`], but with internal synchronization.
#[expect(private_bounds)]
pub struct AtomicFutureSlot<const SIZE: usize, const ALIGN: usize>
where
    Aligned<ALIGN>: AlignmentValid,
{
    slot: UnsafeCell<FutureSlot<SIZE, ALIGN>>,
    in_use: AtomicBool,
}
#[expect(private_bounds)]
impl<const SIZE: usize, const ALIGN: usize> AtomicFutureSlot<SIZE, ALIGN>
where
    Aligned<ALIGN>: AlignmentValid,
{
    /// Create an empty box. It is up to the caller to ensure that it is pinned
    /// before use.
    pub const fn new() -> Self {
        Self {
            slot: UnsafeCell::new(FutureSlot::new()),
            in_use: AtomicBool::new(false),
        }
    }
    /// Try to get write access to the slot. Call [`AtomicSlotLock::write`] to
    /// populate it. Returns None if there is still a outstanding
    /// [`AtomicOwnedFuture`] residing in this slot.
    pub fn try_lock(self: Pin<&'_ Self>) -> Option<AtomicSlotLock<'_, SIZE, ALIGN>> {
        if self.in_use.swap(true, Ordering::Acquire) {
            // Already in use.
            None
        } else {
            Some(AtomicSlotLock(self))
        }
    }
    /// Try to insert a future into this slot, returning None if there is still
    /// a outstanding [`AtomicOwnedFuture`] residing in it.
    ///
    /// `f` is only called when the slot is successfully acquired.
    pub fn try_write_with<'a, F: Future + Sized + 'a>(
        self: Pin<&'a Self>,
        f: impl FnOnce() -> F,
    ) -> Option<AtomicOwnedFuture<'a, F::Output>> {
        self.try_lock().map(|slot| slot.write(f()))
    }
    /// Try to insert a future into this slot, returning None if there is still
    /// a outstanding [`AtomicOwnedFuture`] residing in it.
    ///
    /// If `o` is expensive to compute, use [`Self::try_write_with`] instead to
    /// avoid constructing the future in cases when the slot is full anyway.
    pub fn try_write<'a, F: Future + Sized + 'a>(
        self: Pin<&'a Self>,
        o: F,
    ) -> Option<AtomicOwnedFuture<'a, F::Output>> {
        self.try_lock().map(|slot| slot.write(o))
    }
}
// We use internal synchronization, it's fine to pass reference to this slot
// across threads.
unsafe impl<const SIZE: usize, const ALIGN: usize> Sync for AtomicFutureSlot<SIZE, ALIGN> where
    Aligned<ALIGN>: AlignmentValid
{
}
/// A lock representing an empty [`AtomicFutureSlot`] which can be infallibly
/// [written to](Self::write).
///
/// This could be type-erased, but at the cost of compile-time size/align
/// verification. I don't currently have a use for it to be type erased anyway.
#[expect(private_bounds)]
pub struct AtomicSlotLock<'a, const SIZE: usize, const ALIGN: usize>(
    /// Invariant: We have exclusive access to this slot, and in_use is true.
    Pin<&'a AtomicFutureSlot<SIZE, ALIGN>>,
)
where
    Aligned<ALIGN>: AlignmentValid;
impl<const SIZE: usize, const ALIGN: usize> Drop for AtomicSlotLock<'_, SIZE, ALIGN>
where
    Aligned<ALIGN>: AlignmentValid,
{
    fn drop(&mut self) {
        // Well, the user didn't write to us. Mean. We can use relaxed ordering
        // since no writes were ever issued to the slot, and no memory barrier
        // needs to be inserted.
        self.0.in_use.store(false, Ordering::Relaxed);
    }
}
// We're a pointer! The pointer can move and we dont care, it's the pointee
// that's pinned.
impl<const SIZE: usize, const ALIGN: usize> Unpin for AtomicSlotLock<'_, SIZE, ALIGN> where
    Aligned<ALIGN>: AlignmentValid
{
}

#[expect(private_bounds)]
impl<'a, const SIZE: usize, const ALIGN: usize> AtomicSlotLock<'a, SIZE, ALIGN>
where
    Aligned<ALIGN>: AlignmentValid,
{
    /// Populate the locked slot.
    ///
    /// See [`FutureSlot::write`].
    pub fn write<F: Future + Sized + 'a>(self, o: F) -> AtomicOwnedFuture<'a, F::Output> {
        let slot = self.0.get_ref();
        let release_on_drop = ReleaseOnDrop(&slot.in_use);
        let slot = unsafe {
            // Safety: invariant of self is that we have exclusive access
            // (in_use is true).
            let slot = &mut *slot.slot.get();
            Pin::new_unchecked(slot)
        };
        let future = slot.write(o);
        AtomicOwnedFuture(future, release_on_drop)
    }
}

/// Sets the flag to `false` with `Release` ordering on drop.
struct ReleaseOnDrop<'a>(&'a AtomicBool);
impl Drop for ReleaseOnDrop<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release)
    }
}
/// Like [`OwnedFuture`], but with internal synchronization.
pub struct AtomicOwnedFuture<'a, Output: 'static>(
    // Ref into unsafe cell. UnsafeCell's docs carve out an exception for this,
    // allowing unused references to an *unsound* unsafecell to exist during
    // drop glue.
    OwnedFuture<'a, Output>,
    // Mark the unsafe cell as unused on drop, *after* the future is dropped.
    // Glue runs in field order. Unused because it's the autogen'd drop-glue
    // that handles the logic.
    #[expect(unused)] ReleaseOnDrop<'a>,
);
// We're a pointer! The pointer can move and we dont care, it's the pointee
// that's pinned.
impl<'a, Output> Unpin for AtomicOwnedFuture<'_, Output> {}
impl<'a, Output> Future for AtomicOwnedFuture<'a, Output> {
    type Output = Output;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // Since Self and OwnedFuture are Unpin, this is all safe APIs, yay!
        Pin::new(&mut self.0).poll(cx)
    }
}

/// Declare a static [`AtomicFutureSlot`] that maps the future type of the call
/// to a concrete type. Note that this allocates a permanent static section of
/// RAM, which is re-used only by the same callsite. It is possible to share RAM
/// between several callsites with direct use of the slot APIs.
///
/// The argument of the macro is the *function call* who's result should be
/// pinned, not just the result of the call, as the full name of the function is
/// used to inform the size and align of the slot.
///
/// Note that outer generics cannot be present in this function, so
/// `MyStruct::foo` will work, but `T::foo` will not - this is a purposeful
/// limitation built into the compiler.
///
/// Usage:
/// ```rust
/// use future_slot::pin_to_static;
/// fn something_generic<T>(t: T) -> impl Future<Output = T> {
///     async move { t }
/// }
/// async fn something_else() -> i32 {
///     something_generic(7).await
/// }
///
///
/// fn something_concrete(t: i32) {
///     // Demonstrates mapping impl Future to a concrete type. Despite the two
///     // branches returning different opaque types, they are both "boxed"
///     // into a `AtomicOwnedFuture<'static, i32>`
///     let _future = if false {
///         // The generic parameter must be specified explicity.
///         pin_to_static!(something_generic::<i32>(t)).unwrap();
///     } else {
///         // ...but it doesn't have to be generic.
///
///         // Note that this does not share memory with the above even though
///         // it could! Use the API directly to achieve that effect.
///         pin_to_static!(something_else()).unwrap();
///     };
/// }
/// ```
///
/// Expands to the return value of a call to
/// [`AtomicFutureSlot::try_write_with`], which may be [`None`] if the last
/// call's future still exists.
#[macro_export]
macro_rules! pin_to_static {
    (
        // FIXME: no leading ::

        // Function path (there's no built in type for this?!)
        $first:ident
        $(
            ::$path:ident
        )*
        // Generic args, which cannot be implicit.
        $(
            // FIXME: no lifetimes.

            // FIXME: no const generics.
            ::<
            $(
                $generic:ty
            ),*
            $(,)?
            >
        )?
        // Actual arguments, for when it gets called.
        (
        $(
            $args:expr
        ),*
        $(,)?
        )
    ) => {
        {
            const SIZE: usize = $crate::size_of_output($first$(::$path)*$(::<$($generic),*>)?);
            const ALIGN: usize = $crate::align_of_output($first$(::$path)*$(::<$($generic),*>)?);
            const SLOT: ::core::pin::Pin<&'static $crate::atomic::AtomicFutureSlot<SIZE, ALIGN>> = {
                static SLOT: $crate::atomic::AtomicFutureSlot<SIZE, ALIGN> = $crate::atomic::AtomicFutureSlot::new();
                ::core::pin::Pin::static_ref(&SLOT)
            };

            SLOT.try_write_with(
                move || {
                    ($first$(::$path)*$(::<$($generic),*>)?)
                    (
                        $($args),*
                    )
                }
            )
        }
    };
}
