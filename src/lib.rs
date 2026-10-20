//! # `future-slot`
//! A no-std no-alloc box-like datastructure [`FutureSlot`] that allows any
//! `impl Future` to be written into pinned memory, yielding a concretely typed
//! [handle](`OwnedFuture`) similar to `Pin<Box<dyn Future>>`.
//!
//! This way, many different future types, including the anonymous futures
//! returned by `async` blocks and functions, can be placed into a non-generic
//! nameable concrete type.
//!
//! Also includes [a macro](pin_to_static) for wrapping a function call in such
//! a slot, statically allocating the correct amount of memory automatically.
#![no_std]

use core::{
    marker::{PhantomData, PhantomPinned},
    mem::{ManuallyDrop, MaybeUninit},
    pin::Pin,
    ptr::NonNull,
    task::{Context, Poll},
};

#[cfg(target_has_atomic = "8")]
pub mod atomic;

/// Trait smuggling the output type out of a function, where the type may be
/// anonymous.
///
/// Copy bound asserts no dtor, which allows it to be used as the argument to a
/// const function.
#[diagnostic::on_unimplemented(message = "implemented on functions with arity up to 12")]
trait HasOutput<Args>: Copy {
    type Output;
}
macro_rules! fns {
        {
            $(
            (
                $(
                    $arg:ident
                ),*
                $(,)?
            )
            ),*
            $(,)?
        } => {
            $(
                impl<Func, Output, $($arg),*> HasOutput<($($arg),*,)>
                    for Func
                    where
                        Func: Copy + FnOnce($($arg),*) -> Output
                {
                    type Output = Output;
                }
            )*
        };
    }
// Special-case zero arg version.
impl<Func, Output> HasOutput<()> for Func
where
    Func: Copy + FnOnce() -> Output,
{
    type Output = Output;
}
fns! {
    (A),
    (A,B),
    (A,B,C),
    (A,B,C,D),
    (A,B,C,D,E),
    (A,B,C,D,E,F),
    (A,B,C,D,E,F,G),
    (A,B,C,D,E,F,G,H),
    (A,B,C,D,E,F,G,H,I),
    (A,B,C,D,E,F,G,H,I,J),
    (A,B,C,D,E,F,G,H,I,J,K),
    (A,B,C,D,E,F,G,H,I,J,K,L),
    // I got bored at an arbitrary depth
}

/// Returns the size of the return type of the function `F`, even if the type is
/// anonymous.
#[expect(private_bounds)]
pub const fn size_of_output<Args, F: HasOutput<Args>>(_: F) -> usize {
    size_of::<F::Output>()
}
/// Returns the align of the return type of the function `F`, even if the type
/// is anonymous.
#[expect(private_bounds)]
pub const fn align_of_output<Args, F: HasOutput<Args>>(_: F) -> usize {
    align_of::<F::Output>()
}

/// Struct which is aligned to the amount given by it's generic parameter.
#[derive(Default, Clone, Copy)]
#[repr(C)]
struct Aligned<const N: usize>([<Self as AlignmentValid>::Aligned; 0])
where
    Self: AlignmentValid;
#[diagnostic::on_unimplemented(message = "only implemented for powers of two.")]
trait AlignmentValid {
    type Aligned: Copy + Default;
}

/// Implement [`AlignmentValid`] for a list of literal byte align amounts.
macro_rules! align_tys {
    ($(
        $align:literal
    ),*$(,)?) => {
        $(
            // anonymous module trick to prevent "Align" structs (implementation
            // detail) from leaking into the caller's scope.
            const _ : () = {
                #[derive(Copy, Clone, Default)]
                #[repr(align($align))]
                struct Align;
                impl $crate::AlignmentValid for Aligned<$align> {
                    type Aligned = Align;
                }
            };
        )*
    };
}
#[cfg(target_pointer_width = "16")]
align_tys! {
    1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,
    // The max alignment is 2**14, see 2**29 in the other set for details.
    16384,
}
// 64bit also has the same 2**29 align limit.
#[cfg(not(target_pointer_width = "16"))]
align_tys! {
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    // okay now this is getting rediculous
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    // r u ok
    131072,
    262144,
    524288,
    1048576,
    2097152,
    4194304,
    8388608,
    16777216,
    33554432,
    67108864,
    134217728,
    // who hurt you
    268435456,
    // The max alignment is NOT the max power of two that can fit in usize! This
    // is because the rust address space is actually only positive isizes (the
    // distance between two usize addresses must never exceed the capacity of an
    // isize). Hence, the max alignment is 2**29.
    536870912,

}

/// A slot which can be pinned in memory and used to name unnamable futures.
///
/// Use [`Self::write`] to insert a (possibly anonymous) [`impl Future`](Future)
/// and recieve a type-erased (but fully concrete!) [`OwnedFuture`] which polls
/// it.
///
/// This type does not implement any synchronization. It has a fully safe
/// interface due to having `&mut self` as a receiver. See the [`atomic`] module
/// for a synchronized version!
// Repr - lay out fields linearly as specified, so that bytes is placed
// immediately after, and thus aligned to the same amount as [`Aligned`].
#[repr(C)]
// We can be Clone + Copy since those take self by shared ref, which means that
// the slot is not full. Doesn't matter whether the concrete type is Copy or
// Clone, since we know there is no concrete type inside!
#[derive(Clone, Copy)]
#[expect(private_bounds)]
pub struct FutureSlot<const SIZE: usize, const ALIGN: usize>
where
    Aligned<ALIGN>: AlignmentValid,
{
    // Field order is important. Bytes must come immediately after _align.
    _align: Aligned<ALIGN>,
    bytes: MaybeUninit<[u8; SIZE]>,
    // Once we are pinned, we must never move. This allows !Unpin types to
    // reside in the slot.
    _pinned: PhantomPinned,
}
#[expect(private_bounds)]
impl<const SIZE: usize, const ALIGN: usize> FutureSlot<SIZE, ALIGN>
where
    Aligned<ALIGN>: AlignmentValid,
{
    /// Create an empty box. It is up to the caller to ensure that it is pinned
    /// before use.
    pub const fn new() -> Self {
        Self {
            _align: Aligned(unsafe { core::mem::zeroed() }),
            bytes: MaybeUninit::uninit(),
            _pinned: PhantomPinned,
        }
    }
    /// Insert a concrete type into the box.
    ///
    /// Statically asserts that the box has sufficient size/alignment for the
    /// concrete type, causing a compile error if not.
    pub fn write<'a, F: Future + Sized + 'a>(
        self: Pin<&'a mut Self>,
        // We take F by value, so we know it hasn't been pinned already.
        o: F,
    ) -> OwnedFuture<'a, F::Output> {
        // Note: The box is known to be empty, as the `mut` reciever requires
        // that no outstanding `OwnedFuture` occupies this box.
        const {
            assert!(
                core::mem::size_of::<F>() <= SIZE,
                "concrete type exceeds box size"
            );
            assert!(
                core::mem::align_of::<F>() <= ALIGN,
                "concrete type exceeds box alignment"
            );
        }

        // Safety: we promise not to move out of this.
        let this = unsafe { self.get_unchecked_mut() };

        // Copy o as bytes into our space.
        let o = ManuallyDrop::new(o);
        let size = core::mem::size_of_val(&o);
        unsafe {
            core::ptr::copy_nonoverlapping(
                (&raw const o).cast::<u8>(),
                this.bytes.as_mut_ptr().cast::<u8>(),
                size,
            )
        };
        // Moved out of, explicitly forget the old copy on the stack.
        core::mem::forget(o);

        // View the bytes as an `F` again for lifetime 'a. Remember, this is
        // pinned now, dont move it!
        let fut = unsafe { this.bytes.as_mut_ptr().cast::<F>().as_mut_unchecked::<'a>() };
        OwnedFuture {
            // Hand ownership over to this handle, which will handle dropping it
            // too.

            // Coerces into an erased type.
            slot: NonNull::from_mut(fut),
            _borrow: PhantomData,
        }
    }
}

/// A mutable reference to the future contained in a [`FutureSlot`]. It is a
/// non-allocating counterpart to `Box<dyn Future>`. It even has the same repr!
#[repr(transparent)]
pub struct OwnedFuture<'a, Output> {
    /// Pointer into Pinned slot memory, which contains an *owned* concrete type
    /// represented by VTable. This needs to be a pointer instead of a ref since
    /// we need to drop-in-place in our dtor.
    slot: NonNull<dyn Future<Output = Output> + 'a>,
    /// Marker for lifetime variance.
    _borrow: PhantomData<&'a mut (dyn Future<Output = Output> + 'a)>,
}
impl<'a, Output> OwnedFuture<'a, Output> {
    /// Helper function to access the future with its very nuanced lifetimes.
    fn get_fut<'borrow>(&'borrow mut self) -> Pin<&'borrow mut (dyn Future<Output = Output> + 'a)> {
        unsafe {
            let fut = self.slot.as_mut();
            Pin::new_unchecked(fut)
        }
    }
}
// We're a pointer! The pointer can move and we dont care, it's the pointee
// that's pinned.
impl<Output> Unpin for OwnedFuture<'_, Output> {}
impl<Output> Drop for OwnedFuture<'_, Output> {
    fn drop(&mut self) {
        // Safety: this drop glue only runs once.
        unsafe {
            self.slot.drop_in_place();
        }
    }
}
impl<Output> Future for OwnedFuture<'_, Output> {
    type Output = Output;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.get_fut().poll(cx)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    /// Naive polling executor.
    fn poll_once<F: core::future::Future>(future: &mut F) -> Option<F::Output>
    where
        F: Unpin,
    {
        match Pin::new(future).poll(&mut core::task::Context::from_waker(
            core::task::Waker::noop(),
        )) {
            Poll::Pending => None,
            Poll::Ready(r) => Some(r),
        }
    }
    /// Naive polling executor.
    fn dumb_block_on<F: core::future::Future>(future: F) -> F::Output {
        let mut future = core::pin::pin!(future);
        loop {
            if let core::task::Poll::Ready(r) = future.as_mut().poll(
                &mut core::task::Context::from_waker(core::task::Waker::noop()),
            ) {
                return r;
            }
        }
    }
    fn yield_now() -> impl Future<Output = ()> {
        let mut yielded = false;
        core::future::poll_fn(move |cx| {
            if yielded {
                Poll::Ready(())
            } else {
                yielded = true;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        })
    }

    #[test]
    fn output_size() {
        use core::convert::identity;

        assert_eq!(size_of_output(identity::<u128>), size_of::<u128>());
        assert_eq!(size_of_output(identity::<u64>), size_of::<u64>());
        assert_eq!(size_of_output(identity::<[u8; 25]>), 25);
    }
    #[test]
    fn output_align() {
        use core::convert::identity;

        assert_eq!(align_of_output(identity::<(u8, u128)>), align_of::<u128>());
        assert_eq!(align_of_output(identity::<u64>), align_of::<u64>());
        assert_eq!(align_of_output(identity::<[u8; 13]>), 1);
    }
    #[test]
    fn basic() {
        fn hello() -> impl Future<Output = usize> {
            async {
                let mut test = 0;
                for i in 0..10 {
                    test += i;
                    yield_now().await;
                }
                test
            }
        }
        // Test to make sure our future works normally
        assert_eq!(dumb_block_on(hello()), 45);

        let box_hello = || pin_to_static!(hello()).unwrap();

        // Box it once...
        assert_eq!(dumb_block_on(box_hello()), 45);
        // Re-use that box... Test this with miri to assure that no aliasing
        // occurs!
        assert_eq!(dumb_block_on(box_hello()), 45);
    }
    #[test]
    fn drop_glue() {
        use core::sync::atomic::{AtomicUsize, Ordering};
        /// Test drop glue, cancel safety, and preventing double-drops.
        static DROP_COUNTER: AtomicUsize = AtomicUsize::new(0);

        #[derive(Clone)]
        struct OnDrop;
        impl Drop for OnDrop {
            fn drop(&mut self) {
                DROP_COUNTER.fetch_add(1, Ordering::Relaxed);
            }
        }
        async fn hello(on_drop: OnDrop) {
            yield_now().await;
            drop(on_drop);
            yield_now().await;
        }

        let box_hello = |on_drop| pin_to_static!(hello(on_drop)).unwrap();

        // Normal operation;
        {
            let mut b = box_hello(OnDrop);

            assert_eq!(poll_once(&mut b), None);
            assert_eq!(DROP_COUNTER.load(Ordering::Relaxed), 0);
            assert_eq!(poll_once(&mut b), None);
            assert_eq!(DROP_COUNTER.load(Ordering::Relaxed), 1);
            assert_eq!(poll_once(&mut b), Some(()));
            drop(b);
            // Ensure didn't double-drop
            assert_eq!(DROP_COUNTER.load(Ordering::Relaxed), 1);
        }

        DROP_COUNTER.store(0, Ordering::Relaxed);

        // Cancel early.
        {
            let mut b = box_hello(OnDrop);

            assert_eq!(poll_once(&mut b), None);
            assert_eq!(DROP_COUNTER.load(Ordering::Relaxed), 0);
            drop(b);
            assert_eq!(DROP_COUNTER.load(Ordering::Relaxed), 1);
        }

        DROP_COUNTER.store(0, Ordering::Relaxed);

        // Cancel immediately..
        {
            let _ = box_hello(OnDrop);
            assert_eq!(DROP_COUNTER.load(Ordering::Relaxed), 1);
        }
    }
}
