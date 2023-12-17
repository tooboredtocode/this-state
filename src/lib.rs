//! A library for managing state changes.
//!
//! # Examples
//! The examples below use the following state:
//! ```
//! #[derive(Clone, Debug, PartialEq)]
//! enum MyState {
//!     A,
//!     B,
//!     C
//! }
//! ```
//!
//! ## Waiting for a state change
//!
//! ```
//! # use this_state::State;
//! # use tokio::runtime;
//! #
//! # #[derive(Clone, Debug, PartialEq)]
//! # enum MyState {
//! #     A,
//! #     B,
//! #     C
//! # }
//! #
//! # let mut rt = runtime::Builder::new_current_thread()
//! #     .enable_all()
//! #     .build()
//! #     .unwrap();
//! #
//! # rt.block_on(async {
//! let state = State::new(MyState::A);
//!
//! let state_clone = state.clone();
//! tokio::spawn(async move {
//!     // do some work
//!     # tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
//!     state_clone.set(MyState::B);
//!     // do some more work
//!     # tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
//!     state_clone.set(MyState::C);
//! });
//!
//! state.wait_for_state(MyState::C).await;
//!
//! assert_eq!(state.get(), MyState::C);
//! # })
//! ```

use std::cell::UnsafeCell;
use std::fmt;
use std::future::Future;
use std::marker::PhantomPinned;
use std::ops::Deref;
use std::pin::Pin;
use std::ptr::{addr_of_mut, NonNull};
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use parking_lot::{RwLock, RwLockReadGuard};

use crate::util::linked_list;
use crate::util::linked_list::LinkedList;

mod util;

/// A thread-safe state, that can be used to share application state globally.
///
/// It is similar to a `RWLock<S>`, but it also allows asynchronous waiting for state changes.
/// This can be useful to coordinate between different parts of an application.
#[derive(Clone)]
pub struct State<S> {
    /// The state wraps an `Arc` to allow easy cloning.
    inner: Arc<StateInner<S>>,
}

/// The inner state of a `State`, this contains the actual state and the wait queue.
struct StateInner<S> {
    /// The actual state.
    state: RwLock<S>,
    /// The wait queue, containing all tasks waiting for a state change.
    waiters: RwLock<LinkedList<Waiter, <Waiter as linked_list::Link>::Target>>,
    /// Callback that is called when the state changes.
    on_change: Box<dyn Fn(&S, &S) + 'static>,
}

/// An entry in the wait queue.
struct Waiter {
    /// Indicates whether the task is queued on the wait queue.
    queued: bool,

    /// Task waiting on a state change.
    waker: Option<Waker>,

    /// Intrusive linked-list pointers.
    pointers: linked_list::Pointers<Waiter>,

    /// Should not be `Unpin`.
    _p: PhantomPinned,
}

/// A future that completes when the state matches the given predicate.
/// This is returned by `State::wait_for`.
///
/// # Notes
/// Unlike most futures, this future can be polled multiple times, even after it has completed.
#[must_use = "futures do nothing unless you `.await` or poll them"]
pub struct StateFuture<S, C> {
    state: State<S>,
    waiter: UnsafeCell<Waiter>,
    wait_for: C,
}

/// A reference to the current state, returned by `State::get_ref`.
/// It wraps a `RwLockReadGuard` and can be used to avoid cloning the state.
#[must_use]
pub struct StateRef<'a, S>(RwLockReadGuard<'a, S>);

unsafe impl<S> Send for State<S> {}
unsafe impl<S> Sync for State<S> {}

unsafe impl<S, C> Send for StateFuture<S, C> {}
unsafe impl<S, C> Sync for StateFuture<S, C> {}

impl<S> State<S> {
    /// Creates a new state.
    pub fn new(state: S) -> Self {
        Self {
            inner: Arc::new(StateInner {
                state: RwLock::new(state),
                waiters: RwLock::new(LinkedList::new()),
                on_change: Box::new(|_, _| {}),
            }),
        }
    }

    /// Creates a new state with the given `on_change` callback.
    ///
    /// # Notes
    /// The callback is not called when the state is set for the first time, as well as on
    /// the `State::update` method. You must call the callback manually in these cases.
    pub fn new_with_on_change(state: S, on_change: impl Fn(&S, &S) + 'static) -> Self {
        Self {
            inner: Arc::new(StateInner {
                state: RwLock::new(state),
                waiters: RwLock::new(LinkedList::new()),
                on_change: Box::new(on_change),
            }),
        }
    }

    /// Returns a reference to the current state.
    /// This can be used if the state does not implement `Clone` or if you want to avoid cloning.
    pub fn get_ref(&self) -> StateRef<S> {
        StateRef(self.inner.state.read())
    }

    /// Returns a future that completes when the state matches the given predicate.
    pub fn wait_for<C>(&self, wait_for: C) -> StateFuture<S, C>
    where
        C: Fn(&S) -> bool,
    {
        StateFuture::new(
            State {
                inner: self.inner.clone(),
            },
            wait_for,
        )
    }

    /// Sets the state to the given value.
    pub fn set(&self, state: S) {
        let mut write = self.inner.state.write();
        (self.inner.on_change)(&*write, &state);
        *write = state;
        drop(write);
        self.wake_waiters();
    }

    /// Updates the state using the given function.
    /// This avoids having to create a new state value, which can be useful for large state values.
    ///
    /// # Notes
    /// This *DOES NOT* call the `on_change` callback, as it is not possible to get the old state.
    pub fn update(&self, f: impl FnOnce(&mut S)) {
        let mut write = self.inner.state.write();
        f(&mut write);
        drop(write);
        self.wake_waiters();
    }

    /// Wakes all waiters.
    fn wake_waiters(&self) {
        let mut waiters = self.inner.waiters.write();

        for mut waiter in waiters.iter() {
            // Safety: list lock is still held.
            let waiter = unsafe { waiter.as_mut() };

            assert!(waiter.queued);

            if let Some(waker) = waiter.waker.take() {
                waker.wake();
            }
        }
    }
}

impl<S> State<S>
where
    S: Clone,
{
    /// Returns a clone of the current state.
    /// This is particularly useful for `State`s that implement `Copy`.
    pub fn get(&self) -> S {
        self.get_ref().clone()
    }
}

impl<S> State<S>
where
    S: PartialEq<S>,
{
    /// Returns a future that resolves when the state is equal to the given value.
    pub fn wait_for_state(&self, wait_for: S) -> StateFuture<S, impl Fn(&S) -> bool> {
        StateFuture::new(
            State {
                inner: self.inner.clone(),
            },
            move |s| wait_for.eq(s),
        )
    }
}

impl<S, O> PartialEq<O> for State<S>
where
    S: PartialEq<O>,
{
    fn eq(&self, other: &O) -> bool {
        self.get_ref().eq(other)
    }
}

impl<S: fmt::Debug> fmt::Debug for State<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("State").field(&self.get_ref()).finish()
    }
}

impl<S: Default> Default for State<S> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl Waiter {
    fn new() -> Waiter {
        Waiter {
            queued: false,
            waker: None,
            pointers: linked_list::Pointers::new(),
            _p: PhantomPinned,
        }
    }
}

/// # Safety
///
/// `Waiter` is forced to be !Unpin.
unsafe impl linked_list::Link for Waiter {
    type Handle = NonNull<Waiter>;
    type Target = Waiter;

    fn as_raw(handle: &NonNull<Waiter>) -> NonNull<Waiter> {
        *handle
    }

    unsafe fn from_raw(ptr: NonNull<Waiter>) -> NonNull<Waiter> {
        ptr
    }

    unsafe fn pointers(target: NonNull<Waiter>) -> NonNull<linked_list::Pointers<Waiter>> {
        let me = target.as_ptr();
        let field = addr_of_mut!((*me).pointers);
        NonNull::new_unchecked(field)
    }
}

impl<S, C> StateFuture<S, C> {
    /// Returns a reference to the current state.
    ///
    /// This may be useful to create other futures or simply getting the current state.
    pub fn state(&self) -> &State<S> {
        &self.state
    }

    fn queue_waker(self: Pin<&mut Self>, waker: &Waker) {
        // Acquire a read lock so we guarantee the list is not used while we're modifying the waiter.
        let lock = self.state.inner.waiters.read();
        // Safety: We have a read lock, so the list is not being modified, and only one thread can
        // poll the future at a time.
        let waiter = unsafe { &mut *self.waiter.get() };

        if !waiter.queued {
            drop(lock);
            // Acquire a write lock to add ourselves to the list.
            let mut lock = self.state.inner.waiters.write();

            // Note: We dont need to check if we got queued in the meantime the lock was acquired,
            // since only the future itself adds the waiter to the list.
            waiter.queued = true;
            waiter.waker = Some(waker.clone());

            lock.push_front(unsafe { NonNull::new_unchecked(waiter) });
            return;
        }

        // Safety: list lock is held.
        match waiter.waker {
            Some(ref w) if w.will_wake(waker) => {}
            _ => {
                waiter.waker = Some(waker.clone());
            }
        }
    }

    fn remove_waiter(&self) {
        let waiters = self.state.inner.waiters.read();

        let waiter = unsafe { &mut *self.waiter.get() };
        if !waiter.queued {
            // Return since the waiter is not queued.
            return;
        }

        drop(waiters);
        let mut waiters = self.state.inner.waiters.write();

        // We don't have to check if the waiter was dropped in the meantime, since only the future
        // itself removes the waiter from the list.

        unsafe {
            // Safety: waiter is not null and !Unpin.
            let nonnull = NonNull::new_unchecked(self.waiter.get());
            // Safety: we have checked that the waiter is queued and therefore in the list.
            waiters.remove(nonnull);
        }

        drop(waiters);
    }
}

impl<S, C> StateFuture<S, C>
where
    C: Fn(&S) -> bool,
{
    fn new(state: State<S>, wait_for: C) -> Self {
        Self {
            state,
            waiter: UnsafeCell::new(Waiter::new()),
            wait_for,
        }
    }
}

impl<S, C> Future for StateFuture<S, C>
where
    C: Fn(&S) -> bool,
{
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let state = self.state.inner.state.read();
        if (self.wait_for)(&*state) {
            drop(state);
            // remove the waiter from the list, since we're done waiting.
            self.remove_waiter();
            return Poll::Ready(());
        }
        drop(state);

        self.queue_waker(cx.waker());
        Poll::Pending
    }
}

impl<S, C> Drop for StateFuture<S, C> {
    fn drop(&mut self) {
        // remove the waiter from the list, since we're done waiting.
        self.remove_waiter();
    }
}

impl<'a, S> Deref for StateRef<'a, S> {
    type Target = S;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, S: fmt::Debug> fmt::Debug for StateRef<'a, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl<'a, S: fmt::Display> fmt::Display for StateRef<'a, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use tokio::time;

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum StateEnum {
        A,
        B,
        C,
    }

    #[test]
    fn test_state() {
        let state = State::new(StateEnum::A);

        assert_eq!(state.get(), StateEnum::A);

        state.set(StateEnum::B);

        assert_eq!(state.get(), StateEnum::B);
    }

    #[tokio::test]
    async fn test_future1() {
        let state = State::new(StateEnum::A);

        let state_clone = state.clone();
        let fut = tokio::spawn(async move { state_clone.wait_for_state(StateEnum::B).await });

        assert_eq!(state.get(), StateEnum::A);

        state.set(StateEnum::B);

        assert_eq!(state.get(), StateEnum::B);
        // Wait for the future to finish.
        time::sleep(time::Duration::from_millis(100)).await;
        assert!(fut.is_finished());
    }

    #[tokio::test]
    async fn test_future2() {
        let state = State::new(StateEnum::A);

        let state_clone = state.clone();
        let fut = tokio::spawn(async move { state_clone.wait_for_state(StateEnum::B).await });

        assert_eq!(state.get(), StateEnum::A);

        state.set(StateEnum::C);

        assert_eq!(state.get(), StateEnum::C);
        // Wait for the future to potentially finish.
        time::sleep(time::Duration::from_millis(100)).await;
        assert!(!fut.is_finished());

        state.set(StateEnum::B);

        assert_eq!(state.get(), StateEnum::B);
        // Wait for the future to finish.
        time::sleep(time::Duration::from_millis(100)).await;
        assert!(fut.is_finished());
    }

    #[tokio::test]
    async fn multiple_waiters() {
        const NUM_WAITERS: usize = 100;

        let state = State::new(StateEnum::A);

        let mut handles = Vec::with_capacity(NUM_WAITERS);
        for _ in 0..NUM_WAITERS {
            let state_clone = state.clone();
            let handle =
                tokio::spawn(async move { state_clone.wait_for_state(StateEnum::B).await });
            handles.push(handle);
        }

        assert_eq!(state.get(), StateEnum::A);

        state.set(StateEnum::C);

        assert_eq!(state.get(), StateEnum::C);
        // Wait for the future to potentially finish.
        time::sleep(time::Duration::from_millis(100)).await;
        assert!(!handles.iter().any(|h| h.is_finished()));

        state.set(StateEnum::B);

        assert_eq!(state.get(), StateEnum::B);
        // Wait for the future to finish.
        time::sleep(time::Duration::from_millis(100)).await;
        assert!(handles.iter().all(|h| h.is_finished()));
    }
}
