# this-state

this-state provides a way to store state in a thread-safe manner
as well as a way to asynchronously wait for state changes.

# Examples
The example below uses the following state:

````rust
#[derive(Clone, Debug, PartialEq)]
enum MyState {
    A,
    B,
    C
}
````

## Waiting for a state change

````rust
let state = State::new(MyState::A);

let state_clone = state.clone();
tokio::spawn(async move {
    // do some work
    state_clone.set(MyState::B);
    // do some more work
    state_clone.set(MyState::C);
});

state.wait_for_state(MyState::C).await;

assert_eq!(state.get(), MyState::C);
````