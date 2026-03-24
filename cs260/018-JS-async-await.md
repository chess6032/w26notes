# JS async/await

`async` and `await` are two more tools for asynchronous execution. They are still built on top of promises, but they provide some syntactical sugar&mdash;and some extra functionality.

## `await`

- `await` wraps the **execution of a promise**. It allows you to **block until the promise finishes executing**.
    - `await` **blocks** until promise's state moves to **fulfilled**, at which point `await` would **return the result** of the promise.
    - If, while blocking, the promise's state moves to **rejected**, `await` will **throw an exception**.
- Use of `await` removes the need to chain functions.
- `await` can only be called at top level of your JS, OR inside a func defined w/ `async`.

Here's a quick example of using `await` to block logic until a Promise finishes.

```js
const wait10secs = new Promise((resolve) => {
    setTimeout(() => resolve('done'), 1000);
});
console.log(wait10secs);       // OUTPUT: Promise { <pending> }
const result = await wait10secs;
console.log(wait10secs); // OUTPUT: Promise { 'done' }
console.log(result); // OUTPUT: done
console.log('ten seconds have passed'); // OUTPUT: ten seconds have passed
```

> [!NOTE]
> `await` can be called on ANY promise, but (it sounds like) we most often use it on *functions* that *return a promise*. (That could be totally wrong tho.)


### `await` vs. promise function chaining

- With promises, you had to use function chaining. But you can **wrap a promise executed w/ `await` w/ a try/catch/finally block** for the same result (but arguably much cleaner and more intuitive).
- `await` blocks, while Promise function chaining does not.

#### Example

Promise:

```js
const somePromise = new Promise((resolve) => {
    console.log('a promise is made...');
    setTimeout(() => resolve('...and it is kept'), 5000);
});
```

**WITHOUT await:**

```js
somePromise
    .then((result) => console.log(`RESULT: ${result}`))
    .catch((err) => console.error(`ERROR: ${err}`))
    .finally(() => console.log('DONE'));

console.log('---- skibidi rizz ----');
```
    a promise is made...
    ---- skibidi rizz ----
    RESULT: ...and it is kept
    DONE

**WITH await:**

```js
try {
    const result = await somePromise;
    console.log(`RESULT: ${result}`);
} catch (err) {
    console.error(`ERROR: ${err}`);
} finally {
    console.log('DONE');
}

console.log('---- skibidi rizz ----');
```

    a promise is made...
    RESULT: ...and it is kept
    DONE
    ---- skibidi rizz ----

> [!IMPORTANT]
> **`await` blocks**, while Promise **function chaining does NOT block**.

## `async`

- You can **define a function w/ `async`** to allow you to **use `await` in its body**.
  - (The instruction material says ab this: "Basically this turns any function into an asynchronous function, so that it can in turn make asynchronous requests." ...Not rly sure what they mean by that, but that's cool ig.)
- Functions defined w/ `async` **always return a Promise**.
  - If your return is not explicitly a Promise, then at runtime return will be *turned into* a promise which is immediately resolved.

### Demonstration: `async` makes a function return a Promise

```js
function cow() {
    return 'moo';
}
console.log(cow());
// OUTPUT: moo
```

```js
async function cow() {
    return 'moo';
}
// OUTPUT: Promise {<fulfilled>: 'moo'}
// (the return 'moo' is turned into a promise that is immediately resolved.)
```

```js
async function cow() {
    return new Promise((resolve) => {
        resolve('moo');
    });
}
console.log(cow());
// OUTPUT: Promise {<pending>}
// (promise returned by cow() has not resolved.
// (you would use the await keyword to resolve it).)
```

<!-- ### `await` w/ `async` functions

```js
async function cow() {
    return new Promise((resolve) => {
        resolve('moo');
    });
}
const result = cow();
console.log(result);       // OUTPUT: Promise { <pending> }
console.log(await result); // OUTPUT: moo
``` -->

> [!NOTE]
> A promise starts executing the moment it is created. **By wrapping your Promise in a function that returns it, you can choose specifically when that Promise will start executing**&mdash;because it won't be created until you call that function. (In contrast, if you instantiated a new promise outside a function, its execution would always begin right after where you put its logic.)

## Example: using `await` to synchronize two Promises

Consider a `fetch` web API on an endpoint that returns JSON, which requires TWO promises: one for the network call, and another for converting the result to JSON. BUT, the JSON promise must execute AFTER the network call promise, which makes things trickier.

Only using promises, the code would look smth like this:

```js
const httpPromise = fetch('https://simon.cs260.click/api/user/me');
const jsonPromise = httpPromise.then((r) => r.json());
jsonPromise.then((j) => console.log(j));
console.log('done');
```
    done
    {email: 'bud@mail.com', authenticated: true}

With async/await, you clarify the code's intent by hiding the Promise syntax&mdash;and bock execution until the promise is resolved:

```js
const httpResponse = await fetch('https://simon.cs260.click/api/user/me');
const jsonResponse = httpResponse.json();
console.log('done');
```

    {email: 'bud@mail.com', authenticated: true}
    done