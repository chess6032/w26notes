# JS async/await

`async` and `await` provide a more concise way to achieve asynchronization. They're still built on top of Promises, but provide some syntactical sugar.

## `async`

- You can **define a function w/ `async`** to turn it into an asynchronous function.
  - By declaring a function as `async`, you can **use `await` in its body**.
- Functions defined w/ `async` **always return a Promise**.
  - If your return is not explicitly a Promise, then at runtime return will be *turned into* a promise which is immediately resolved.

### Using `async` to make a function synchronous

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
// (you woud use the await keyword to resolve it).)
```



## `await`

- `await` wraps the **execution of a promise**.
    - `await` **blocks** until promise's state moves to **fulfilled**, at which point `await` would **return the result** of the promise.
    - If, while blocking, the promise's state moves to **rejected**, `await` will **throw an exception**.
- Use of `await` removes the need to chain functions.
- `await` can only be called at top level of your JS, OR inside a func defined w/ `async`.

> [!NOTE]
> You do not need to use 

### Example

**WITHOUT await:**

```js
somePromise()
    .then((result) => console.log(`result: ${result}`))
    .catch((err) => console.error(`error: ${err}`))
    .finally(() => console.log('done'));
```

**WITH await:**

```js
try {
    const result = await somePromise();
    console.log(`result: ${result}`);
} catch (err) {
    console.error(`error: ${err}`);
} finally {
    console.log('done');
}
```

> [!NOTE]
> **`await` blocks**, while Promise function **chaining does NOT block**.

### Cow example

```js
async function cow() {
    return new Promise((resolve) => {
        resolve('moo');
    });
}
const result = cow();
console.log(cow());       // OUTPUT: Promise { <pending> }
console.log(await cow()); // OUTPUT: moo
```

## async/await vs. Promise function chaining

- `await` BLOCKS, while promise function chaining does NOT block.

