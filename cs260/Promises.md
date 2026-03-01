> [!TIP]
> Tee-hee I got ahead of myself. This content doesn't come until **after React Pt. 2**. (Which I hadn't completed when I started taking these notes...no wonder I was so confused.)

# Promises

**HTML rendering is single-threaded**. Hence, long-running/blocking JS tasks should **work in the background, via a `Promise`**.

## Creating a promise

- Creating a promise is done by calling the **Promise object constructor** and passing it an **executor function**.
  - The executor function **runs immediately** when the Promise is created (you don't call it yourself), and it **runs asynchronously**.
- Because it's asynchronous, **the Promise constructor may return before the executor function runs**.

## Promise execution state

1. **Pending**: Currently running (asynchronously).
2. **Fulfilled**: Completed successfully.
3. **Rejected**: Failed to complete.

<br>

### Updating promise state (`resolve` & `reject`)

- The promise executor function takes two function parameters, `resolve` and `reject`.
  - Inside the executor function, you **call `resolve()` and `reject()` to update the Promises's state** to fulfilled & rejected, respectively.
    - Either function takes a single argument of any type, and is passed along to `then` or `catch` (see below).
      <!-- - The input your pass in to`resolve()` is passed as the input to `.then()`'s function (see below). -->
      <!-- - `reject()`'s  -->
  - You do not define any `resolve()` and `reject()` functions: the JS interpreter passes in built-in functions for these arguments itself.

### Handling promise result (`then`/`catch`/`finally`)

The Promise object has three functions that you chain together to define behavior for doing smth w/ the promises result. Each of these functions takes a function as its input.

- `.then()`: Called if the promise is fulfilled (`result()`).
  - Your input into `result()` becomes the input into the function you pass in for `.then()`.
- `.catch()`: Called if the promise is rejected (`reject()`).
  - Your input into `reject()` becomes the input into the function you pass in for `.catch()`.
- `.finally()`: Always called after promises finishes.

(You may notice that the naming of these functions resembles exception handling.)

## Example

```js
const coinToss = new Promise((resolve, reject) => {
    setTimeout(() => {
        if (Math.random() > 0.5) {
            resolve('success');
        } else {
            reject('ereror');
        }
    }, 10000);
});

coinToss
    .then((result) => console.log(`Coin toss result: ${result}`))
    .catch((err) => console.log(`Error: ${err}`))
    .finally(() => console.log('Toss completed'));
```

