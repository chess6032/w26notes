# JS: `setTimeout` and `setInterval`

## `setTimeout()`

- `setTimeout()` allows you to **delay the execution of something** until after a period of time has expired.
- Params: `func`, `time` (ms).
  - Runs `func()` after `time` milliseconds have passed.
- `setTimeout()` **does not block**:
  - After being called, other JS code will continue running.
  - Then, after the time period passes, the current JS code will switch and the setTimeout's function will run.

### Example

```js
setTimeout(() => console.log('time is up'), 2000);

console.log('hey');

// OUTPUT:
// hey
// time is up
```

## `setInterval()`

- `setInterval()` allows you to **execute code periodically** at a given time interval.
- Params: `func`, `time` (ms).
  - Every `time` milliseconds, runs `func()`.
- Use `clearInterval()` to cancel `setInterval()`:s
  - You must capture the result of `setInterval()` and pass it into `clearInterval()` (see below).

```js
const interval = setInterval(() => console.log('do something'), 1000);
// print "do something" every 1000ms.

setTimeout(() => clearInterval(interval), 5000);
// stop "do something" interval after 5000ms.
```
