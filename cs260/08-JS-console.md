# JavaScript Console

## `console.log()`

Prints stuff to console.

## `console.time()` and `console.timeEnd()`

- `console.time(timerLabel)`: starts a timer referred to w/ `timerLabel`.
- `console.timeEnd(timerLabel)`: ends the timer.

### Example

```js
console.time('demo time');

// ... some code that takes a long time.

console.timeEnd('demo time');
// OUTPUT: demo time: 12.74 ms
```

## `console.count()`

- `console.count(label)` increments &  prints out an internal integer, indicating how many times `console.count()` was ran w/ that `label` input.
  - (`label` is optional.)

### Example

```js
for (let i = 0; i < 5; i++) {
  console.count("myLabel");
}

// OUTPUT:
// myLabel: 1
// myLabel: 2
// myLabel: 3
// myLabel: 4
// myLabel: 5
```
