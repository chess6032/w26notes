# Arrays in JS

Arrays in JS are **like lists in Python**.

- Arrays in JS are **zero-indexed**.
- Arrays in JS are **dynamically-sized**. They can grow & shrink.
- Arrays in JS **can hold elements of different types**.
- Arrays in JS are concatenated with the `.concat()` function, NOT the `+` operator you'd use in Python. (Sadge.)
- You **cannot negative index an array** in JS the way you can a list in Python.

```js
const a = ['a', 'b', 'c'];

console.log(a[1]);
// OUTPUT: b

console.log(a.length);
// OUTPUT: 3
```

## Array methods

Hey this is actually pretty useful:

| Function | Meaning                                                   | Example                       |
| -------- | --------------------------------------------------------- | ----------------------------- |
| push     | Add an item to the end of the array                       | `a.push(4)`                   |
| pop      | Remove an item from the end of the array                  | `x = a.pop()`                 |
| sort     | Run a function to sort an array in place                  | `a.sort((a,b) => b-a)`        |
| slice    | Return a sub-array                                        | `a.slice(1,-1)`               |
| values   | Creates an iterator for use with a `for of` loop          | `for (i of a.values()) {...}` |
| find     | Find the first item satisfied by a test function          | `a.find(i => i < 2)`          |
| forEach  | Run a function on each array item                         | `a.forEach(console.log)`      |
| reduce   | Run a function to reduce each array item to a single item | `a.reduce((a, c) => a + c)`   |
| map      | Run a function to map an array to a new array             | `a.map(i => i+i)`             |
| filter   | Run a function to remove items                            | `a.filter(i => i%2)`          |
| every    | Run a function to test if all items match                 | `a.every(i => i < 3)`         |
| some     | Run a function to test if any items match                 | `a.some(i => i < 1)`          |


All of these are **non-mutating** EXCEPT `.push()`, `.pop()`, and `.sort()`.

### Examples

```js
const a = [1, 2, 3];

console.log(a.map((i) => i + i));
// OUTPUT: [2,4,6]

console.log(a.reduce((v1, v2) => v1 + v2));
// OUTPUT: 6

console.log(a.sort((v1, v2) => v2 - v1));
// OUTPUT: [3,2,1]

a.push(4);
console.log(a.length);
// OUTPUT: 4
```