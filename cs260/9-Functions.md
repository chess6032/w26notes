# Functions in JS


In JS:

- Functions are defined with the `function` keyword.
- Functions are **first-class objects**.
- Functions **cannot be overloaded**.
  - When multiple functions are defined w/ the same name, the last one defined overwrites previous definitions, like in Python.
- Functions may be **defined inside other functions**, like in Python.
  - These are called **"inner functions"**.
- **Parameters can be given a default value**, like in Python or C++.

## Anonymous functions

You can create a function inline. When you do this, that function doesn't have a name associated to it, hence why it's called an "anonymous" function.

With "arrow functions", you can create anonymous functions with minimal syntax.

### Example

The function `doMath` takes a function parameter `operation`:

```js
function doMath(operation, a, b) {
    return operation(a, b);
}
```

Below are examples of passing anonymous functions in as the input for `operation`.

```js
// Anonymous function assigned to a variable
const add = function (a, b) {
    return a + b;
};

const result = doMath(add, 5, 3);
console.log(result); // OUTPUT: 8
```

```js
// Anonymous function passed in as a parameter
const result = doMath(function (a, b) { return a - b; }, 5, 3);
console.log(result);
```

```js
// Arrow function passed in as a parameter
const result = doMath((a, b) => a - b, 5, 3);
console.log(result);
```