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

## Closure

- A "closure" is a function that **retains access to variables from the scope it was defined in**. 
- **All functions in JS form closures**, but arrow functions (see below) are special in what their `this` pointer references.
  - Regular functions' `this` pointer references **where they're CALLED**.
  - Arrow functions' `this` pointer references **where they're CREATED**.
    - In the instructions: "arrow functions inherit the `this` pointer from the scope in which they're created."
- Closure is clutch for working w/ JS in HTML.

### Example

Here's an example of closure w/ arrow functions (see below).

```js
function makeClosure(init) {
    let closureValue = init;
    return () => {
        return `closure ${++closureValue}`;
    };
}

const closure = makeClosure(0);

console.log(closure());
// OUTPUT: closure 1

console.log(closure());
// OUTPUT: closure 2
```

# Arrow functions

Arrow functions in JS provide an abbreviated way of creating anonymous functions.

<pre><code>(<i>parameters</i>) => <i>function body</i>;</code></pre>

I find it helps to see examples. Here is a function that takes no parameters and always returns 67.

```js
() => 67;
```

Here's a function that takes a single parameter `n` and returns its square.

```js
(n) => n*n;
```

With curly brackets, you can have multiple statements in the function body.

```js
(n) => {
    console.log(`squaring ${n}`);
    return n*n;
}
```

## Return value

- If your function body is NOT inside curly brackets, the `return` keyword is optional. 
  - Without it, it returns whatever expression you write after `=>`.
- If you DO define your function body inside curly brackets, you must have a `return` statement inside your arrow function's body for it to return anything.

```js
() => 67;
// RETURNS 67

() => {
    67;
}
// doesn't return anything.
// (technically speaking, its return value would be considered UNDEFINED, I think.)

() => {
    return 67;
}
// RETURNS 67
```

<!-- ## Closure -->

<!-- - Arrow functions **can continue to reference the scope of their creation**&mdash;even after they've passed out of that scope. This property is known as **closure**.
  - i.e. "Arrow functions inherit the `this` pointer from the scope in which they are created."
  - i.e. "It remembers the values of variables that were in scope when the function was created."
- Closure is clutch for working w/ JS in HTML. -->

<!-- > [!WARNING]
> So uh... I asked Claude, and it said that **all functions in JS can form closures**...so idk wtf these instructions are on. -->

<!-- - Arrow functions inherit the `this` pointer from the scope in which they are created. This is known as **closure**.
  - Closure allows an arrow function to **continue referencing its creation scope&mdash;even when it's passed out of that scope.** -->

## Closure

Remember: Arrow functions inherit the `this` pointer from where the scope in which they're created. i.e., **they close over the `this` from their surrounding scope** upon creation. This is what separates them behaviorly from regular functions or non-arrow anonymous functions.

## Example: Using arrow functions w/ React

Hoh boy...after looking at this example, I think I can see why React is so hard. Anyways here goes.

Let's start with a React application that increments & decrements a counter when buttons are pressed:

```jsx
function App() {
    const [count, setCount] = React.useState(0);

    function Increment() {
        setCount(count + 1);
    }

    function Decrement() {
        setCount(count - 1);
    }

    return (
        <div>
          <h1>Count: {count}</h1>
          <button onClick={Increment}>press to add</button>
          <button onClick={Decrement}>press to subtract</button>
        </div>
    );
}
```

This code actually has some thread-safety problems, but before we get there let's condense this code by moving the increment/decrement logic directly into the HTML w/ arrow functions:

```jsx
function App() {
    const [count, setCount] = React.usestate(0);

    return (
        <div>
          <h1>Count: {count}</h1>
          <button onClick={() => setCount(count + 1)}>press to add</button>
          <button onClick={() => setCount(count - 1)}>press to subtract</button>
        </div>
    );
}
```

But there's a problem! The `setCount` function provided by React's `useState` is asynchronous. So you don't know if other concurrently running code has changed the value of `count` between when you read it and when you set it. (This can lead to counter being incremented multiple times at once sometimes or not at all other times.)

To fix this, you must supply an arrow function to `setCount` function that sets the state, instead of simply supplying the desired value.

```jsx
// may corrupt value
setCount(count + 1);

// safe
setCount((prevCount) => prevCount + 1);
```

Now, instead of allowing your code to do the read operation, React controls when the state variable is updated. And **React guarantees that functional updates are applied sequentially against the latest state**. (NOTE: it's not the fact that it's an arrow function that's providing this thread safety, it's the fact that we're passing a *function* into `setCount`, instead of just a new value.)

Here's what the code looks like w/ this improvement:

```jsx
function App() {
    const [count, setCount] = React.useState(0);

    return (
        <div>
          <h1>Count: {count}</h1>
          <button onClick={() => setCount((prevCount) => prevCount + 1)}>press to add</button>
          <button onClick={() => setCount((prevCount) => prevCount - 1)}>press to subtract</button>
        </div>
    );
}
```

This is the part where the instructions get overkill. Now, our concise code has become clunky w/ the duplicated logic for our `onClick` handlers. So, let's solve this with a factory!

```jsx
function App() {
    const [count, setCount] = React.useState(0);

    function CounterOpFactory(op) {
        return () => setCount((prevCount) => op(prevCount));
    }

    const incOp = counterOpFactory((c) => c + 1);
    const decOp = counterOpFactory((c) => c - 1);

    return (
        <div>
          <h1>Count: {count}</h1>
          <button onClick={incOp}>press to add</button>
          <button onClick={decOp}>press to subtract</button>
        </div>
    );
}
```

Closure is being used here to reference the operation that is used by the arrow function that is returned from the factory.

The instruction concludes with, "This results in concise, simple, thread safe code in a functional programming style." But nothing about this looks simple to me. And we've gone full circle to having our increment/decrement functions defined outside our HTML. So IMO the only real benefit I see is that it's thread safe. OK ig.