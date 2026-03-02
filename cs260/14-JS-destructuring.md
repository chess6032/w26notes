# JS destructuring

> [!IMPORTANT]
> This is talking about JS's "destructuring", NOT object destructors.

- In JS, destructuring is the process of **pulling individual items out of an existing collection**, or **removing structure**.
- Destructuring can be done with **arrays or objects**.
- Destructuring is helpful when you only care about a few items in a structure.
- Destructuring is **used extensively within React**, so you'll need to master this concept in order to build your startup.

## Destructuring arrays

- Uses `[]`.

```js
const a = [1, 2, 4, 5];

// destructure the first two items from a into new variables b, c
const [b, c] = a;

console.log(b);
// OUTPUT: 1
console.log(c);
// OUTPUT: 2
```

Although it LOOKS like you're *declaring* an array, you're actually just *grabbing values* from an array.

### `...` to grab the rest

```js
const a = [1, 2, 4, 5];
const [b, c, ...others] = a;

console.log(b, c, others);
// OUTPUT: 1, 2, [4, 5]
```

## Destructuring objects

- Uses `{}`.
- Instead of getting associated value by positions, you **specify the properties you want to pull** from the source object.

```js
const o = { a: 1, b: 'animals', c: ['fish', 'cats'] };

const { a, c } = o;

console.log(a);
// OUTPUT: 1
console.log(c);
// OUTPUT: ['fish', 'cats']
```

## Mapping to new variable names

**You can also map the property values to *new* variable names** instead of just using the original property names **using `:`**.

```js
const o = { a: 1, b: 'animals', c: ['fish', 'cats'] };

const { a: count, b: type } = o;

console.log(count);
// OUTPUT: 1
console.log(type);
// OUTPUT: animals
```

## Providing default values

You can provide default values that are used when properties are missing.

```js
const { a, b=22 } = {}; // trying to destructure an empty object
const [c=44] = []; // trying to destructure an empty array

console.log(a);
// OUTPUT: undefined
console.log(b);
// OUTPUT: 22
console.log(c);
// OUTPUT: 
```

## Destructuring in React

React makes extensive use of destructuring when you **pass paramters to components and create state**.

### Example

In the example below:

- The `Clicker` **function's parameters are destructured**.
- The **returned array of `React.useState` is destructured** to just the variable and said variable's update function

```jsx
function Clicker({ initialCount }) { // destructuring!
    const [count, updateCount] = React.useState(initialCount); // destructuring!
    return <div onClick={() => updateCount(count + 1)}>Click count: {count}</div>;
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<Clicker initialCount={3} />);
```
