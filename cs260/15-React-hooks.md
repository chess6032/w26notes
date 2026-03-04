# React hooks

- React hooks allow function-style components to do everything class-style components can&mdash;and more!
  - B/c of hooks, function-style components are the preferred way of doing shih in React.
- Hooks **MUST be called at the top scope** of component functions. They can NOT be called inside a loop or conditional.
  - This restriction ensures that hooks are always called in the same order when a component is rendered.

## `useState` hook

```js
const [state, stateSetterFunc] = React.useState(initialVal);
```

- `state` is the value of a variable of the same name defined internally within React. 
- `stateSetterFunc` is for setting that variable.

(The initial value is only used on the first render&mdash;subsequent renders ignore it.)

### Example

The previous instruction demonstrated usage of the `useState` hook:

```jsx
function Clicker({ initialCount }) {
    const [count, updateCount] = React.useState(initialCount);
    return <div onClick={() => updateCount(count + 1)}>Click count: {count}</div>;
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<Clicker initialCount={3} />);
```

## `useEffect` hook

- The `useEffect` hook allows you to **represent lifecycle events**.
- `React.useEffect()` takes two parameters (that I know of):
  1. A **function** that is run when the useEffect hook is triggered. If this function returns a function, then that is called when the component "cleans up" (whatever ts means).
  2. (Optional) An **array of dependencies** that can trigger the useEffect hook. (See below.)

e.g., if you wanted to run a function every time a component completes rendering, you could do the following:

```jsx
function UseEffectHookDemo() {
    React.useEffect(() => {
        console.log('rendered');
    });

    return <div>useEffectExample</div>;
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<UseEffectHookDemo />);
```

### `useEffect` dependencies


- By default, the `useEffect` callback is **called every time the component is rendered**.
- You can control what triggers a `useEffect` hook by specifying its **dependencies**.
  - Dependencies are specified in an **array** passed in as the **second parameter of `React.useEffect()`**.
    - By passing in `[]`, the effect hook is **only triggered the first time** the component is rendered.
  - ^ If instead of an array you pass in a **number**&mdash;let's call it *`time`*&mdash;then `useEffect` will run every *`time`* milliseconds.
    - Kinda akin to JS's [setInterval method](11-JS-setTimeout-setInterval.md#setinterval).

```jsx
function UseEffectHookDemo() {
    const [count1, updateCount1] = React.useState(0);
    const [count2, updateCount2] = React.useState(0);

    React.useEffect(() => {
        console.log(`count1 effect triggered ${count1}`);
    }, [count1]); // <-- count1 dependency defined here

    return (
        <ol>
          <li onClick={() => updateCount1(count1 + 1)}>Item 1: {count1}</li>
          <li> onClick={() => updateCount2(count2 + 1)}Item 2: {count2}</li>
        </ol>
    );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<UseEffectHookDemo />);
```

### `useEffect` clean up

- If the function you pass into `React.useEffect()` **returns a function**, that function will be **called when the component "cleans up"**.

e.g., in the code below, the `Db` function creates a database connection that must be released when Db component is destroyed. This code will release destroy the Db component (and release the DB connection) after five clicks from the user.

```jsx
function Clicker() {
    const [count, update] = React.useState(5);

    return (
        <div onClick={() => update(count - 1)}>
          Click count: {count}
          {count > 0 ? <Db/> : <div>DB Connection Closed</div>}
        </div>
    );
}

function Db() {
    React.useEffect(() => {
        console.log('connected');

        return function cleanup() {
            console.log('disconnected');
        };
    }, []); // <-- useEffect callback triggered only the first time the component is rendered.

    return <div>DB Connection</div>;
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<Clicker />);
```

### `useNavigate` hook

> [!NOTE]
> `useNavigate` msut be imported from `'react-router-dom'`


#### Getting the hook

```jsx
const navigate = useNavigate(); // navigate becomes a func
```

#### Using the hook

<pre><code class="language-js">navigate(<i>pathToWebpage</i>)</code></pre>

