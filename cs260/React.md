# React notes

React is focused on making **reactive web page components** (hence the name) that automatically update based on user interactions or changes in underlying data.

## JSX

React abstracts HTML into a JS variant called JSX. JSX is then converted to valid HTML & JS using a preprocessor (e.g. Vite or Babel).

<!-- ## Ummm

Soooo lowkirkenuinely there seems to be a LOT on this...but I think I have to be done w/ React pt. 1 deliverable by tn at midnight...so I'm going to run and do that and if I lack knowledge I'll come back here. -->

## Usage w/ NPM & Vite

Initialization:

```bash
$ npm init -y
$ npm install vite@latest -D
$ npm install react react-dom
```

^ Updates `package.json`. You can see source code for modules in `node_modules/`.

(Make React code.)

Running:

```bash
$ npx vite
```

`npx` is a variant of `npm` which directlyl executes a Node package w/o referencing the `package.json` file. (The instructions allege this is really useful for running JS code as a CLI program, such as Vite.)

### Hello World

index.html:

```html
<!DOCTYPE html>
<html lang="en">
	<head>
		<title>React Demo</title>
	</head>	
	<body>
		<noscript>You need to enable JavaScript to run this app.</noscript>
		<div id="root"></div>
		<script type="module" src="/index.jsx"></script>
	</body>
</html>
```

index.jsx:

```jsx
import React from 'react'; // for building components
import ReactDOM from 'react-dom/client'; // for rendering to webpage

// **** in React, components are just function that return UI. *****
function App() {
          // var    setter
    const [bgColor, setBgColor] = React.useState('white');
    // React stores bgColor somewhere for persistency.
    // ^ setBgColor() modifies this.
    // setBgColor() also flags App as needing a re-render (bc the .useState() that created it is in App), 
    //  and tells React to begin a re-render.

    // create a function that calls bgColor's setter, which we'll put in the div.
    // (we could just put this logic in the div directly, but abstracting it here makes the JSX more readable.)
    const handleClick = () => {
        setBgColor(bgColor === 'white' ? 'yellow' : 'white');
    };

    // JSX: this is the <div> that gets loaded into the webpage.
    return (
        <div onClick={handleClick} style={{
                backgroundColor: bgColor,
                height: '100vh', font: 'bold 20vh Arial', 
                display: 'flex', alignItems: 'center', justifyContent: 'center'
            }}>
            <div>Hello React</div>
        </div>
    );
}

// this only runs ones once when the page loads
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
```

# Component notes

A component is a **function that returns UI**.

## JSX

**The JSX returned by a component is used to generate UI.** 

## Styling

You can put in-line CSS in your HTML parts in JSX if you're insane, or you can also import a CSS file like normal. 

- To use an external CSS file, you would include `import 'path/to/css.css';` statement at the top of your code.
  - The styling now applies like normal, with the only difference being that in your JSX code you must **use `className` instead of `class`** for elements to be targeted by class selectors.
    - (This is because "class" is already a keyword in JavaScript.)
  - (If the css file is in the same directory as your jsx file, start your path with `./`)

## Child components

The JSX returned by a component **may reference other components**. This way, you can achieve the same tree-like structure as you can in HTML.

## Properties

You can pass information to React components in the form of **element properties**. The component receives these properties **in its constructor**.

### Example

JSX:

```jsx
<div> Component: <Demo who="Walke" /><div>
```

React component:

```jsx
function Demo(props) {
    return <b>Hello {props.who}</b>;
}
```

## State

A component can have an **internal state**. 

Component state is **created by calling `React.useState()`** (which the instructions refer to as a "hook" function). `useState()` returns a tuple, where: 

- FIRST ITEM: variable that contains the **current state**.
- SECOND ITEM: function that **updates the state**.

## Reactivity

The React framework uses each component's properties and states to determine the reactivity of its interface:

- Whenever a component's **state or properties change**, the `render()` function for that component&mdash;and all of its dependent component `render()` functions&mdash;are called.

# Routing

Routing is the process of using JS to modify a webpage at runtime to give the appearance of loading different pages.

## `react-router-dom`

React does not have a standard router package, so developers choose one of many. We will use [`react-router-dom`](https://www.npmjs.com/package/react-router-dom).

- `npm install react-router-dom`

> `react-router-dom` and `react-router` are NOT the same thing. We use the former, not the latter.

### Import

In your JSX files, include this import statement:

```jsx
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
```

And also probably import `Link` if you're using that.

### Implementation

- `BrowserRouter` component: encapsulates the entire application & controls the routing action.
- `Link` or `NavLink` component: captures user navigation events and modifies what is rendered by `Routes` component by matching up the `to` and `path` attributes.

### Example

```jsx
function Page({ color }) {
  return (
    <div className="page" style={{ backgroundColor: color }}>
      <h1>{color}</h1>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <nav>
          <NavLink to="/">Red</NavLink>
          <NavLink to="/green">Green</NavLink>
          <NavLink to="/blue">Blue</NavLink>
        </nav>

        <main>
          <Routes>
            <Route path="/" element={<Page color="red" />} exact />
            <Route path="/green" element={<Page color="green" />} />
            <Route path="/blue" element={<Page color="blue" />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
```