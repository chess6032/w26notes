# Local Storage

- Each browser has an API accessible in JS via `localStorage` that provides the ability to **presistently store and retrieve data** across user sessions & HTML page renderings
  - e.g., your frontend JS could store a user's name on one HTML page, then retrieve that name later when a diff HTML page is loaded. 
    - That user's name would also be available in local storage the next time hte same browser is used to access the same website.
- In addition to persisting data, `localStorage` is **also used as a cache for when dat can't be obtained from the server**.
  - e.g., your frontend JS could store the last high scores obtained from the service, then display thsoe scores in the future if said service is not available.

## Using `localStorage`

`localStorage` has four main methods:

| Function                                        | Description                                       |  
| ----------------------------------------------- | ------------------------------------------------- |  
| <code>setItem(<i>name</i>, <i>value</i>)</code> | **Sets a named item's value** INTO local storage. |  
| <code>getItem(<i>name</i>)</code>               | **Gets a named item's value** FROM local storage. |  
| <code>removeItem(<i>name</i>)</code>            | **Removes a named item** from local storage.      |  
| `clear()`                                       | **Clears all items** in local storage.            |  

- Local storage **values are always saved as strings**.
  - All *`value`* parameters MUST either be **strings, numbers, or bools**. For any other type, you must first convert it to JSON (`JSON.stringify()`).
  - To retrieve any non-string value in its original type, you must parse it as JSON (`JSON.parse()`).

## Seeing local storage in browser DevTools

(I can't tell if this is specifically for Chrome or just browsers in general.)

- Console output will show when local storage values are saved & retrieved.
- You can see what values are currently set at devtools > `Application` > `Storage` > `Local Storage` > *`(domain name)`*
  - W/ your browser's devtools you can also add, view, update, and delete any local storage values.

## Example

Code:

```js
let user = 'Alice';

let myObject = {
  name: 'Bob',
  info: {
    favoriteClass: 'CS 260',
    likesCS: true,
  },
};

let myArray = [1, 'One', true];

localStorage.setItem('user', user);
localStorage.setItem('object', JSON.stringify(myObject));
localStorage.setItem('array', JSON.stringify(myArray));

console.log(localStorage.getItem('user'));
console.log(JSON.parse(localStorage.getItem('object')));
console.log(JSON.parse(localStorage.getItem('array')));
```

Output:

    Alice
    {name: 'Bob', info: {favoriteClass: 'CS 260', likesCS: true}
    [1, 'One', true]
