# Objects in JS

## `Object`

- In JS, it looks like **objects are glorified name-value pairs** ((string) names of members and their (any type) values).
- Objects can be created w/ the `new` operator.
- Referencing properties: **`obj.name` or `obj["name"]` are both valid.**
- **You can add properties to an object *after declaration*** by simply referencing the property name in an assignment.
  - Dude objects look like glorified maps/dictionaries wtf.
- In JS, (almost) everything inherits from the `Object` object (like Java).
  - `null` doesn't inherit from `Object`.

```js
const obj = new Object({ a: 3 });
obj['b'] = 'fish';
obj.c = [1, 2, 3];
obj.hello = function () {
  console.log('hello');
};

console.log(obj);
// OUTPUT: {a: 3, b: 'fish', c: [1,2,3], hello: func}
```

## Declaring objects w/ object-literals

You can declare an object w/ properties via a special **JSON-like syntax** called **"object-literal"** syntax.

```js
const obj = {
  a: 3,
  b: 'fish',
  c: [1, true, 'dog'],
  d: { e: false },
  f: function () {
    return 'hello';
  },
};
```

## `Object` functions

`Object` has several functions assoiciated w/ it:

| Function    | Return                        |  
| ----------- | ----------------------------- |  
| `entries()` | Array of **key-value pairs**. (Each entry  in the array is a two-sized array `[key, value]`.) |  
| `keys()`    | Array of **keys**.            |  
| `values()`  | Array of **values**.          |  

These function are demonstrated here:

```js
const obj = {
  a: 3,
  b: 'fish',
};

console.log(Object.entries(obj));
// OUTPUT: [['a', 3], ['b', 'fish']]

console.log(Object.keys(obj));
// OUTPUT: ['a', 'b']

console.log(Object.values(obj));
// OUTPUT: [3, 'fish']
```

## Constructors

- **Any function that returns an object is considered a "constructor"**.
- Constructors can be **invoked w/ the `new` operator**.

```js
function Person(name) {
    return {
        name: name,
        log: function () {
            console.log('My name is ' + this.name);
        }
    };
}
```

## `this`

The meaning of `this` depends upon the scope where it is used, but in the context of an Object it **refers to a pointer to the object**.

(We'll talk more ab this in a later instruction.)

# Classes in jS

- keyword: `class`.
- Class constructors are named `constructor`. 
  - (As opposed to the classes name like in C++ or Java.)

## Private members

- In a class's constructor, **you can make members private by prefixing them w/ `#`**.

## Inheritance

- keyword: `extends`.
- Inside a subclass's constructor, use `super()` to invoke the parent class's constructor.
