# JSON

JSON stands for "JavaScript Object Notation". It was conceived by Douglas Crockford in 2001 while working at Yahoo! It received official standardization in 2013 (ECMA-404) and 2017 ([RFC 8259](https://datatracker.ietf.org/doc/html/rfc8259)).

- Provides simple yet effective way to share and store data.
- By design, JSON easily convertible to & from JS objects.
- JSON is always encoded w/ [UTF-8](https://en.wikipedia.org/wiki/UTF-8).

## Data types

| Type                  | Example |  
| --------------------- | ------- |  
| string                | "crockford" |  
| number (int or float) | 42.1 |  
| bool                  | true |  
| array                 | [null, 42.1, "crockford"] |  
| object                | {"a":1, "b":"crockford"} |  
| null                  | null |  

Most commonly, a JSON document contains a root object that holds all these.

## Converting to/from JS

- JSON &rightarrow; JS: `JSON.parse()`
- JS &rightarrow; JSON: `JSON.stringify()`

When converting from JS to JSON, `undefined` objects are dropped.
