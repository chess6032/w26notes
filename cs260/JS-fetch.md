# Fetch API in JS

Today, the [fetch API](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API) is the preferred way to make HTTP requests. In JS we do this w/ the built-in `fetch()` function.

## `fetch()`

- Runtime function.
- **Input: URL** (string), and **(optional) options** ([Request object](https://developer.mozilla.org/en-US/docs/Web/API/Request)).
  - Default HTTP method: GET.
- **Returns a Promise**, whose `.then()` function takes a callback that is called when the URL content is obtained.
  - If the response is application/json, you can use `.json()` to convert it to a JS object.

### Examples

Logging a quote API to console w/ a GET request:

```js
fetch('https://quote.cs260.click')
  .then((response) => response.json())
  .then((jsonResponse) => {
    console.log(jsonResponse);
  });
```

POST request:

```js
fetch('https://jsonplaceholder.typicode.com/posts', {
  method: 'POST',
  body: JSON.stringify({
    title: 'test title',
    body: 'test body',
    userId: 1,
  }),
  headers: {
    'Content-type': 'application/json; charset=UTF-8',
  },
})
  .then((response) => response.json())
  .then((jsonResponse) => {
    console.log(jsonResponse);
  });
```
