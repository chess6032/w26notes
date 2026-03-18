# Node web service

W/ JS, we can write code that listens on a port, receives & processes HTTP requests, and sends HTTP responses. We can use this to create a (simple) web service that we then execute via Node.js

## Setup

> [!NOTE]
> This is the backend code we'll be building upon for the remainder of the project for serving up frontend to the browser. We will no longer be using the VS Code Live Server extension.

First, create your webservice project:

    $ mkdir webservice-test
    $ cd webservice-test
    $ npm init -y

Now, in the project create a file named `index.js` with the following code:

```js
// index.js

const http = require('http');
const server = http.createServer(function (req, res) {
  res.writeHead(200, { 'Content-Type': 'text/html' });
  res.write(`<h1>Hello Node.js! [${req.method}] ${req.url}</h1>`);
  res.end();
});

server.listen(8080, () => {
  console.log(`Web service listening on port 8080`);
});
```

This code uses Node.js's built-in `http` package to simply display the HTTP request's method and path:

- Creates server w/ `http.createServer`, 
  - Input: a callback func that takes a request obj (`req`) and response obj (`res`, which is a **buffered stream**). **The request obj holds all the data from the request**, and you **populate the response obj** with the data that will be sent back as a response.
    - In the example, the callback returns an HTML snippet w/ status code 200 & Content-Type header&mdash;no matter what request is made. (A real server would examine the HTTP path & request to determine what kind of response to return.)
    - This callback function is **invoked on every incoming request**.
- `server.listen` starts listening on port 8080 and blocks until program is terminated.
- `res.writeHead()` adds headers to the response stream, and `res.write()` adds to the response's body.
- `res.end()` **flushes any data still in the response stream**, sending it to the client. It also indicates to the client that there is no more content to receive, and its **connection may be closed**. 
  - **Must ALWAYS be called**, or else the client will hang waiting for more data.
- `req.method` is the HTTP method used by the request. `req.url` is the request's path.

> [!NOTE]
> Because the response object given to `http.createServer`'s callback (`res`) is a stream, anything you write to it (`res.writeHead()` or `res.write()`) cannot be modified or taken back. 

To execute this program, you'd run Node.js to execute `index.js`:

    $ node index.js
    Web service listening on port 8080

From here, you can **open `localhost:8080` in your browser** to view the result. You could also just **press `F5` in VS Code** while you have `index.js` open, which opens Node.js AND attaches a debugger (so you can use breakpoints and shih).
