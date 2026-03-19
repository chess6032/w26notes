# Implementing account creation/login (BACKEND)

These notes go over how to implement authentication & authorization by implementing register/login/logout services in your backend.

## Endpoint design

| Endpoint         | Purpose (high-level)               | Action (low-level) |  
| ---------------- | ---------------------------------- | ------------------ |  
| **Registration** | Create account.                    | Create authtoken and create user in DB.  |  
| **Login**        | Log in to account.                 | Create authtoken. |  
| **Logout**       | Log out of account.                | Delete authtoken. |  
| **Get Me**       | Return info ab authenticated user. | |  

Here's the implementation details:

| Endpoint         | HTTP Method   | Path        | Input                | Return (success)    | Return (failure)     |  
| ---------------- | ------------- | ----------- | -------------------- | ------------------- | -------------------- |  
| **Registration** | `POST`        | `/api/auth` | Email & password     | Auth token (cookie) | `409 (conflict)`     |  
| **Login**        | `PUT`         | `/api/auth` | Email & password     | Auth token (cookie) | `401 (unauthorized)` |  
| **Logout**       | `DELETE`      | `/api/auth` | Auth token (cookie)  | (nothing)           | N/A                  |  
| **Get Me**       | `GET`         | `/api/user` | Auth token (cookie)  | Info ab user        | `401 (unauthorized)` |  

> [!NOTE]
> On success, all endpoints will return `200 (ok)` in their response.

> [!IMPORTANT]
> Every request body will be formatted as JSON.

### Registration endpoint

- TAKES: **email & password.**
- RETURNS: 
  - SUCCESS: **Cookie containing auth token.**
  - FAILURE: **409 (conflict)**.
    - May happen if email already exists. (i.e., email is taken. i.e., email is already associated w/ a user), .

```http
POST /api/auth HTTP/2
Content-Type: application/json
{
    "email":"marta@id.com",
    "password":"toomanysecrets"
}
```

```http
HTTP/2 200 OK
Content-Type: application/json
Set-Cookie: auth=tokenHere
{
    "email":"marta@id.com"
}
```

### Login endpoint

- TAKES: **email & password.**
- RETURNS: 
  - SUCCESS: **Cookie containing auth token.**
  - FAILURE: **401 (unauthorized)**.
    - May happen if email does not exist (i.e., no user is ass. w/ email) or inputted password was incorrect.


```http
PUT /api/auth HTTP/2
Content-Type: application/json
{
  "email":"marta@id.com",
  "password":"toomanysecrets"
}
```

```http
HTTP/2 200 OK
Content-Type: application/json
Set-Cookie: auth=tokenHere
{
  "email":"marta@id.com"
}
```

### Logout endpoint

- TAKES: **Cookie containing auth token.**
- ACTION: **Marks auth token as invalid** for future use.
- RETURN: **200 (ok)**.
  - (ALWAYS returns 200).

```http
DELETE /api/auth HTTP/2
Cookie: auth=tokenHere
```

```http
HTTP/2 200 OK
Content-Type: application/json
{
}
```

### GetMe endpoint

- TAKES: Cookie containing auth token.
- RETURNS:
  - SUCCESS: **info ab user**.
  - FAILURE: **401 (unauthorized)**.
    - May happen if auth token is invalid, or if user does not exist.

```http
GET /api/user HTTP/2
Cookie: auth=tokenHere
```

```http
HTTP/2 200 OK
Content-Type: application/json
{
  "email":"marta@id.com"
}
```

## Web service (setup)

Now let's build our web service to support these services!

Inside your project directory:

1. Create a directory `login/` to your project root.
2. Create a directory `login/service/`.
3. Initialize NPM in `login/service/` and install `express`, `cookie-parser`, `uuid`, and `bcryptjs`.
4. Create a file called `service.js` (I assume in `login/service`??), and fill it with the code below.
5. Start up your webservice (`node --watch service.js`&mdash;or start the VS Code Debugger w/ `F5`)
6. (In a separate terminal) Use the `curl` command (see below) to try out an endpoint.

### Terminal commands & code

Steps 1-3:

```
$ mkdir login && cd login
$ mkdir service && cd service
$ npm init -y
$ npm install express cookie-parser uuid bcryptjs
```

Step 4:

```js
// login/service/service.js
// these are stubs for the service endponts that we'll fill in later.

const express = require('express');
const app = express();

// registration
app.post('/api/auth', async (req, res) => {
  res.send({ email: 'marta@id.com' });
});

// login
app.put('/api/auth', async (req, res) => {
  res.send({ email: 'marta@id.com' });
});

// logout
app.delete('/api/auth', async (req, res) => {
  res.send({});
});

// getMe
app.get('/api/user', async (req, res) => {
  res.send({ email: 'marta@id.com' });
});

const port = 3000;
app.listen(port, function () {
  console.log(`Listening on port ${port}`);
});
```

Step 5:

```
$ node --watch service.js
```

Step 6:

```
$ curl -X POST localhost:3000/api/auth -d '{"email":"test@id.com", "password":"a"}'

{"email":"marta@id.com"}
```

## Handling requests

### Parsing JSON w/ middleware

Since our request bodies are designed to contain JSON, we'll **use a middleware to parse the body of every request**. In particular, we'll use the middleware built into express.

```js
app.use(express.json());
```

To demonstrate this, we'll just rewrite the registration endpoint to display the request body:

```js
app.use(express.json());

app.post('/api/auth', (req, res) => {
    res.send(req.body);
});
```

You can then test this w/ `curl`:

```
$ curl -X POST localhost:3000/api/auth -H "Content-Type: application/json" -d '{"email":"test@id.com", "password":"a"}'

{"email":"test@id.com","password":"a"}
```

### Storing users & hashing passwords

- We use the `bcryptjs` package to hash passwords, which uses the [bcrypt](https://en.wikipedia.org/wiki/Bcrypt) algorithm.
  - `bcrypt.hash()` to create a hash for a password. (For registration.)
  - `bcrypt.compare()` to see if a password, when hashed, matches another hash. (For login.)

Here's the code for creating a user w/ a hashed password:

```js
const bcrypt = require('bcryptjs');

const users = []; 
// ^ this is a dummy DB. It would actually clear every time you stop your server.
// In the future, this will be replaced w/ DB shih.

async function createUser(email, password) {
  const passwordHash = await bcrypt.hash(password, 10);

  const user = {
    email: email,
    password: passwordHash,
  };

  users.push(user);

  return user;
}

function getUser(field, value) {
  if (value) {
    return users.find((user) => user[field] === value);
  }
  return null;
}
```

> [!NOTE]
> The password itself is NEVER stored, only the hash.

### Generating auth tokens

- We use the `uuid` package to generate random authtokens, which uses the [UUID](https://en.wikipedia.org/wiki/Universally_unique_identifier) algorithm.

```js
const uuid = require('uuid');
```

### Generating & deleting cookies (to store auth tokens)

- We use the `cookie-arser` package for generating HTTP cookies, which does all the work of sending a cookie to the browser & parsing it when the browser makes subsequent requests.
- To make the site as secure as possible, we use the cookie `httpOnly`, `secure`, and `sameSite` options:
  - `httpOnly`: Tells browser to not allow JS running on the browser to read the cookie.
  - `secure`: requires HTTPS to be used when sending cookie back to the server.
  - `sameSite`: will only return the cookie to the domain that generated it.

```js
const cookieParser = require('cookie-parser');
app.use(cookieParser());

// Create a token for the user and send a cookie containing the token
function setAuthCookie(res, user) {
  user.token = uuid.v4();

  res.cookie('token', user.token, {
    secure: true,
    httpOnly: true,
    sameSite: 'strict',
  });
}

// Delete the user's auth token and the cookie containing it (or smth like that idk this func didn't have a comment on it)
function clearAuthCookie(res, user) {
  delete user.token;
  res.clearCookie('token');
}
```

## Endpoint implementations

### Registration code

After adding in all the code from the [Handling requests](#handling-requests) section, you can replace the registration endpoint stub w/ this:

```js
// registration
app.post('/api/auth', async (req, res) => {
  if (await getUser('email', req.body.email)) {
    // FAIL: user email already exists
    res.status(409).send({ msg: 'Existing user' });
  } else {
    // SUCCESS:
    const user = await createUser(req.body.email, req.body.password);

    setAuthCookie(res, user);

    res.send({ email: user.email });
  }
});
```

### Login code

Use `getUser()` to see if a user w/ the given email exists, and then `bcryptjs.compare()` to see if the user's password was correct.

```js
// login
app.put('/api/auth', async (req, res) => {
  const user = await getUser('email', req.body.email); // get user by email
  if (user && (await bcrypt.compare(req.body.password, user.password))) { // check that user w/ email exists & that password was correct
    // SUCCESS:
    setAuthCookie(res, user);

    res.send({ email: user.email });
  } else {
    // FAILURE: email doesn't exist, or password incorrect.
    res.status(401).send({ msg: 'Unauthorized' });
  }
});
```

### Logout code

For logout, we check if there exists a user authenticated w/ an auth token. If there is, we delete said auth token. If not, then we just ignore the request.

```js
// logout
app.delete('/api/auth', async (req, res) => {
  const token = req.cookies['token'];
  const user = await getUser('token', token);
  if (user) { // check if there exists a user w/ token
    clearAuthCookie(res, user);
  }

  // we don't care if token doesn't exist tho lol

  res.send({});
});
```

### GetMe code

The GetMe endpoint just looks into the database by querying w/ the request's auth token.

```js
// getMe
app.get('/api/user/me', async (req, res) => {
  const token = req.cookies['token'];
  const user = await getUser('token', token);
  if (user) { // check if there exists a user authenticated w/ token
    // SUCCESS:
    res.send({ email: user.email });
  } else {
    // FAILURE: no user w/ token exists
    res.status(401).send({ msg: 'Unauthorized' });
  }
});
```

### FINAL (BACKEND) CODE

Here is the full example code:

```js
const express = require('express');
const app = express();
const cookieParser = require('cookie-parser');
const uuid = require('uuid');
const bcrypt = require('bcryptjs');

app.use(express.json());
app.use(cookieParser());

app.post('/api/auth', async (req, res) => {
  if (await getUser('email', req.body.email)) {
    res.status(409).send({ msg: 'Existing user' });
  } else {
    const user = await createUser(req.body.email, req.body.password);
    setAuthCookie(res, user);

    res.send({ email: user.email });
  }
});

app.put('/api/auth', async (req, res) => {
  const user = await getUser('email', req.body.email);
  if (user && (await bcrypt.compare(req.body.password, user.password))) {
    setAuthCookie(res, user);

    res.send({ email: user.email });
  } else {
    res.status(401).send({ msg: 'Unauthorized' });
  }
});

app.delete('/api/auth', async (req, res) => {
  const token = req.cookies['token'];
  const user = await getUser('token', token);
  if (user) {
    clearAuthCookie(res, user);
  }

  res.send({});
});

app.get('/api/user/me', async (req, res) => {
  const token = req.cookies['token'];
  const user = await getUser('token', token);
  if (user) {
    res.send({ email: user.email });
  } else {
    res.status(401).send({ msg: 'Unauthorized' });
  }
});

const users = [];

async function createUser(email, password) {
  const passwordHash = await bcrypt.hash(password, 10);

  const user = {
    email: email,
    password: passwordHash,
  };

  users.push(user);

  return user;
}

async function getUser(field, value) {
  if (value) {
    return users.find((user) => user[field] === value);
  }
  return null;
}

function setAuthCookie(res, user) {
  user.token = uuid.v4();

  res.cookie('token', user.token, {
    secure: true,
    httpOnly: true,
    sameSite: 'strict',
  });
}

function clearAuthCookie(res, user) {
  delete user.token;
  res.clearCookie('token');
}

const port = 3000;
app.listen(port, function () {
  console.log(`Listening on port ${port}`);
});
```

## Testing

Here are some `curl` commands to test your endpoints. Note that `-c` and `-b` params tell curl to store and use cookies w/ a given file.

```
$ curl -X POST localhost:3000/api/auth -H 'Content-Type:application/json' -d '{"email":"지안@id.com", "password":"toomanysecrets"}'

{"email":"지안@id.com"}
```

```
$ curl -c cookie.txt -X PUT localhost:3000/api/auth -H 'Content-Type:application/json' -d '{"email":"지안@id.com", "password":"toomanysecrets"}'

{"email":"지안@id.com"}
```

```
$ curl -b cookie.txt localhost:3000/api/user/me

{"email":"지안@id.com"}
```
