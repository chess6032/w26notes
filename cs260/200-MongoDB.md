# MongoDB

- MongoDB uses **JSON objects as its core data model**.
  - This makes it easy to use JSON from top-to-bottom in your application's tech stack.
  - This makes it pair well w/ JS.
- A MongoDB database is composed of **one or more collections, each containing a JSON doc**.
  - You can think of it like an **array of JSON objects, each w/ a unique ID**.
- MongoDB has **no strict schema requirements**, unlike relational databses.
  - Allows DB to organically morph as application evolves.

## MongoDB Atlas

For this class, we use MongoDB's [Atlas](https://www.mongodb.com/atlas/database) service to host our DB. It's free. (Yay!)

In addition to the [instructions on the CS260 GitHub](https://github.com/webprogramming260/webprogramming/blob/main/instruction/webServices/dataServices/dataServices.md#mongodb-atlas), we've also been provided a [tutorial video](https://youtu.be/f75muk9W-Jc) for signing up for Atlas.

- Set your DB's network access to be **available from anywhere**, as described in the tutorial.
- You can get your DB's **connection string** by going to *Database > DataServices* in Atlas and **clicking the Connect button** next to your DB.
  - DO NOT PUT THIS INFO IN ANY FILE THAT GETS COMMITED TO YOUR REPO/PUSHED TO GITHUB. We'll store it in a `dbConfig.json` file, which we add to `.gitignore`, and then load the credentials from it.

> [!CAUTION] Don't check your MongoDB credentials into your repo.

## Using MongoDB

### Accessing your DB from your code

First, you'll need the MongoDB library:

```
$ npm install mongodb
```

Then, create a file named **`dbConfig.json`** to insert your MongoDB credentials (connection string). **MAKE SURE YOU ADD THIS FILE TO YOUR `.gitignore`!!!**

```json
// dbConfig.json
// ADD THIS TO .gitignore !!!!!
{
    "hostname": "skibidi.rizz",
    "username": "rizzler",
    "password": "skibidi"
}
```

> [!IMPORTANT] 
>
> Include `dbConfig.json` in `.gitignore`!  
>
> DO NOT PUSH `dbConfig.json` TO YOUR GITHUB!!  

To access your DB from your code, you'll load your credentials from this JSON file. The instructional material gave this as an example for doing that:

```jsx
const { MongoClient } = require('mongodb');
const config = require('./dbConfig.json'); // access credentials in separate file

const url = `mongodb+srv://${config.username}:${config.password}@${config.hostname}`;

const client = new MongoClient(url);
const db = client.db('rental'); // not sure what to input here ngl
const collection = db.collection('house'); // not sure what to input here ngl

// < put code here to test connection to db (see below) >

try {
    // < do stuff w/ DB here >
} catch (e) {
    console.log(`DB (${url}) error: ${e.message}`);
} finally {
    await client.close();
}
```

> [!IMPORTANT]
> Remember to **close your connection to the DB** w/ <code><u>client</u>.close()</code>

### Testing connection on startup

It's a good idea to **ping the DB before trying to access/modify any data** to make sure your connection's all well and good.

```js
try {
    await db.command({ ping: 1 });
    console.log(`DB connected to ${config.hostname}`); // `config` is the dbConfig.json file we loaded in earlier.
} catch (e) {
    console.log(`Error with ${url} because ${e.message}`); // `url` is the url to our DB we created earlier.
    process.exit(1);
}
```

Then, if your server isn't starting, you can check your logs to see if an exception was thrown.

### Inserting: `.insertOne()`

<pre><code>await <u>collection</u>.<span style="color: lightyellow;">insertOne</span>(<em>jsObject</em>);</code></pre>

- When the doc is inserted into the collection it is **automatically assigned a unique ID.**
- If the DB (`client.db()`) or collection (`db.collection()`) you insert to don't exist, MongoDB will create them for you.

### Query: `.find()`

<pre><code>const cursor = <u>collection</u>.<span style="color:lightyellow;">find</span>(<em>query</em>, <em>options</em>);</code></pre>

- *`query`* is a **JSON document** that documents in your DB are compared to when querying.
  - If empty (`{}`), or if you don't given any parameters at all, then `.find()` will **return all documents in collection**.
  - Matching exact values: `field: value`.
  - Using query operations: `field: { $op: x }`.
    - e.g., `bed: { $lt: 10 } ` will match w/ all documents who have a `bed` field w/ a value less than 10.
- *`options`* is an object that you populate with options for your query.
  - See the [MongoDB docs](https://www.mongodb.com/docs/manual/reference/method/db.collection.find/#options) for a list of options you can use.
- Each document returned by `.find()` includes its unique ID in a field called `_id`.

Example:

```js
const query = { property_type: 'Condo', beds: { $lt: 2} };
const options = {
    sort: { name: -1 },
    limit: 10,
};

const cursor = collection.find(query, options);
const rentals = await cursor.toArray();
rentals.forEach((i) => console.log(i));
```

### Update

<pre><code>await <u>collection</u>.<span style="color:lightyellow;">updateMany</span>(<em>query</em>, { $set: { <em>field: value, field: value,</em> ... } });</code></pre>

You can also use `.updateOne()` to update the FIRST matching document.

### Delete

<pre><code>await <u>collection</u>.<span style="color:lightyellow;">deleteMany</span>(<em>query</em>);</code></pre> 

^ This will delete ALL documents that match *`query`*.

You can delete a single document by passing in **document's ID** into `.deleteOne()`. e.g.:

```js
const insertResult = await collection.insertOne(house);

const deleteQuery = { _id: insertResult.insertedId };
await collection.deleteOne(deleteQuery);
```
