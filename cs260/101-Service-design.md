# Webservice design

## Endpoints

Service endpoints are often referred to w/ "API"&mdash;"Application Programming Interface." But this term is a throwback to old desktop applications. Furthermore, sometimes "API" is used to refer to an entire collection endpoint, and sometimes it's used to refer to a single endpoint.

### Principles

Keep these in mind.

- **Grammatical**: Idk wtf ts is talking ab.
- **Readable**: The resource an HTTP request references should be clearly readable in the URL path. Human readable URLs make your endpoints easier to remember.
- **Discoverable**: Consider having an endpoint that returns a list of service endpoints a client can call.
- **Compatible**: When you add new functionality, try to do it w/o breaking existing clients. (If you control all the client code, you don't have to worry ab ts.)
- **Simple**: Keep your endpoints focused on the primary resources. Endpoints should not focus on the data structure or devices used to host resources. There should only be one way to act on a resource. Endpoints should only do one thing.
- **Documented**: Write documentation for your endpoints, nerd. If you create an initial draft even before you start coding, it can help you mentally clarify your design as you work on it.

## RPC

A Remote Procedure Call (RPC) **exposes service endpoints as function calls**.

- When RPC is used over HTTP it usually just leverages the POST HTTP verb.
  - HTTP method & body are represented by function name.
- Advantage: maps directly to function calls that might exist w/in the server.
- Disadvantage: directly exposes inner workings of the services. (thus creating coupling btwn endpoints & implementation.)

### Examples

Example 1: Function name in request URL.

    POST /updateOrder HTTP/2

    {"id": 2197, "date": "20220505"}

Example 2: Function name in request body.

    POST /rpc HTTP/2

    {"cmd":"updateOrder", "params":{"id": 2197, "date": "20220505"}}

## REST

Representational State Transfer (REST) attempts to take adv of foundational principles of HTTP. (It was made by Roy FIelding, a contributor to the HTTP specification.)

- REST HTTP methods **always act upon a resource**.
- Operations on a resource impact the state of the reousrce as it is transferred by a REST endpoint call.
  - Allows for optimal caching of HTTP.
  - e.g., GET will always return the same resource until a PUT is executed on the resource.

There are several other pieces of [Fielding's dissertation](https://www.ics.uci.edu/~fielding/pubs/dissertation/top.htm) on REST that are required for a truly "restful" implementation...but they are often ignored.

### Example

w/ REST, the updateOrder endpoint (from above) would look this:

    PUT /order/2197 HTTP/2

    {"date": "20220505"}

## GraphQL

GraphQL focuses on the **manipulation of data**, as opposed to a func call (like RPC) or resource (like REST). GraphQL was dev'd to address the massive number of REST or RPC calls a web app client needed to make for even simple stuff.

- Heart of GraphQL: a query that specifies desired data, and how it should be joined and filtered.
  - e.g., instead of making a call for getting a store, and then a bunch of calls for getting the store's orders and employees, GraphQL would send a single query that would request all of that info in one big JSON response. The server would examine that query, join the desired data, then filter out anything unwanted.
- Upside: More flexibility. One single endpoint call instead of sixty-seven morbillion.
- Downsides: 
  - Client now has significant power to consume server resources.
  - Difficult for server to implement authorization rights to data, as they have to be baked into the data schema.
    - (BUT there are standards for how to define a complex schema, provided by common GraphQL packages.)

### Example

```GraphQL
query {
    getOrder(id: "2197") {
        orders(filter: { date: { allofterms: "20220505" } }) {
            store
            description
            orderedBy
        }
    }
}
```
