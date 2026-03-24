# Authorization services

For authorization, you need to uniquely associated a user w/ a particular credential.

When a user is authenticated, you store an **authentication token** on that user's device (often in a cookie). As long as that auth token is valid (in your server), the user is considered authenticated.

So **you'll have to expose an authentication service that, given a username/email and password, returns an auth token.**

(If you wanted to get fancy, you would also store in that auth token a level of authentication&mdash;e.g. allowing some users to read and others to edit.)

## Solutions

Many service providers and package developers have created solutions for your to use so that you don't have to do the authenticating yourself. ...But **for this class, you'll implement your own**.

Regardless, let's talk about features provided by solutions:

- Following standards & standard protocols.
  - e.g. [OAuth](https://en.wikipedia.org/wiki/OAuth), [SAML](https://en.wikipedia.org/wiki/Security_Assertion_Markup_Language), and [OIDC](https://en.wikipedia.org/wiki/OpenID).
- Single Sign On: Allowing a user to authenticate using credentials of another web application. 
  - e.g., using Google to login to GitHub.
- Federated login: Reusing auth tokens across multiple websites.
  - e.g., logging into Gmail allows you to use Google Docs & YouTube w/o logging in again.

But alas, those features are beyond the scope of this course.
