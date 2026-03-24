# SOP and CORS

SOP (Same Origin Policy) & CORS (Cross-Origin Resource Sharing) protect the user from **browser-side attacks**. They **control which responses browser-side scripts are allowed to read**.

- SOP: Restricts scripts on one origin from interacting w/ resources from a different origin.
  - "Origin" = scheme + domain + port.
  - Response origin is defined in the response's **`Origin` header**.
  - i.e., **only allows JS scripts to read responses whose origin is the same as the origin the browser user is currently visiting**.
- CORS: Allows exceptions to SOP. 
  - A CORS-compliant response includes a **`Access-Control-Allow-Origin` (`ACAO`)** header that **lists request origins that may break SOP**.
    - A response w/ `*` (wildcard) for its `ACAO` will allow ANY origin to read it.
  - If **no `ACAO` header** is given in a response, a SOP/CORS-enforcing browser will **enforce strict SOP** for that response.
    - (ALL modern browsers are SOP/CORS-enforcing.)

> [!NOTE]
> **Different subdomains** = different origins, and **different ports** = different origins, even if said origins share the same domain.

The security protection offered by SOP/CORS is browser-side. You write your server to support them, but it's the browser that offers the protection. Non-browser clients (e.g., `curl`, or a Python script) ignore them entirely.

SOP/CORS deals with reading responses only. It does not provide any server-side security for guarding against malicious requests or anything.

> [!IMPORTANT]
> **ALL modern browsers enforce SOP/CORS**: 
> 
> - If you give a response that lacks an `ACAO` header, modern browsers will always enforce strict SOP.
> - If your server makes a request to third party service, you should verify that said service's response's `ACAO` header does not block your server from reading it. 
> 
> Uhhhh Claude says there's a few exceptions to this: form submissions (`<form action="url">`), HTML tags that load cross-origin resources (`<img src="url">`, `<link href="url">`, `<script src="url">`, etc.), and (generally) redirects as well.


## Using third party services

If you want to use a third-party service, test it first to ensure that your server's origin isn't blocked by the service's `ACAO` header.