# Vite notes

Vite is a **build tool** and **development server** for web apps. In our toolchain, we use Vite to **translate from React to browser**.

## Why we use Vite

There's a lot of steps that must be taken before a React project can be deployed to the web: React uses JSX code (which most browsers don't natively understand) and may have TypeScript and/or multiple files that need to be combined. 

Our toolchain has several tools to turn React into a deployable web app (Babel, Rollup, PostCSS, etc.); Vite serves as a **Command-Line Interface (CLI)** that **wraps around those tools**, abstracting them away.

## Features

Vite...

- Bundles yoru code quickly.
- Has great debugging support.
- Allows you to easily support JSX, TypeScript, and diff CSS flavors.

## JS vs JSX

JSX is "JavaScript Syntax Extension", or "JavaScript XML". Google says it's a syntax extension for JS primarily used w/ React to describe UI components.

Technically, you can name your JSX files w/ `.js` extension and it'll run just fine for the Babel transpiler we use. But some editor tools work differently based on file extensions, so **you should always use `.jsx` extension for JSX files**.

(Uhhhhh apparently AirBnB devs have an [interesting discussion](https://github.com/airbnb/javascript/pull/985) about this?)

## Example

[The notes](https://github.com/webprogramming260/webprogramming/blob/main/instruction/webFrameworks/react/vite/vite.md) go over using Vite to quickly set up a default application.

## Usage

In an NPM-initialized directory:

- **Import Vite w/ `npm create vite@latest`**: Vite sets up a bunch of scripts in `package.json` you can run.
  - (Chiefly, it adds `npm run dev` and `npm run build` scripts.)
- **Debug w/ `npm run dev`**.
  - This bundles code to a temporary directory that the Vite dbg HTTP server then loads from.
- **Build a deployable app w/ `npm run build`**.
  - The resulting production distribution's files are **stored in `dist/`**.

## Usage w/ `deployReact.sh`

For our project, we run `deployReact.sh`, a Bash script, to deploy our web app to our server.

`deployReact.sh` **runs `npm run build`** and then **deploys the resulting `dist/` directory** to your production server.

## Folder structure

*(EDIT: uhhh oops! looks like I forgot to make notes on this!)*
