# Web frameworks notes

## Web frameworks

Frameworks make web dev easier by **providing tools for common tasks**, e.g.:

- Modularizing code.
- Creating single page applications.
- Simplifying reactivity.
- Supporting diverse hardware devices.
- etc.

### Hybrids

Some frameworks combine HTML/CSS/JS into a single hybrid file. Abstracting away the core web file formats **puts the focus on functional components rather than files.**

E.g.:

- React JSX
- Vue SFC
- Svelte

### There are so many frameworks!

[The notes](https://github.com/webprogramming260/webprogramming/blob/main/instruction/webFrameworks/introduction/introduction.md#hello-world-examples) go over a quadrillion different frameworks. But from what I know, we only use React in this class, so I didn't take notes on this part.

If you're curious popularity of different frameworks, check out the popularity poll at [StateOfJS](https://stateofjs.com/).

## Toolchains

"Toolchain" refers to the list of tools you use for development, and how they all connect to each other.

The notes listed some common functional pieces in a web app tool chain, which I thought was interesting. So here they are:

- Code repo: For storing & sharing code.
- Linter: For keeping your code idiomatic.
- Prettier: For keeping your code formatted to a shared standard.
- Transpiler: For compiling code to a different format.
- Polyfill: For generating backwards compatible code for supporting old browser versions.
- Bundler: For packaging your code into bundles for delivery to browser, enabling compatibility (e.g. ES6 module support (huh?)) or enhancing performance (e.g. w/ lazy loading).
- Minifier: For reducing code to make code files smaller & thus more efficient to deploy. (e.g., removing whitespace, renaming vars, etc.)
- Testing: For automating tests at multiple levels.
- Deployment: For automating packaging & delivery of code from dev environment to production environment.

### Toolchain for our React project

| Tool | Purpose |  
| ---- | ------- |  
| [Github](https://github.com/) | Code repo. |  
| [Vite](https://vitejs.dev/) | JSX, TS, dev & debugging support. |  
| [ESBuild](https://esbuild.github.io/) (w/ [Babel](https://babeljs.io/docs/en/) underneath) | Converting to ES6 modules & transpiling. |  
| [Rollup](https://rollupjs.org/) | Bundling & tree shaking. |  
| [PostCSS](https://github.com/webprogramming260/webprogramming/blob/main/instruction/webFrameworks/react/toolChains) | CSS transpiling. |  
| Bash script (`deployReact.sh`) | Deployment. |  

You don't have to fully understand each of these pieces, but the more you know about them the more you can optimize your development efforts, the notes say.
