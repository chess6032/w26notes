# CSS Notes

## Methods for adding CSS styling

### Inline styling

```html
<!DOCTYPE html>
<head>
    <title>Inline styling</title>
</head>
<body>
    <p style="color:red">CSS</p>
</body>
```

### `<style>` element

```html
<!DOCTYPE html>
<head>
    <title>Inline styling</title>
    <style>
        p {
            color: red;
        }
    </style>
</head>
<body>
    <p>CSS</p>
</body>
```

### `<link>` to external CSS document

```html
<!DOCTYPE html>
<head>
    <title>Inline styling</title>
    <link rel="stylesheet" href="styles.css" />
</head>
<body>
    <p>CSS</p>
</body>
```

(You can link multiple stylesheets. Later links have precedence over previous ones.)

styles.css:

```css
p {
    color: green;
}
```

### Precedence ("Specificity")

- Inline styling has precedence over an HTML `<style>` element or a stylesheet `<link>`.
- Between a `<style>` element and a stylesheet `<link>`, whichever one is loaded *last* has higher precedence.
  - (i.e., the one that's written on a lower line.)

## CSS rule terminology

This is a CSS **"rule"**&mdash;or **"rule set"**, depending on who you ask:

```
p {
    color: green;
}
```

- `p` is the **"selector"**.
- Everything inside `{ }` is part of the **"declaration block"**.
- The statement `color: green;` is called a **"declaration"**.
  - `color` is the declaration's **"property"**.
  - `green` is the declaration's **"value"**.

## Inheritance

### Elements inherit their parent's styling

- Elements inherit styling from the elements they're nested inside of.

For example, in the following code, the `<p>` element's text will be GREEN, because it inherits `<body>`'s `color` property.

```html
<head>
    <style>
        body {
            color: green;
        }
    </style>
</head>
<body>
    <p>I'm inheriting my parent's <code>color</code> property.</p>
</body>
```

### Child declarations overwrite parent declarations

- After a child element inherits from its parent, its own styling rule is then applied. 
  - Thus, for any properties modified in both the parent's and child's rule, the child's declaration will overwrite the parent's.

In this example, the `<span>` element first inherits red coloring from the `body` rule, then that gets overwritten with `p`'s green coloring, and then finally that is overwritten to purple by the `span` rule. So the text will display as purple.

```html
 <style>
  body {
   font-size: 25vh;
   color: red;
  }
  p {
   color: green;
  }
  span {
   color: purple;
  }
 </style>
 <body>
  <p>
    <span>this will be purple</span>
  </p>
 </body>
```

## Selectors

Selectors are used to "find" the HTML element(s) to apply a rule's styling to.

### Simple selectors

| Selector               | Meaning                                           | CSS Example   | Ex. of Matching HTML |  
| --------               | -------                                           | -----------   | -------------------- |  
| **element**            | All elements of a specific name.                  | `div`         | `<div>` |  
| **ID**                 | All elements with a given ID.                     | `#root`       | `<div id="root">` |  
| **class**              | All elements with a given class.                  | `.highlight`  | `<div class="highlight">` <br/> `<p class="highlight">` |  
| **element & class**    | Any elements of a specific name w/ a given class. | `p.highlight` | `<p class="highlight">` |  
| **universal selector** | ALL elements.                                     | `*`           | Literally anything bruv. |  
| **list of elements**   | All elements with any of a list of names.         | `div, p`      | `<p>` <br/> `<div>` |  

(Actually, I think the list of elements entry in that table technically describes a CSS combinator...)

### Combinators

A combinator defines the relationship btwn two or more selectors for a single rule.

| Combinator         | Symbol        | Description | CSS Example |  
| ----------         | ------        | ----------- | ----------- |  
| Descendant         | <code>&nbsp;</code> (space)   | Selects all descendents of a specified element. That includes children, grandchildren, and on. | `div p` would select all `<p>` elements inside `<div>` elements. |  
| Child              | `>`           | Selects all direct children of a specfied element. | `div > p` would select only `<p>` elements that are *direct* children of a `<div>` element. | 
| Next-sibling       | `+`           | Selects an element if it immediately follows an element&mdash;and they're siblings (i.e. share a parent). | `div + p` would select any `<p>` that immediately follows a `<div>`&mdash;and share its parent. | 
| Subsequent-sibling | `~`           | Selects all sibling elements that follow an element. | `div ~ p` would select all `<p>` elements that come after a `<div>`&mdash;and share its parent. |  

I got all this info from the [w3schools page on CSS combinators](https://www.w3schools.com/css/css_combinators.asp).

(There's also a namespace selector, which is listed in the [w3school's CSS Combinators Reference](https://www.w3schools.com/cssref/css_ref_combinators.php) but  not the page talking about combinators.)

### Pseudo-classes (`:`)

Pseudo-class selectors are **state-based**. They target a simple selector when it's in a specific state.

Pseudo-classes are denoted by a **single colon**, followed by the pseudo-class name: `selector:psuedo-class-name`. Pseudo-class names are not case-sensitive.

Examples of pseudo-classes:

- `:focus` (activates when focused on&mdash;by clicking).
- `:hover` (activates when the mouse hovers over the element).
- `:link` (for unvisited links).
- `:visited` (for visited links).
- `:active` (for "activated" links).

(NOTE: For links, `a:hover` MUST come after `a:link` and `a:visited`, and `a:active` MUST come after `a:hover`...for some reason)

Looks like you can get really fancy with it. See the [w3schools page on CSS pseudo-classes](https://www.w3schools.com/css/css_pseudo_classes.asp) for this info here, and more.

### Pseudo-elements (`::`)

A CSS pseudo-element is a keyword that can be added to a selector to style a specific part of an element.

Syntax: `selector::pseudo-element-name`. You can also add pseudo-element rules to all selectors by not putting anything bofore the double colon: `::pseudo-element-name`

Common uses for psuedo-elements:

- Style the first letter (`::first-letter`) or first line (`::first-line`) of an element.
- Insert content before `::before` or after (`::after`) an element. (Use the `content` property.)
- Style the markers of list items (`::marker`).
- Style the user-selected portion of an element (`::selection`). (e.g., text selected by a user.)
- Style the viewbox behind dialog box elements (`dialog::backdrop`).

For more information, see the [w3schools page on CSS pseudo-elements](https://www.w3schools.com/css/css_pseudo_elements.asp). For a list of all CSS pseudo-elements, see the [w3schools CSS Pseudo-elements Reference](https://www.w3schools.com/cssref/css_ref_pseudo_elements.php).

### Attribute selectors

[TODO](https://www.w3schools.com/css/css_attribute_selectors.asp)

### Precedence

The levels of "specificity" (i.e. precedence) for CSS selector types is thus (in descending order): 

1. ID (`#`)
2. Class (`.`), Attribute `[]`, Pseudo-class `:`
3. Element (e.g. `div`), Pseudo Element `::`
4. Universal selector (`*`)

So, referencing that order, rule precedence works (within a single stylesheet) like this:

- Rules with higher specificity override rules with lower specificity.
- Between rules with equal specificity, the one that appears LATER in the stylesheet wins.