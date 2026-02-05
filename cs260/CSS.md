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

<!-- (Actually, I think the list of elements entry in that table technically describes a CSS combinator...maybe?) -->

The `*` selector is pretty dangerous. If you're importing a library, `*` might override that library's rules!

### Combinators (selecting by relationship)

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

### Attribute selectors (`[]`)

CSS attribute selectors select & style HTML elements w/ a specific attribute or attribute value&mdash;or both.

- `[attribute]` selects elements that have an `attribute` attribute.
- `[attribute="value"]` selects elements whose `attribute` attr is set to exactly `"value"`. 
- `[attribute~="value"]` selects elements whose `attribute` attr is `"value"` or contains `"value"` in a space-separated list.
  - e.g., `[title~="flower"]` matches elements w/ `title="flower"`, `title="summer flower"`, `title="flower new"`, but NOT `title="my-flower"` or `title="flowers"`.
- `[attribute|="value"]` selects elements whose `attribute` attr is `"value"` or starts with `value-` (`value` followed by a hyphen).
  - e.g. `[class|="top"]` would match `class="top"` or `class="top-text"`.
- `[attribute^="value"]` selects elements whose `attribute` attr STARTS w/ `value`.
- `[attribute$="value"]` selects elements whose `attribute` attr ENDS w/ `value`.
- `[attributes*="value"]` selects elements whose `attribute` attr CONTAINS `value` anywhere within it.

For more information, see [w3school's page on attribute selectors](https://www.w3schools.com/css/css_attribute_selectors.asp).

### Precedence

The levels of "specificity" (i.e. precedence) for CSS selector types is thus (in descending order): 

1. ID (`#`)
2. Class (`.`), Attribute `[]`, Pseudo-class `:`
3. Element (e.g. `div`), Pseudo Element `::`
4. Universal selector (`*`)

So, referencing that order, rule precedence works (within a single stylesheet) like this:

- Rules with higher specificity override rules with lower specificity.
- Between rules with equal specificity, the one that appears LATER in the stylesheet wins.

You can also add the `!important` keyword before a rule to forget about these specificity rules and always override.

## Properties

Here are some properties that are commonly styled.

- `background-color`.
- `border` (value: `color width style`) gives an element a border.
  - or you can set `border-color`, `border-width`, and `border-style` individually.
- `color` sets the text color within an element.
- `display` defines how to display the element and its children...?
- `font` (value: `family size style`) defines text font, size, & style (bold, italic, etc.).
  - or you can just set `font-family`, `font-size`, and `font-style` individually. 
- `margin` (value: `top right bottom left`) adds spacing between an element's edges and elements around it. (i.e. EXTERNAL spacing.)
- `padding` (value: `top right bottom left`) adds spacing between the element's contents and its edges. (i.e. INTERNAL spacing.)

For the `margin` and `padding` shorthands, use TRBL ("TRouBLe") mnemonic to remember the sequence of the four values you assign. (Or you can remember it's ordered clockwise starting at the top, if you're lame.)

## CSS Length Units

[(w3school's page on CSS units)](https://www.w3schools.com/cssref/css_units.php)

### Absolute lengths

Absolute length units are fixed: a length expressed in any of these will appear as exactly that size. 

They are not recommended for use on a screen b/c screen sizes vary so much. But they're chill if you're output medium is known&mdash;e.g. for print. (When printing a web page, browsers try much harder to match physical reality.)

Absolute units are anchored to each other:

$96 \text{px} = 1 \text{in} = 2.54 \text{cm} = 25.4 \text{mm} = 72 \text{pt} = 6 \text{pc}$

Those last two&mdash;`pt` and `pc`&mdash;are "points" (1/72 of an inch) and "peca" (12 pts), respectively.

`px` is not actually a screen pixel, it's a "CSS pixel". CSS spec defines 1px as the visual angle (i.e. amt of space taken up in your FOV) of one pixel on a 96dpi device when viewed at arm's length (28 inches, to be exact). Mathematically, this visual angle is 0.0213 degrees. (For more info, look up "CSS Reference Pixel".)

<!-- 
| Unit   | Description |  
| ------ | ----------- |  
| `px`   | Pixels.* |  
| `pt`   | "Points" (1/72 of an inch). |  
| `in`   | Inches. (96px). |   -->

### Relative lengths (use these)

| Unit   | Description |  
| ------ | ----------- |  
| `%`    | A percentage of the **parent's size**. |  
| `em`   | A multiplier of **element's font size**. |  
| `rem`  | A multiplier of the **root element font size**. |  
| `vw`   | 1vw = 1% of **viewport's width**. |  
| `vh`   | 1vh = 1% of **viewport's height**. |  
| `vmin` | 1vmin = 1% of **viewport's smaller dimension**. |  
| `vmax` | 1vmax = 1% of **viewport's larger dimension**. |  

- "viewport" is the browser's window size.
- "root element" is the very top-level element of the document, accessible via the `:root` selector in CSS.
  - For web pages, this is almost ALWAYS the `<html>` tag.
    - (Even if you don't write your HTML code w/ an `<html>` tag at the root, most browsers will "auto-correct" your code to make it the root.)

### Using units

- Whitespace may not appear between a number and its unit.
  - However, if the value is `0`, the unit may be omitted.
- w3school says that `em` and `rem` are the most practical units for creating a perfectly scalable layout.
- If you use a percentage to change the root element's font size, it takes a percentage of the browser's default font size.
  - In almost all browsers, this is 16px.
    - Many developers find this default annoying. A common hack is to set the root's `font-size` to `62.5%`. That way, 1rem = 10px, so the math is easy.

## Fonts

### Importing fonts

Here's how you would use a different font in your website:

```css
@font-face {
    font-family: 'Quicksand';
    src: url('https://yourstartup.click/path/to/font.ttf');
}

body {
    font-family: Quicksand, Helvetica, Arial, sans-serif;
}
```

### Importing fonts using a Google Fonts API call

```css
@import url('https://fonts.googleapis.com/css?family=Montserrat:100,400,900|Quicksand');

h1 {
 font-family: Quicksand, sans-serif;
}

p {
 font-family: 'Montserrat', sans-serif;
 font-weight: 100;
 font-size: 2em;
}
```

The above code downloads Montserrat font from Google Fonts w/ font weights 100, 400, and 900 (each a separate file), and it downloads Quicksand (one file for all weights because it's a variable font).

### Font fallbacks

When you set a `font-family` in a CSS declaration, you can include many in a (comma-separated) list. If the first font fails to load, then the browser will try to use the next one (i.e. "fallback"), and if that fails it'll try to use the next one, and so on. 

## Animations

Bruh I ain't doing all of ts for my website.

