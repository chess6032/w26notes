# Creating responsive design w/ CSS

Responsive design is about making your website's layout dynamic, resizing/scaling so that it is looks good on any screen.

## Viewport meta tag

To make your website scale well on mobile devices, you should include this `<meta>` tag inside your web page's `<head>` section.

```html
<meta name="viewport" content="width=device-width,initial-scale=1" />
```

- `width=device-width` tells the browser to set the page width to match the device's screen width.
- `initial-scale=1` sets the initial zoom level to 100% when the page first loads.

Without this tag, mobile browsers oft try to display the full desktop version of a website by zooming way out. But that makes everything tiny and hard to read.

> **You should include this tag in ALL of your web pages.**

## Display property

The CSS `display` property allows you to change how an HTML element is rendered by the browser.

Here are some commonly used options:

| Value    | Effect                                     | Notes |  
| -----    | ------                                     | ----- |  
| `none`   | Element is **not displayed**.              | The element still exists; it just isn't rendered by the browser. |  
| `block`  | Element's **width fills parent** element.  | `p` and `div` elements use block display by default. |  
| `inline` | Element's **width hugs content**.          | `b` and `span` elements use inline display by default. |  
| `grid`   | Displays element's children in rows AND columns. | (See [grid](#grid)) |  
| `flex`   | Displays element's children in rows OR columns.  | (See [flex](#flex)) |  

## Grid display

The `grid` display layout lets you display a group of child elements in a responsive grid. 

### Length unit: `fr`

`fr` stands for "fractional unit", meaning a fraction of the parent's size. They're only usable in the context of grid layouts.

For columns, 1fr = parent's width; for rows, 1fr = parent's height. ...sorta.

`fr` is similar to `%`, but with a key difference: The length of 1fr is calculated after fixed-sized content, gaps, padding, etc. are accounted for. So in effect, **`fr` distributes available space**. This provides several advantages:

- `fr` prevents overflows that would occur w/ `%`.
- `fr` handles gaps elegantly (when using `grid-gap`).
- `fr` can be used alongside fixed units without breaking everything.


### Example

```css
.container {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  grid-auto-rows: 300px;
  grid-gap: 1em;
}
```

- `grid-template-columns` specifies the number & widths of columns.
  - So for a declaration like `grid-template-columns: w1 w2 w3 ... wn;`, there would be $n$ columns, and every length $w_n$ corresponds to the width of the $n^\text{th}$ column.
  - In the code above, each column will have an equal width, and the number of columns will change w/ the viewport size.
    - `repeat()` repeats the column pattern `auto-fill`, which creates as many columns as will fit in the container.
      - (`auto-fill` is only usable within a `repeat()` call)
    - `minmax(300px, 1fr)` sets a range for the columns' widths: Each column is at least 300px, but can grow up to 1fr.
- `grid-auto-rows: 300px;` fixes every row's height to 300px.
- `grid-gap` specifies the distance btwn each grid cell. (As if it were the `margin` for the cells.)

Check out [this CodePen](https://codepen.io/leesjensen/pen/GRGXoWP) to see this code in action.

### `grid-auto-` vs. `grid-template-` for cols/rows

For defining columns/rows in a grid display, you have two options: `grid-auto-columns`/`grid-auto-rows` and `grid-template-columns`/`grid-template-rows`

- `template` defines the size & number of cols/rows in the **explicit grid** (the one you intentionally set).
- `auto` defines the size of any **implicit cols/rows** that are automatically created by the browser to accommodate extra content.
  - Cols/rows generate automatically when content is placed outside the boundaries of the `template` definitions.

## Flex display

The `flex` display layout displays a group of chilren in either rows OR columns. It's useful when you want to partition your application into areas that resposnively move around as the window resizes or rotates.

A container element that uses `display: flex;` is oft called a "flexbox".

To see flexbox in action, check out [this CodePen](https://codepen.io/leesjensen/pen/MWOVYpZ) (or [this one](https://codepen.io/leesjensen/pen/abamMqL)). (Seriously the [GitHub instructions](https://github.com/webprogramming260/webprogramming/blob/main/instruction/css/flexbox/flexbox.md) for flexbox are vague.)

### Container properties

- `flex-flow`: Shorthand for `flex-direction` and `flex-wrap`.
  - `flex-direction`: Sets direction of flex items.
    - `row` (default) or `row-reverse`.
    - `column` or `column-reverse`.
  - `flex-wrap`: Specifies whether/not flex items should wrap when there is not enough room for them on one flex line.
    - `nowrap` (default)
    - `wrap`
    - `wrap-reverse`
- `justify-content`: Aligns the flex items when they do not use all available space on the **main-axis** (i.e. horizontally). (More info [here](https://www.w3schools.com/css/css3_flexbox_container_justify.asp).)
- `align-items`: Aligns the flex items when they do not use all available space on the **cross-axis** (i.e. vertically). (More info [here](https://www.w3schools.com/css/css3_flexbox_container_align.asp))
- `align-content`: Aligns the flex *lines* when there is extra space in the cross axis and flex items wrap.

(For more information, see the [w3schools page on flex containers](https://www.w3schools.com/css/css3_flexbox_container.asp) in CSS.)

### Containee properties

- `flex` property.
  - Shorthand for `flex-grow`, `flex-shrink`, and `flex-basis`.


## Float property

The `float` CSS property allows an element to "float around" in its container, allowing inline elements to wrap around it.

Some common values for `float`:

- `none`
- `left` or `right`
- `inline-end` or `inline-start`

Check out [this CodePen](https://codepen.io/leesjensen/pen/MWBRWPP) to see this in action.

## Media queries

With the `@media` at-rule, you can dynamically detect the size & orientation of the user's device and apply CSS rules to accommodate the change. This is called a "media query" (ig).

A media query takes one or more predicates, each separated by boolean operators.

Check out [this CodePen](https://codepen.io/leesjensen/pen/rNKZOva) (or [this one](https://codepen.io/leesjensen/pen/NWzLGmJ)) to see this in action..

## Misc

### `:nth-child()` pseudo-class

`selector:nth-child(N)` will target the $N^\text{th}$ child of the element grabbed by `selector`.

### Inline `flex` and `grid` layouts

It looks like in addition to `display: flex;` and `display: grid;`, you can do `display: inline-flex;` and `display: inline-grid;`...idrk what that means tho.