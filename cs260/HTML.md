# HTMl

## Common HTML Elements

| element   | meaning                                                                |
| --------- | ---------------------------------------------------------------------- |
| `html`    | The page container                                                     |
| `head`    | Header information                                                     |
| `title`   | Title of the page                                                      |
| `meta`    | Metadata for the page such as character set or viewport settings       |
| `script`  | JavaScript reference. Either a external reference, or inline           |
| `include` | External content reference                                             |
| `body`    | The entire content body of the page                                    |
| `header`  | Header of the main content                                             |
| `footer`  | Footer of the main content                                             |
| `nav`     | Navigational inputs                                                    |
| `main`    | Main content of the page                                               |
| `section` | A section of the main content                                          |
| `aside`   | Aside content from the main content                                    |
| `div`     | A block division of content                                            |
| `span`    | An inline span of content                                              |
| `h<1-9>`  | Text heading. From h1, the highest level, down to h9, the lowest level |
| `p`       | A paragraph of text                                                    |
| `b`       | Bring attention                                                        |
| `table`   | Table                                                                  |
| `tr`      | Table row                                                              |
| `th`      | Table header                                                           |
| `td`      | Table data                                                             |
| `ol,ul`   | Ordered or unordered list                                              |
| `li`      | List item                                                              |
| `a`       | Anchor the text to a hyperlink                                         |
| `img`     | Graphical image reference                                              |
| `dialog`  | Interactive component such as a confirmation                           |
| `form`    | A collection of user input                                             |
| `input`   | User input field                                                       |
| `audio`   | Audio content                                                          |
| `video`   | Video content                                                          |
| `svg`     | Scalable vector graphic content                                        |
| `iframe`  | Inline frame of another HTML page                                      |


<br>

Copied from the [CS260 HTML introduction page.](https://github.com/webprogramming260/webprogramming/blob/main/instruction/html/introduction/introduction.md#common-elements)

## Special characters



| Character | Entity      |
| --------- | ----------- |
| &amp;     | `&amp;`     |
| <         | `&lt;`      |
| >         | `&gt;`      |
| "         | `&quot;`    |
| '         | `&apos;`    |
| &#128512; | `&#128512;` |

Copied from the [CS260 HTML introduction page.](|)

## Common Attributes

- `id`: unique identifier for an element.
- `class`: classifies element into a named group ("class") of elements.

## Hyperlinks

```html
<a href="LINK">TEXT</a>
```

## Images

```html
<img src="LINK" alt="ALT TEXT">
```

You can optionally add `width` and `height` attributes to specify the image's dimensions.

### Hyperlinks to images

```html
<a href="https://www.w3schools.com">
  <img src="w3html.gif" alt="W3Schools.com" width="100" height="132">
</a>
```

(Thanks to [GeeksForGeeks for this one :D](https://www.w3schools.com/tags/tag_img.asp#:~:text=Try%20it%20Yourself%20%C2%BB-,Example,-How%20to%20add))