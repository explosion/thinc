[![Netlify Status](https://api.netlify.com/api/v1/badges/d249ffd8-1790-4053-b6e8-5967ac68e4e1/deploy-status)](https://app.netlify.com/sites/cocky-hodgkin-996e5b/deploys)

## Setup and installation

The site is powered by [Gatsby](https://www.gatsbyjs.org/) and
[Markdown Remark](https://github.com/remarkjs/remark). To run the site, Node
10.15+ is required.

```bash
npm install  # install dependencies
npm run dev  # start dev server
```

A `.prettierrc` is included in the repo, so if you set up auto-formatting with
Prettier, it should match the style.

## Directory structure

- `/docs`: Docs pages as Markdown.
- `/src/pages`: JavaScript-formatted landing pages relative to the root.

## Markdown reference

The docs use various customized Markdown components for better visual
documentation. Here are the most relevant:

### Special syntax

#### Headings

Headings can specify optional attributes in curly braces, e.g. `#some_id` to add
a permalink and `id="some_id"` or a `tag` attribute with a string of one or more
comma-separated tags to be added after the headline text.

```markdown
## Headline 2

## Headline 2 {#some_id}

## Headline 2 {#some_id tag="method"}
```

#### Code

Code blocks can specify an optional title on the first line, prefixed by `###`.
The title also supports specifying attributes, including `small="true"` (small
font) and `highlight`, mapped to valid line numbers or line number ranges.

````markdown
```python
### This is a title {highlight="1,3-4"}
from thinc.api import Model, chain, Relu, Softmax

with Model.define_operators({">>": chain}):
    model = Relu(512) >> Relu(512) >> Softmax()
```
````

#### Tables

If a table row defines an italicized label in its first column and is otherwise
empty, it will be rendered as divider with the given label. This is currently
used for the "keyword-only" divider that separates positional and regular
keyword arguments from keyword-only arguments.

If the last row contains a bold `RETURNS` or `YIELDS`, the row is rendered as
the footer row with an additional divider.

```markdown
| Argument       | Type             | Description                                            |
| -------------- | ---------------- | ------------------------------------------------------ |
| `X`            | <tt>ArrayXd</tt> | The array.                                             |
| _keyword-only_ |                  |                                                        |
| `dim`          | <tt>int</tt>     | Which dimension to get the size for. Defaults to `-1`. |
| **RETURNS**    | <tt>int</tt>     | The array's inferred width.                            |
```

If you're specifying tables in Markdown, you always need a head row – otherwise,
the markup is invalid. However, if all head cells are empty, the header row will
not be rendered.

```markdown
|          |          |          |
| -------- | -------- | -------- |
| Column 1 | Column 2 | Column 3 |
```

### Custom markdown elements

#### `<infobox>` Infobox

Infobox with an optional variant attribute: `variant="warning"` or
`variant="danger"`.

```markdown
<infobox variant="warning">

This is a warning.

</infobox>
```

#### `<tt>` Type annotation

Should be used for Python type annotations like `bool`, `Optional[int]` or
`Model[ArrayXd, ArrayXd]`. Similar to regular inline code but will highlight the
elements and link the types if available. See
[`type-links.js`](src/type-links.js) for the type to link mapping.

```markdown
<tt>Tuple[str, int]</tt>
```

#### `<ndarray>` Arrays

Special type annotation for arrays with option to specify shape. Will link types
if available. See [`_type_links.json`](docs/_type_links.json) for the type to
link mapping.

```markdown
<ndarray shape="nI, nO">Array2d</ndarray>
```

#### `<tabs>` `<tabs>` Tabbed components

Will make each tab title a selectable button and allow tabbing between content.
Mostly used for longer code examples to show a variety of different examples
without making the page too verbose. The `id` is needed to distinguish multiple
tabbed components on the same page.

```markdown
<tabs id="some_tabs">
<tab title="Example 1">

Tab content 1 goes here

</tab>
<tab title="Example 2">

Tab content 2 goes here

</tab>
</tabs>
```

#### `<grid>` Simple two-column grid

Responsive grid for displaying child elements in two columns. Mostly used to
display code examples side-by-side.

````markdown
<grid>

```ini
### config.cfg {small="true"}
[training]
patience = 10
dropout = 0.2
```

```json
### Parsed {small="true"}
{
    "training": {
        "patience": 10,
        "dropout": 0.2
    }
}
```

</grid>
````

#### `<inline-list>` Inline list of meta info

Should contain an unnumbered list with items optionally prefixed with a bold
label. Mostly used in the layers API reference to document input/output types,
parameters and attributes in a concise way.

```markdown
<inline-list>

- **Label 1:** Some content
- **Label 2:** Some content

</inline-list>
```

#### `<tutorials>` Tutorial links with Colab buttons

Should contain an unnumbered list with IDs of the tutorials to include. See
[`_tutorials.json`](docs/_tutorials.json) for options. The tutorials section
will show each tutorial name and description with a button to launch the
notebook on Colab, if available.

```markdown
<tutorials>

- intro
- transformers_tagger
- parallel_training_ray

</tutorials>
```
