import React from 'react'
import rehypeReact from 'rehype-react'

import Link, { Button } from './components/link'
import { H2, H3, H4, H5, Hr, Tag, Small, InlineList } from './components/typography'
import { Tr } from './components/table'
import { CodeComponent, Pre, Kbd, TypeAnnotation, Ndarray, CodeScreenshot } from './components/code'
import { Infobox } from './components/box'
import Tutorials, { Colab } from './components/tutorials'
import Grid from './components/grid'
import { Tabs, Tab } from './components/tabs'
import Icon, { Emoji } from './components/icon'
import Quickstart from './components/quickstart'

export const renderAst = new rehypeReact({
    createElement: React.createElement,
    components: {
        h2: H2,
        h3: H3,
        h4: H4,
        h5: H5,
        hr: Hr,
        a: Link,
        button: Button,
        colab: Colab,
        code: CodeComponent,
        pre: Pre,
        kbd: Kbd,
        blockquote: Infobox,
        infobox: Infobox,
        tutorials: Tutorials,
        grid: Grid,
        tag: Tag,
        i: Icon,
        emoji: Emoji,
        tr: Tr,
        tt: TypeAnnotation,
        ndarray: Ndarray,
        'inline-list': InlineList,
        tabs: Tabs,
        tab: Tab,
        quickstart: Quickstart,
        'code-screenshot': CodeScreenshot,
        small: Small,
    },
}).Compiler
