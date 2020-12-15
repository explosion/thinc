import React, { useState, useEffect, Fragment } from 'react'
import classNames from 'classnames'
import highlightCode from 'gatsby-remark-prismjs/highlight-code.js'
import rangeParser from 'parse-numeric-range'

import Icon from './icon'
import Link from './link'
import { H5 } from './typography'
import { isString, htmlToReact, isTrue, getStringChildren } from '../util'
import CUSTOM_TYPES from '../../docs/_type_links.json'
import classes from '../styles/code.module.sass'

const GITHUB_URL_SPLIT_PATH = 'thinc/'
const DEFAULT_TYPE_URL = CUSTOM_TYPES.__default__

function getGitHubPath(url) {
    if (!url.includes(GITHUB_URL_SPLIT_PATH)) {
        return url.split('github.com/').slice(-1)
    }
    const path = url.split(`/${GITHUB_URL_SPLIT_PATH}`).slice(-1)
    return `${GITHUB_URL_SPLIT_PATH}${path}`
}

function linkType(el, showLink = true) {
    if (!isString(el) || !el.length) return el
    const elStr = el.trim()
    if (!elStr) return el
    const typeUrl = CUSTOM_TYPES[elStr]
    const url = typeUrl == true ? DEFAULT_TYPE_URL : typeUrl
    const ws = el[0] == ' '
    return url && showLink ? (
        <Fragment>
            {ws && ' '}
            <Link to={url}>{elStr}</Link>
        </Fragment>
    ) : (
        el
    )
}

export const InlineCode = ({ wrap, prompt, className, children, ...props }) => {
    const codeClassNames = classNames(classes.inlineCode, className, {
        [classes.wrap]: wrap || (isString(children) && children.length >= 20),
    })
    return (
        <code data-prompt={prompt} className={codeClassNames} {...props}>
            {children}
        </code>
    )
}

export const Kbd = ({ children }) => <kbd className={classes.kbd}>{children}</kbd>

export const Pre = ({ children, className, ...props }) => (
    <pre className={classNames(classes.pre, className)} {...props}>
        {children}
    </pre>
)

export const CodeComponent = ({
    lang = 'none',
    highlight,
    github = false,
    children,
    className,
    ...props
}) => {
    const codeText = Array.isArray(children) ? children.join('') : children || ''
    const isGitHub = isTrue(github)

    if (isGitHub) {
        return <GitHubCode url={codeText} lang={lang} />
    }
    const highlightRange = highlight ? rangeParser.parse(highlight).filter(n => n > 0) : []
    const html = lang === 'none' ? codeText : highlightCode(lang, codeText, {}, highlightRange)
    return <CodeWrapper html={html} lang={lang} {...props} />
}

const CodeWrapper = ({
    title,
    html,
    lang = 'none',
    small = false,
    wrap = false,
    prompt,
    className,
    children,
    ...props
}) => {
    const codeClassNames = classNames(className, `language-${lang}`, {
        [classes.small]: isTrue(small),
        [classes.wrap]: isTrue(wrap),
    })
    const tagClassNames = classNames(classes.titleTag, {
        [classes.titleTagSmall]: isTrue(small),
    })
    return (
        <>
            {title && (
                <span className={classes.title}>
                    <span className={tagClassNames}>{title}</span>
                </span>
            )}
            {children}
            <InlineCode prompt={prompt} className={codeClassNames} {...props}>
                {htmlToReact(html)}
            </InlineCode>
        </>
    )
}

const GitHubCode = React.memo(({ url, lang }) => {
    const errorMsg = `Can't fetch code example from GitHub :(

Please use the link above to view the example. If you've come across
a broken link, we always appreciate a pull request to the repository,
or a report on the issue tracker. Thanks!`
    const [initialized, setInitialized] = useState(false)
    const [code, setCode] = useState(errorMsg)
    const rawUrl = url
        .replace('github.com', 'raw.githubusercontent.com')
        .replace('/blob', '')
        .replace('/tree', '')

    useEffect(() => {
        if (!initialized) {
            setCode(null)
            fetch(rawUrl)
                .then(res => res.text().then(text => ({ text, ok: res.ok })))
                .then(({ text, ok }) => {
                    setCode(ok ? text : errorMsg)
                })
                .catch(err => {
                    setCode(errorMsg)
                    console.error(err)
                })
            setInitialized(true)
        }
    }, [initialized, rawUrl, errorMsg])
    const html = lang === 'none' || !code ? code : highlightCode(lang, code)
    const title = (
        <Link to={url} hidden>
            <Icon name="github" size={12} /> View on GitHub
        </Link>
    )
    return (
        <CodeWrapper html={html} lang={lang} title={title} className={classes.maxHeight} small>
            <H5 className={classes.githubTitle}>
                <Link to={url} hidden>
                    <Icon name="file" size={14} className={classes.githubIcon} />{' '}
                    {getGitHubPath(url)}
                </Link>
            </H5>
        </CodeWrapper>
    )
})

export const CodeBlock = props => (
    <Pre>
        <CodeComponent {...props} />
    </Pre>
)

export const TypeAnnotation = ({ lang = 'python', link = true, children }) => {
    const showLink = isTrue(link)
    const code = Array.isArray(children) ? children.join('') : children || ''
    const html = lang === 'none' || !code ? code : highlightCode(lang, code)
    const result = htmlToReact(html)
    const elements = Array.isArray(result) ? result : [result]
    const annotClassNames = classNames(
        'type-annotation',
        `language-${lang}`,
        classes.typeAnnotation,
        {
            [classes.wrap]: code.length >= 20,
        }
    )
    return (
        <code className={annotClassNames}>
            {elements.map((el, i) => (
                <Fragment key={i}>{linkType(el, showLink)}</Fragment>
            ))}
        </code>
    )
}

export const Ndarray = ({ shape, link = true, children }) => {
    const strChildren = getStringChildren(children)
    const html = highlightCode('python', strChildren)
    const result = htmlToReact(html)
    const elements = Array.isArray(result) ? result : [result]
    return (
        <dfn className={classNames(classes.ndarray)}>
            <InlineCode>
                {elements.map((el, i) => (
                    <Fragment key={i}>{linkType(el, isTrue(link))}</Fragment>
                ))}
            </InlineCode>{' '}
            {shape && (
                <span className={classes.ndarrayShape}>
                    <Icon name="cube" size={13} /> {`(${shape})`}
                </span>
            )}
        </dfn>
    )
}

export const CodeScreenshot = ({ width, children }) => (
    <figure style={{ width }} className={classes.screenshot}>
        {children}
    </figure>
)
