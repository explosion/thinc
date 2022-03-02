import React from 'react'
import classNames from 'classnames'
import Tippy from '@tippy.js/react'
import slugify from 'slugify'

import { getStringChildren } from '../util'
import classes from '../styles/typography.module.sass'

export const H0 = ({ Component = 'h1', className, ...props }) => (
    <Headline Component={Component} className={classNames(classes.h0, className)} {...props} />
)
export const H1 = ({ Component = 'h1', className, ...props }) => (
    <Headline Component={Component} className={classNames(classes.h1, className)} {...props} />
)
export const H2 = ({ className, ...props }) => (
    <Headline Component="h2" className={classNames(classes.h2, className)} {...props} />
)
export const H3 = ({ className, ...props }) => (
    <Headline Component="h3" className={classNames(classes.h3, className)} {...props} />
)
export const H4 = ({ className, ...props }) => (
    <Headline Component="h4" className={classNames(classes.h4, className)} {...props} />
)
export const H5 = ({ className, ...props }) => (
    <Headline Component="h5" className={classNames(classes.h5, className)} {...props} />
)

const Permalink = ({ id, children }) => {
    if (!id) return children
    return (
        <a href={`#${id}`} className={classes.permalink}>
            {children}
        </a>
    )
}

const Headline = ({
    Component,
    id,
    name,
    banner,
    tag,
    new: newIn,
    meta,
    icon,
    hidden,
    className,
    children,
}) => {
    // This can be set via hidden="true" and as a prop, so we need to accept both
    if (hidden === true || hidden === 'true') return null
    const tags = tag ? tag.split(',').map(t => t.trim()) : []
    const wrapperClassNames = classNames(classes.headingWrapper, {
        [classes.headingWrapperBanner]: banner,
    })
    return (
        <Component id={id} name={name} className={classNames(classes.heading, className)}>
            <Permalink id={id}>
                {icon && <Icon name={icon} size="0.75em" />}
                <span className={wrapperClassNames}>
                    {children} {banner && <Banner text={banner} id={id} />}
                </span>{' '}
                {tags.map((t, i) => (
                    <Tag key={i}>{t}</Tag>
                ))}
                {newIn && (
                    <Tooltip
                        small
                        dark
                        content={`This feature is new and was introduced in version ${newIn}`}
                    >
                        <span>
                            <Tag variant="new">{newIn}</Tag>
                        </span>
                    </Tooltip>
                )}
            </Permalink>
            {meta && <div className={classes.meta}>{meta}</div>}
        </Component>
    )
}

const Banner = ({ text, id }) => {
    const anchor = `banner-${id || slugify(text)}`
    return (
        <svg viewBox="0 0 100 100" className={classes.banner}>
            <defs>
                <circle cx="50" cy="50" r="35" id={anchor} fill="none" stroke="#000" />
            </defs>
            <text className={classes.bannerText}>
                <textPath href={`#${anchor}`} textAnchor="middle" startOffset="50%">
                    {text}
                </textPath>
            </text>
        </svg>
    )
}

export const Tag = ({ variant, className, children }) => {
    const stringVersion = getStringChildren(children)
    const tagClassNames = classNames(classes.tag, className, {
        [classes.tagNew]: stringVersion === 'new' || variant === 'new',
    })
    return (
        <span className={tagClassNames}>
            {variant === 'new' && 'New: v'}
            {children}
        </span>
    )
}

export const Hr = () => <hr className={classes.hr} />

export const Tooltip = ({ small = false, dark = false, className, children, ...props }) => {
    const tooltipClassNames = classNames(classes.tooltip, className, {
        [classes.tooltipSmall]: !!small,
        [classes.tooltipDark]: !!dark,
    })
    return (
        <Tippy
            animateFill={false}
            distance={small ? 5 : 15}
            placement="bottom"
            className={tooltipClassNames}
            {...props}
        >
            {children}
        </Tippy>
    )
}

export const InlineList = props => <div className={classes.inlineList} {...props}></div>

export const Small = ({ element = 'div', children, ...props }) => {
    const Component = element
    return (
        <Component className={classes.small} {...props}>
            {children}
        </Component>
    )
}
