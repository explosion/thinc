import React from 'react'
import { Link as GatsbyLink } from 'gatsby'
import classNames from 'classnames'

import Icon from './icon'
import classes from '../styles/link.module.sass'

const internalRegex = /(http(s?)):\/\/(explosion.ai|prodi.gy|spacy.io|irl.spacy.io|support.prodi.gy)/gi

const Link = ({ type, ...props }) => {
    // This is a workaround for the gatsby-remark-copy-linked-files, which
    // only recognizes <a> elements, not a custom <button>
    if (type === 'button') return <Button {...props} />
    const { children, to, href, onClick, hidden, className, ...other } = props
    const dest = to || href
    const external = /((http(s?)):\/\/|mailto:)/gi.test(dest)
    const internal = internalRegex.test(dest)
    const linkClassNames = classNames(classes.root, className, {
        [classes.hidden]: hidden,
    })

    if (!external) {
        if ((dest && /^#/.test(dest)) || onClick || other.target === '_blank') {
            return (
                <a href={dest} onClick={onClick} className={linkClassNames} {...other}>
                    {children}
                </a>
            )
        }
        return (
            <GatsbyLink to={dest} className={linkClassNames} {...other}>
                {children}
            </GatsbyLink>
        )
    }
    return (
        <a
            href={dest}
            className={linkClassNames}
            target="_blank"
            rel={internal ? 'noopener' : 'noopener nofollow noreferrer'}
            {...other}
        >
            {children}
        </a>
    )
}

export const Button = ({ children, primary = false, ...props }) => {
    const buttonClassNames = classNames(classes.button, {
        [classes.buttonPrimary]: primary,
    })
    return (
        <Link hidden={true} className={buttonClassNames} {...props}>
            <span className={classes.buttonContent}>
                {children} <Icon name="right" size={24} className={classes.buttonIcon} />
            </span>
        </Link>
    )
}

export default Link
