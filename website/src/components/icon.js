import React from 'react'
import classNames from 'classnames'

import classes from '../styles/icon.module.sass'

import TwitterIcon from '../images/icons/twitter.svg'
import GitHubIcon from '../images/icons/github.svg'
import ArrowRightIcon from '../images/icons/arrow-right.svg'
import YesIcon from '../images/icons/yes.svg'
import NoIcon from '../images/icons/no.svg'
import CubeIcon from '../images/icons/cube.svg'
import FileIcon from '../images/icons/file.svg'

const ICONS = {
    twitter: TwitterIcon,
    github: GitHubIcon,
    right: ArrowRightIcon,
    yes: YesIcon,
    no: NoIcon,
    cube: CubeIcon,
    file: FileIcon,
}

export default ({ name, size = 16, alt = null, className, ...props }) => {
    const SvgIcon = ICONS[name]
    if (!SvgIcon) throw Error(`Invalid icon name: '${name}'`)
    const style = { minWidth: size }
    const iconClassNames = classNames(classes.root, className, {
        [classes.red]: name === 'no',
        [classes.green]: name === 'yes',
    })
    const altTexts = { yes: 'yes', no: 'no' }
    return (
        <SvgIcon
            className={iconClassNames}
            width={size}
            height={size}
            style={style}
            aria-label={alt != null ? alt : altTexts[name] || null}
            {...props}
        />
    )
}

export const Emoji = ({ alt, children }) => {
    const attrs = alt ? { role: 'img', 'aria-label': alt } : { 'aria-hidden': true }
    return (
        <span className={classes.emoji} {...attrs}>
            {children}
        </span>
    )
}
