import React from 'react'
import classNames from 'classnames'

import classes from '../styles/box.module.sass'

export const Box = ({ Component = 'section', id, className, children }) => (
    <Component id={id} className={classNames(classes.root, className)}>
        {children}
    </Component>
)

export const Infobox = ({ variant, children }) => {
    const infoboxClassNames = classNames(classes.infobox, {
        [classes.warning]: variant === 'warning',
        [classes.danger]: variant === 'danger',
    })
    return (
        <Box Component="aside" className={infoboxClassNames}>
            <span className={classes.icon}>{variant ? '!' : 'i'}</span>
            {children}
        </Box>
    )
}
