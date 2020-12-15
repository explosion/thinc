import React from 'react'
import classNames from 'classnames'

import { isString } from '../util'
import classes from '../styles/table.module.sass'

function isDividerRow(children) {
    if (children.length && children[0].type == 'td') {
        return children[0].props.children[0].type == 'em'
    }
    return false
}

function isFootRow(children) {
    const rowRegex = /^(RETURNS|YIELDS)/
    if (children.length && children[0].type == 'td') {
        const cellChildren = children[0].props.children
        if (
            cellChildren[0] &&
            cellChildren[0].props &&
            isString(cellChildren[0].props.children[0])
        ) {
            return rowRegex.test(cellChildren[0].props.children[0])
        }
    }
    return false
}

export const Tr = ({ children, ...props }) => {
    const isDivider = isDividerRow(children)
    const isFoot = isFootRow(children)
    const trClasssNames = classNames(classes.tr, {
        [classes.foot]: isFoot,
        [classes.divider]: isDivider,
    })
    return (
        <tr className={trClasssNames} {...props}>
            {children}
        </tr>
    )
}
