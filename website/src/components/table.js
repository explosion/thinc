import React from 'react'
import classNames from 'classnames'

import { isNumString, isString } from '../util'
import classes from '../styles/table.module.sass'

function isDividerRow(children) {
    if (children.length && children[0].type.name == 'Td') {
        return children[0].props.children[0].type == 'em'
    }
    return false
}

function isFootRow(children) {
    const rowRegex = /^(RETURNS|YIELDS)/
    if (children.length && children[0].type.name == 'Td') {
        const cellChildren = children[0].props.children
        if (
            cellChildren &&
            cellChildren.length &&
            cellChildren[0].props &&
            cellChildren[0].props.children.length &&
            isString(cellChildren[0].props.children[0])
        ) {
            return rowRegex.test(cellChildren[0].props.children[0])
        }
    }
    return false
}

export const Table = props => <table className={classes.root} {...props} />
export const Th = props => <th className={classes.th} {...props} />

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

export const Td = ({ children, ...props }) => {
    const tdClassNames = classNames(classes.td, {
        [classes.num]: isNumString(children),
    })
    return (
        <td className={tdClassNames} {...props}>
            {children}
        </td>
    )
}
