import React from 'react'
import classNames from 'classnames'
import { navigate } from 'gatsby'

import classes from '../styles/dropdown.module.sass'

const Dropdown = ({ defaultValue, className, onChange, children }) => {
    const defaultOnChange = ({ target }) => {
        const isExternal = /((http(s?)):\/\/|mailto:)/gi.test(target.value)
        if (isExternal) {
            window.location.href = target.value
        } else {
            navigate(target.value)
        }
    }
    return (
        <select
            defaultValue={defaultValue}
            className={classNames(classes.root, className)}
            onChange={onChange || defaultOnChange}
        >
            {children}
        </select>
    )
}

export default Dropdown
