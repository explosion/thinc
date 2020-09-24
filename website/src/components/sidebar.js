import React from 'react'
import classNames from 'classnames'

import Link from './link'
import Dropdown from './dropdown'
import Logo from '../images/logo.svg'
import classes from '../styles/sidebar.module.sass'

const DropdownNavigation = ({ items, defaultValue }) => {
    return (
        <div className={classes.dropdown}>
            <Dropdown className={classes.dropdownSelect} defaultValue={defaultValue}>
                <option disabled>Select page...</option>
                {items.map((section, i) =>
                    section.items.map(({ text, url }, j) => (
                        <option value={url} key={j}>
                            {section.label} &rsaquo; {text}
                        </option>
                    ))
                )}
            </Dropdown>
        </div>
    )
}

const Sidebar = ({ items = [], slug }) => {
    return (
        <menu className={classes.root}>
            <DropdownNavigation items={items} defaultValue={slug} />
            <Link to="/" hidden>
                <Logo className={classes.logo} />
            </Link>
            {items.map((section, i) => (
                <ul className={classes.section} key={i}>
                    <li className={classes.label}>{section.label}</li>
                    {section.items.map(({ text, url, onClick, isActive }, j) => {
                        const active = isActive || slug === url
                        const itemClassNames = classNames(classes.link, {
                            [classes.isActive]: active,
                        })
                        return (
                            <li key={j}>
                                <Link to={url} onClick={onClick} className={itemClassNames}>
                                    {text}
                                </Link>
                            </li>
                        )
                    })}
                </ul>
            ))}
        </menu>
    )
}
export default Sidebar
