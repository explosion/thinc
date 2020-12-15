import React, { useState } from 'react'
import classNames from 'classnames'

import classes from '../styles/tabs.module.sass'

export const Tabs = ({ id = 'tabs', defaultActive = 0, children }) => {
    const [active, setActive] = useState(defaultActive)
    const childTabs = children
        .filter(c => c && c != '\n' && c.props)
        .map((c, i) => ({ component: c, title: c.props.title, id: i }))
    return (
        <div className={classes.root}>
            <header className={classes.header}>
                {childTabs.map(tab => {
                    const itemId = `${id}-${tab.id}`
                    const isChecked = tab.id === active
                    const chipClassNames = classNames(classes.chip, {
                        [classes.chipActive]: isChecked,
                    })
                    return (
                        <label className={chipClassNames} htmlFor={itemId} key={tab.id}>
                            <input
                                className={classes.input}
                                name={id}
                                type="radio"
                                id={itemId}
                                defaultChecked={isChecked}
                                onChange={() => setActive(tab.id)}
                            />
                            {tab.title}
                        </label>
                    )
                })}
            </header>
            {childTabs.map(({ component, id }) => (id == active ? component : null))}
        </div>
    )
}

export const Tab = ({ children }) => {
    return <section className={classes.tab}>{children}</section>
}
