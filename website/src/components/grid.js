import React from 'react'

import classes from '../styles/grid.module.sass'

export default ({ children }) => {
    return <div className={classes.root}>{children}</div>
}
