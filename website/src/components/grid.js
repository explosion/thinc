import React from 'react'
import classNames from 'classnames'

import classes from '../styles/grid.module.sass'

export default ({ layout = 'auto', children }) => {
    const gridClassNames = classNames(classes.root, {
        [classes.feature]: layout === 'feature',
    })
    return <div className={gridClassNames}>{children}</div>
}
