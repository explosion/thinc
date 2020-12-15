import React from 'react'

import Link from './link'
import { H2 } from './typography'
import Icon from './icon'
import { isTrue } from '../util'
import ColabBadge from '../images/colab-badge.svg'
import TUTORIALS from '../../docs/_tutorials.json'
import classes from '../styles/tutorials.module.sass'

export const ROOT = 'explosion/thinc/blob/master/examples'

function flattenChildren(children) {
    let result = []
    for (let child of children) {
        if (React.isValidElement(child)) {
            result = [...result, ...flattenChildren(child.props.children)]
        } else {
            result.push(child)
        }
    }
    return result.filter(el => el != '\n')
}

export default ({ title = 'Examples & Tutorials', id = 'tutorials', header = true, children }) => {
    const items = flattenChildren(children).map(name => {
        const item = TUTORIALS[name]
        if (!item) throw Error(`Unknown tutorial: '${name}'`)
        return { name, item }
    })
    const baseUrl = `https://github.com/${ROOT}/`
    return (
        <section className={classes.root}>
            {isTrue(header) && (
                <header className={classes.header}>
                    <H2 id={id} className={classes.headerTitle}>
                        {title}
                    </H2>
                    <Link to={baseUrl} hidden className={classes.headerLink}>
                        View more <Icon name="right" />
                    </Link>
                </header>
            )}
            <ul className={classes.list}>
                {items.map(({ name, item }, i) => (
                    <li className={classes.item} key={i}>
                        <div>
                            <Icon name="file" size={14} />{' '}
                            <Link key={name} to={`${baseUrl}${item.url}`}>
                                <strong>{item.title}</strong>
                            </Link>
                            {item.description && ` · ${item.description}`}
                        </div>
                        <div>{!item.standalone && <Colab url={`${ROOT}/${item.url}`}></Colab>}</div>
                    </li>
                ))}
            </ul>
        </section>
    )
}

export const Colab = ({ url }) => (
    <Link
        to={`https://colab.research.google.com/github/${url}`}
        aria-label="Open in Colab"
        className={classes.badge}
        hidden
    >
        <ColabBadge />
    </Link>
)
