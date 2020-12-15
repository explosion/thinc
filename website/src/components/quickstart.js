import React, { useState } from 'react'

import { CodeBlock } from './code'
import { H2 } from './typography'
import classes from '../styles/quickstart.module.sass'
import QUICKSTART from '../../docs/_quickstart.json'

export default ({ id, title, base = 'pip install thinc', suffix = '' }) => {
    const [extras, setExtras] = useState(QUICKSTART.defaults)
    const result = Object.values(extras).filter(v => v && v !== '-')

    function handleChange({ target }) {
        const { value, checked, name } = target
        const newExtras = { ...extras, [name]: checked ? value : null }
        setExtras(newExtras)
    }

    return (
        <section id={title ? null : id} className={classes.root}>
            {title && (
                <H2 className={classes.title} id={id}>
                    {title}
                </H2>
            )}
            <menu>
                {QUICKSTART.options.map(({ name, label, options = [], multi }, i) => (
                    <fieldset key={i} className={classes.fieldset}>
                        <legend className={classes.legend}>{label}:</legend>
                        {options.map(({ label, id, value, help }, j) => {
                            const inputId = id || value
                            const inputName = multi ? inputId : name
                            const checked = QUICKSTART.defaults[inputName] == value
                            return (
                                <span className={classes.option} key={j}>
                                    <input
                                        value={value}
                                        name={inputName}
                                        id={inputId}
                                        type={multi ? 'checkbox' : 'radio'}
                                        onChange={handleChange}
                                        className={classes.input}
                                        defaultChecked={checked}
                                    />
                                    <label htmlFor={inputId} className={classes.label}>
                                        {label}
                                    </label>
                                    {help && <span className={classes.help}>({help})</span>}
                                </span>
                            )
                        })}
                    </fieldset>
                ))}
            </menu>
            <CodeBlock title="Install command" prompt="$" lang="bash">
                {result.length ? `${base}[${result.join(',')}]` : base}
                {suffix}
            </CodeBlock>
        </section>
    )
}
