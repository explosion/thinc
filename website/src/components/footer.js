import React from 'react'
import { graphql, StaticQuery } from 'gatsby'
import classNames from 'classnames'

import Link from './link'
import Icon from './icon'
import classes from '../styles/footer.module.sass'

export default ({ className }) => (
    <StaticQuery
        query={query}
        render={({ site }) => {
            const { twitter, github, company, companyUrl, imprintUrl } = site.siteMetadata
            return (
                <footer className={classNames(classes.root, className)}>
                    <ul className={classes.meta}>
                        <li>
                            <span style={{ fontFamily: 'sans-serif' }}>&copy;</span> 2017-
                            {new Date().getFullYear()}{' '}
                            <Link to={companyUrl} hidden>
                                {company}
                            </Link>
                        </li>
                        <li>
                            <Link to={imprintUrl} hidden>
                                Legal & Imprint
                            </Link>
                        </li>
                        <li>
                            <Link to={`https://twitter.com/${twitter}`} hidden aria-label="Twitter">
                                <Icon name="twitter" />
                            </Link>
                        </li>
                        <li>
                            <Link to={`https://github.com/${github}`} hidden aria-label="GitHub">
                                <Icon name="github" />
                            </Link>
                        </li>
                    </ul>
                </footer>
            )
        }}
    />
)

const query = graphql`
    query {
        site {
            siteMetadata {
                company
                companyUrl
                imprintUrl
                email
                twitter
                github
            }
        }
    }
`
