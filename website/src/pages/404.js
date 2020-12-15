import React from 'react'
import { graphql } from 'gatsby'

import Layout from '../components/layout'
import { Header } from '../components/landing'
import { H0, H4 } from '../components/typography'
import Link, { Button } from '../components/link'
import Footer from '../components/footer'
import classes from '../styles/landing.module.sass'

export default ({ data }) => {
    const { twitter, email } = data.site.siteMetadata
    return (
        <Layout title="404 Error" className={classes.root}>
            <Header logo={false}>
                <br />
                <H0>
                    404 Error
                    <H4>
                        If something is broken, we’d love to fix it.
                        <br />
                        You can send us an <Link to={`mailto:${email}`}>email</Link> or let us know
                        on <Link to={`https://twitter.com/${twitter}`}>Twitter</Link>.
                    </H4>
                </H0>

                <Button to={'/'} primary>
                    Go home
                </Button>
            </Header>
            <Footer />
        </Layout>
    )
}

export const pageQuery = graphql`
    query {
        site {
            siteMetadata {
                twitter
                email
            }
        }
    }
`
