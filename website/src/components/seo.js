import React from 'react'
import Helmet from 'react-helmet'
import { StaticQuery, graphql } from 'gatsby'

import socialImage from '../images/social.jpg'

const SEO = ({ title, description, lang = 'en' }) => (
    <StaticQuery
        query={query}
        render={data => {
            const site = data.site.siteMetadata
            const sloganTitle = `${site.title}  · ${site.slogan}`
            const pageTitle = title ? `${title} · ${sloganTitle}` : sloganTitle
            const pageDesc = description || site.description
            const image = site.siteUrl + socialImage
            const meta = [
                {
                    name: 'description',
                    content: pageDesc,
                },
                {
                    property: 'og:title',
                    content: pageTitle,
                },
                {
                    property: 'og:description',
                    content: pageDesc,
                },
                {
                    property: 'og:type',
                    content: `website`,
                },
                {
                    property: 'og:site_name',
                    content: site.title,
                },
                {
                    property: 'og:image',
                    content: image,
                },
                {
                    name: 'twitter:card',
                    content: 'summary_large_image',
                },
                {
                    name: 'twitter:image',
                    content: image,
                },
                {
                    name: 'twitter:creator',
                    content: `@${site.twitter}`,
                },
                {
                    name: 'twitter:site',
                    content: `@${site.twitter}`,
                },
                {
                    name: 'twitter:title',
                    content: pageTitle,
                },
                {
                    name: 'twitter:description',
                    content: pageDesc,
                },
            ]

            return <Helmet htmlAttributes={{ lang }} title={pageTitle} meta={meta} />
        }}
    />
)

export default SEO

const query = graphql`
    query DefaultSEOQuery {
        site {
            siteMetadata {
                title
                description
                siteUrl
                slogan
                twitter
            }
        }
    }
`
