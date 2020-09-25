import React from 'react'
import { graphql } from 'gatsby'
import Img from 'gatsby-image'

import Layout from '../components/layout'
import Sidebar from '../components/sidebar'
import Footer from '../components/footer'
import { H1 } from '../components/typography'
import { Button } from '../components/link'
import { renderAst } from '../markdown'

import classes from '../styles/docs.module.sass'

const Template = ({ data, pageContext }) => {
    const { allMarkdownRemark, markdownRemark, site } = data
    // Work around hot reloading race condition
    if (!markdownRemark) return <em>graphql query not ready yet...</em>
    const { frontmatter, htmlAst } = markdownRemark
    const { sidebar } = site.siteMetadata
    const { slug } = pageContext
    const { title, teaser, next } = frontmatter
    const html = renderAst(htmlAst)
    const allPages = allMarkdownRemark.nodes.map(n => ({
        [n.fields.slug]: n.frontmatter.title,
    }))
    const allPagesBySlug = Object.assign({}, ...allPages)
    return (
        <Layout title={title} section="docs">
            <div className={classes.headerImage}>
                <Img fluid={data.headerTopRight.childImageSharp.fluid} />
            </div>
            <div className={classes.root}>
                <div className={classes.sidebar}>
                    <Sidebar slug={slug} items={sidebar} />
                </div>
                <article className={classes.article}>
                    <H1 className={classes.title} meta={teaser}>
                        {title}
                    </H1>
                    {html}
                    {next && allPagesBySlug[next] && (
                        <footer className={classes.next}>
                            <Button to={next}>Next: {allPagesBySlug[next]}</Button>
                        </footer>
                    )}
                </article>
                <div className={classes.footer}>
                    <Footer />
                </div>
            </div>
        </Layout>
    )
}

export default Template

export const pageQuery = graphql`
    query($slug: String!) {
        site {
            siteMetadata {
                sidebar {
                    label
                    items {
                        text
                        url
                    }
                }
            }
        }
        markdownRemark(fields: { slug: { eq: $slug } }) {
            htmlAst
            frontmatter {
                title
                teaser
                next
            }
        }
        allMarkdownRemark {
            nodes {
                fields {
                    slug
                }
                frontmatter {
                    title
                }
            }
        }
        headerTopRight: file(relativePath: { eq: "landing_top-right.png" }) {
            childImageSharp {
                fluid(maxWidth: 500, quality: 100) {
                    ...GatsbyImageSharpFluid
                }
            }
        }
    }
`
