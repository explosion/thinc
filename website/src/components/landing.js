import React from 'react'
import classNames from 'classnames'
import { graphql, StaticQuery } from 'gatsby'
import Img from 'gatsby-image'

import { InlineCode as DefaultInlineCode } from '../components/code'
import Link, { Button } from '../components/link'
import { H3 } from '../components/typography'
import { Emoji } from '../components/icon'
import ExplosionLogo from '../images/logos/explosion.svg'
import Logo from '../images/logo.svg'
import classes from '../styles/landing.module.sass'

const headerQuery = graphql`
    query {
        site {
            siteMetadata {
                company
                companyUrl
            }
        }
        headerTopLeft: file(relativePath: { eq: "landing_top-left.png" }) {
            ...headerImage
        }
        headerTopRight: file(relativePath: { eq: "landing_top-right.png" }) {
            ...headerImage
        }
    }
`

export const headerImage = graphql`
    fragment headerImage on File {
        childImageSharp {
            fluid(maxWidth: 1200, quality: 100) {
                ...GatsbyImageSharpFluid
            }
        }
    }
`

export const Header = ({ logo = true, logoLink, companyLogo = true, children }) => {
    const logoSvg = <Logo className={classes.logo} />
    return (
        <StaticQuery
            query={headerQuery}
            render={data => {
                const { company, companyUrl } = data.site.siteMetadata
                return (
                    <header className={classes.header}>
                        {companyLogo && (
                            <Link to={companyUrl} hidden aria-label={company}>
                                <ExplosionLogo
                                    width={50}
                                    height={50}
                                    className={classes.logoExplosion}
                                />
                            </Link>
                        )}
                        <div className={classNames(classes.headerImage, classes.headerImageLeft)}>
                            <Img fluid={data.headerTopLeft.childImageSharp.fluid} />
                        </div>
                        <div className={classNames(classes.headerImage, classes.headerImageRight)}>
                            <Img fluid={data.headerTopRight.childImageSharp.fluid} />
                        </div>

                        {logo && logoLink ? (
                            <Link to={logoLink} hidden>
                                {logoSvg}
                            </Link>
                        ) : (
                            logoSvg
                        )}
                        <Section narrow>{children}</Section>
                    </header>
                )
            }}
        />
    )
}

export const InlineCode = props => <DefaultInlineCode className={classes.inlineCode} {...props} />

export const Section = ({
    title,
    to,
    buttonText = 'Read more',
    narrow = false,
    className,
    children,
}) => {
    const sectionClassNames = classNames(classes.section, className, {
        [classes.sectionNarrow]: narrow,
    })
    return (
        <section className={sectionClassNames}>
            {title && (
                <H2 Component="h2" className={classes.sectionTitle}>
                    {title}
                </H2>
            )}
            {children}
            {to && (
                <Button to={to} primary>
                    {buttonText}
                </Button>
            )}
        </section>
    )
}

export const Feature = ({ title, emoji, children }) => (
    <section className={classes.feature}>
        <H3 className={classes.featureTitle}>
            {emoji && <Emoji>{emoji}</Emoji>} {title}
        </H3>
        {children}
    </section>
)

export const FeatureGrid = ({ children }) => <div className={classes.featureGrid}>{children}</div>
