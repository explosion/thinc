import React from 'react'

import Layout from '../components/layout'
import { Header, Section, Feature, FeatureGrid, InlineCode } from '../components/landing'
import { H2 } from '../components/typography'
import Link, { Button } from '../components/link'
import Footer from '../components/footer'
import PyTorchLogo from '../images/logos/pytorch.svg'
import TensorFlowLogo from '../images/logos/tensorflow.svg'
import classes from '../styles/landing.module.sass'

export default () => (
    <Layout className={classes.root}>
        <Header>
            <H2>
                A lightweight Python library for <br className={classes.br} />
                framework-agnostic neural networks{' '}
                <span className={classes.slogan}>from the makers of spaCy</span>
            </H2>
        </Header>
        <Section>
            <FeatureGrid>
                <Feature title="Use any framework" emoji="🔮">
                    <p>
                        Switch between PyTorch and TensorFlow models without changing your
                        application, or even create mutant hybrids using zero-copy array
                        interchange.
                    </p>
                    <Link to="/docs/usage-frameworks#pytorch" hidden>
                        <PyTorchLogo className={classes.featureLogo} width={95} height={26} />
                    </Link>
                    <Link to="/docs/usage-frameworks#tensorflow" hidden>
                        <TensorFlowLogo className={classes.featureLogo} width={123} height={32} />
                    </Link>
                </Feature>
                <Feature title="Type checking" emoji="🚀">
                    <p>
                        Develop faster and catch bugs sooner with sophisticated type checking.
                        Trying to pass a 1-dimensional array into a model that expects 2 dimensions?
                        That’s a type error. Your editor can pick it up as the code leaves your
                        fingers.
                    </p>
                </Feature>
                <Feature title="Awesome config" emoji="🐍">
                    <p>
                        Configuration is a major pain for ML. Thinc’s solution is a bit like Gin,
                        but simpler and cleaner – and it works for both research and production.
                    </p>
                </Feature>
                <Feature title="Super lightweight" emoji="🦋">
                    <p>
                        Small and easy to install, with very few required dependencies, available on{' '}
                        <InlineCode>pip</InlineCode> and <InlineCode>conda</InlineCode> for Linux,
                        macOS and Windows.
                    </p>
                </Feature>
                <Feature title="Battle-tested" emoji="⚔️">
                    <p>
                        Thinc’s redesign is brand new, but previous versions have been powering
                        spaCy since its release, putting Thinc into production in thousands of
                        companies.
                    </p>
                </Feature>
                <Feature title="Innovative design" emoji="🔥">
                    <p>
                        Neural networks have changed a lot over the last few years, and Python has
                        too. Armed with new tools, Thinc offers a fresh look at the problem.
                    </p>
                </Feature>
            </FeatureGrid>
        </Section>

        <Section className={classes.callToAction}>
            <Button to="/docs" primary>
                Read more
            </Button>
        </Section>

        <Footer className={classes.footer} />
    </Layout>
)
