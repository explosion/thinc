import React from 'react'

import Layout from '../components/layout'
import { Header, Section, Feature, FeatureGrid, InlineCode } from '../components/landing'
import { H2 } from '../components/typography'
import Link, { Button } from '../components/link'
import Footer from '../components/footer'
import PyTorchLogo from '../images/logos/pytorch.svg'
import TensorFlowLogo from '../images/logos/tensorflow.svg'
import MXNetLogo from '../images/logos/mxnet.svg'
import classes from '../styles/landing.module.sass'

export default () => (
    <Layout className={classes.root}>
        <Header logoLink="/docs">
            <H2>
                A refreshing functional take on deep learning, <br className={classes.mdOnly} />
                compatible with your favorite libraries.{' '}
                <span className={classes.slogan}>
                    from the makers of <br className={classes.smOnly} />
                    <Link to="https://spacy.io" hidden>
                        spaCy
                    </Link>
                    ,{' '}
                    <Link to="https://prodi.gy" hidden>
                        Prodigy
                    </Link>{' '}
                    &amp;{' '}
                    <Link to="https://fastapi.tiangolo.com" hidden>
                        FastAPI
                    </Link>
                </span>
            </H2>
        </Header>
        <Section>
            <FeatureGrid>
                <Feature title="Use any framework" emoji="🔮">
                    <p>
                        Switch between PyTorch, TensorFlow and MXNet models without changing your
                        application, or even create mutant hybrids using zero-copy array
                        interchange.
                    </p>
                    <Link to="/docs/usage-frameworks" hidden>
                        <PyTorchLogo className={classes.featureLogo} width={75} height={20} />
                    </Link>
                    <Link to="/docs/usage-frameworks" hidden>
                        <TensorFlowLogo className={classes.featureLogo} width={100} height={26} />
                    </Link>
                    <Link to="/docs/usage-frameworks" hidden>
                        <MXNetLogo className={classes.featureLogo} width={60} height={20} />
                    </Link>
                </Feature>
                <Feature title="Type checking" emoji="🚀">
                    <p>
                        Develop faster and catch bugs sooner with sophisticated type checking.
                        Trying to pass a 1-dimensional array into a model that expects two dimensions?
                        That’s a type error. Your editor can pick it up as the code leaves your
                        fingers.
                    </p>
                </Feature>
                <Feature title="Awesome config" emoji="🐍">
                    <p>
                        Configuration is a major pain for ML. Thinc lets you describe trees of
                        objects with references to your own functions, so you can stop passing
                        around blobs of settings. It's simple, clean, and it works for both research
                        and production.
                    </p>
                </Feature>
                <Feature title="Super lightweight" emoji="🦋">
                    <p>
                        Small and easy to install with very few required dependencies, available on{' '}
                        <InlineCode>pip</InlineCode> and <InlineCode>conda</InlineCode> for Linux,
                        macOS and Windows. Simple source with a consistent API.
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
