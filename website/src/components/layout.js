import React from 'react'

import SEO from './seo'

const Layout = ({ title, description, className, children }) => (
    <>
        <SEO title={title} description={description} />
        <main className={className}>{children}</main>
    </>
)

export default Layout
