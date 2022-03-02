const meta = require('./meta.json')
const autoprefixer = require('autoprefixer')
const fs = require('fs')

const plugins = [
    {
        resolve: `gatsby-plugin-sass`,
        options: {
            indentedSyntax: true,
            postCssPlugins: [autoprefixer()],
            cssLoaderOptions: {
                localIdentName:
                    process.env.NODE_ENV == 'development' ? '[name]-[local]-[hash:8]' : '[hash:8]',
            },
        },
    },
    `gatsby-plugin-react-helmet`,
    {
        resolve: `gatsby-source-filesystem`,
        options: {
            name: `content`,
            path: `${__dirname}/docs`,
        },
    },
    {
        resolve: `gatsby-source-filesystem`,
        options: {
            name: `images`,
            path: `${__dirname}/src/images`,
        },
    },
    {
        resolve: 'gatsby-plugin-react-svg',
        options: {
            rule: {
                include: /src\/images\/(.*)\.svg/,
            },
        },
    },
    {
        resolve: `gatsby-transformer-remark`,
        options: {
            pedantic: false,
            plugins: [
                `gatsby-remark-copy-linked-files`,
                `gatsby-remark-unwrap`,
                {
                    resolve: `gatsby-remark-images`,
                    options: {
                        maxWidth: 790,
                        linkImagesToOriginal: true,
                        sizeByPixelDensity: false,
                        showCaptions: true,
                        quality: 80,
                        withWebp: { quality: 80 },
                        backgroundColor: 'transparent',
                        disableBgImageOnAlpha: true,
                        loading: 'lazy',
                    },
                },
                `gatsby-remark-custom-attrs`,
                `gatsby-remark-code-blocks`,
                {
                    resolve: `gatsby-remark-smartypants`,
                    options: {
                        dashes: 'oldschool',
                    },
                },
            ],
        },
    },
    `gatsby-transformer-sharp`,
    `gatsby-plugin-sharp`,
    `gatsby-plugin-sitemap`,
    {
        resolve: `gatsby-plugin-manifest`,
        options: {
            name: meta.title,
            short_name: meta.title,
            start_url: `/`,
            background_color: meta.theme,
            theme_color: meta.theme,
            display: `minimal-ui`,
            icon: `src/images/icon.png`,
        },
    },
    {
        resolve: `gatsby-plugin-plausible`,
        options: {
            domain: meta.domain,
        },
    },
    `gatsby-plugin-offline`,
]

if (fs.existsSync('./src/fonts')) {
    plugins.push({
        resolve: `gatsby-plugin-sass-resources`,
        options: {
            resources: ['./src/styles/fonts.sass'],
        },
    })
}

module.exports = {
    siteMetadata: meta,
    plugins,
}
