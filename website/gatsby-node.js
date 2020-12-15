const path = require('path')
const { createFilePath } = require('gatsby-source-filesystem')

const pageTemplate = path.resolve('src/templates/docs.js')

function replacePath(pagePath) {
    return pagePath === `/` ? pagePath : pagePath.replace(/\/$/, ``)
}

exports.onCreatePage = ({ page, actions }) => {
    const { createPage, deletePage } = actions
    const oldPage = Object.assign({}, page)
    if (oldPage.path != '/dev-404-page/') {
        page.path = replacePath(page.path)
        if (page.path !== oldPage.path) {
            deletePage(oldPage)
            createPage(page)
        }
    }
}

exports.onCreateNode = ({ node, actions, getNode }) => {
    const { createNodeField } = actions
    if (node.internal.type === 'MarkdownRemark') {
        const slug = createFilePath({ node, getNode, basePath: 'docs', trailingSlash: false })
        createNodeField({ name: 'slug', node, value: `/docs${slug}` })
    }
}

exports.createPages = ({ actions, graphql }) => {
    const { createPage } = actions
    return graphql(`
        {
            allMarkdownRemark {
                edges {
                    node {
                        frontmatter {
                            title
                        }
                        fields {
                            slug
                        }
                    }
                }
            }
        }
    `).then(result => {
        if (result.errors) {
            return Promise.reject(result.errors)
        }
        const posts = result.data.allMarkdownRemark.edges
        posts.forEach(({ node }) => {
            createPage({
                path: replacePath(node.fields.slug),
                component: pageTemplate,
                context: {
                    slug: node.fields.slug,
                },
            })
        })
    })
}
