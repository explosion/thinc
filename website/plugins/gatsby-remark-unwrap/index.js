/**
 * Unwrap elements and remove outer <p></p>
 * Fork of: https://github.com/xuopled/gatsby-remark-unwrap-images
 */
const visit = require('unist-util-visit')
const remove = require('unist-util-remove')

const defaultOptions = {
    elements: ['image', 'html'],
}

function isWrapped(child, elements = []) {
    return elements.includes(child.type) || (child.type === 'text' && child.value === '\n')
}

module.exports = ({ markdownAST }, userOptions = {}) => {
    const options = Object.assign({}, defaultOptions, userOptions)
    const elements = options.elements || []
    visit(markdownAST, 'paragraph', (node, index, parent) => {
        const wrapped = node.children.every(child => isWrapped(child, elements))
        if (!wrapped) return
        remove(node, 'text')
        parent.children.splice(index, 1, ...node.children)
        return index
    })
    return markdownAST
}
