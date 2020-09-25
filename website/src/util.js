import { Parser as HtmlToReactParser } from 'html-to-react'

const htmlToReactParser = new HtmlToReactParser()

/**
 * @param obj – The object to check.
 * @returns {boolean} - Whether the object is a string.
 */
export function isString(obj) {
    return typeof obj === 'string' || obj instanceof String
}

/**
 * Convert raw HTML to React elements
 * @param {string} html - The HTML markup to convert.
 * @returns {Node} - The converted React elements.
 */
export function htmlToReact(html) {
    return htmlToReactParser.parse(html)
}

/**
 * Get string value of component children
 */
export function getStringChildren(children) {
    if (isString(children)) {
        return children
    }
    return Array.isArray(children) && children.length === 1 ? children[0] : ''
}

/**
 * Check if an attribute value (received via props) is really truthy. This
 * allows setting attr="false" in the content (which would otherwise evaluate
 * as truthy, because the component actually receives the string "false").
 */
export function isTrue(value) {
    return value !== 'false' && !!value
}
