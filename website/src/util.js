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
 * @param obj - The object to check.
 * @returns {boolean} - Whether the object is a number string.
 */
export function isNumString(obj) {
    return isString(obj) && /^\d+[.,]?[\dx]+?(|x|ms|mb|gb|k|m)?$/i.test(obj)
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
 * Create an ID (used for anchor links etc.) given a recipe name
 */
export function makeRecipeId(name) {
    return isString(name) ? name.replace('.', '-') : name
}

/**
 * Check if an attribute value (received via props) is really truthy. This
 * allows setting attr="false" in the content (which would otherwise evaluate
 * as truthy, because the component actually receives the string "false").
 */
export function isTrue(value) {
    return value !== 'false' && !!value
}

/**
 * Check if we're on the client or server-side rendering
 */
export const isClient = typeof window !== 'undefined'
