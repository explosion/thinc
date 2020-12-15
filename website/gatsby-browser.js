exports.onRouteUpdate = ({ location }) => {
    // Fix anchor links, especially if links are opened in new tab
    if (location.hash) {
        setTimeout(() => {
            const el = document.querySelector(`${location.hash}`)
            if (el) {
                // Navigate to targeted element
                el.scrollIntoView()
                // Force recomputing :target pseudo class with pushState/popState
                window.location.hash = location.hash
            }
        }, 0)
    }
}
