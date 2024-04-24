# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "medkit"
author = "HeKA Research Team"
project_copyright = f"2022-2024, {author}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoapi.extension",
    "myst_nb",
    "numpydoc",
    "sphinxcontrib.mermaid",
    "sphinx_design",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- autoapi configuration ---------------------------------------------------
# https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html

autoapi_dirs = ["../medkit"]
autoapi_root = "reference/api"

# -- myst_parser configuration -----------------------------------------------
# https://myst-parser.readthedocs.io/en/latest/configuration.html

myst_enable_extensions = ["attrs_inline", "colon_fence"]
myst_heading_anchors = 2

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/medkit-lib/medkit",
    "repository_branch": "main",
    "navigation_with_keys": False,
}
html_title = "medkit documentation"
html_logo = "_static/medkit-logo.png"
html_favicon = "_static/medkit-icon.png"
html_static_path = ["_static"]
