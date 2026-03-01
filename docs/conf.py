project = "AquaMVS"
copyright = "2024-2026, Tucker Lancaster"
author = "Tucker Lancaster"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.mermaid",
    "myst_nb",
]

# myst-nb: do not re-execute notebooks on RTD; use pre-executed outputs
nb_execution_mode = "off"

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "tutorial/aquamvs-example-dataset",
    "tutorial/.ipynb_checkpoints",
]

# HTML theme and branding
html_theme = "furo"
html_title = "AquaMVS"

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# Autodoc settings
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# Napoleon settings
napoleon_google_docstrings = True

# Suppress warnings for benchmark images that are generated on-demand
suppress_warnings = ["image.not_readable", "toc.not_included", "myst.xref_missing"]
