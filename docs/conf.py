project = "AquaMVS"
copyright = "2024-2025, Tucker Lancaster"
author = "Tucker Lancaster"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
}

# HTML theme and branding
html_theme = "alabaster"
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
