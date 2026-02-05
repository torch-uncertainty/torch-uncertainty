# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
from datetime import datetime
from pathlib import Path

from sphinx_gallery.sorting import ExplicitOrder, FileNameSortKey

sys.path.insert(0, str(Path("../../").resolve()))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "TorchUncertainty"

copyright = (  # noqa: A001
    f"{datetime.now().year!s}, Adrien Lafage and Olivier Laurent"
)
author = "Adrien Lafage and Olivier Laurent"
release = "0.10.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinxcontrib.sass",
    # "sphinxcontrib.katex",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # "sphinx.ext.intersphinx",  # for links to the API in the tutorials
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_codeautolink",
    "sphinx_gallery.gen_gallery",
    # "sphinx_gallery.load_style",
    "sphinx_design",
]
mathjax3_config = {
    "tex": {
        "inlineMath": [["\\(", "\\)"], ["$", "$"]],
        "displayMath": [["\\[", "\\]"], ["$$", "$$"]],
    },
}

subsection_order = ExplicitOrder(
    [
        "../../auto_tutorial_source/Classification",
        "../../auto_tutorial_source/Regression",
        "../../auto_tutorial_source/Post_Hoc_Methods",
        "../../auto_tutorial_source/Bayesian_Methods",
        "../../auto_tutorial_source/Ensemble_Methods",
        "../../auto_tutorial_source/Segmentation",
        "../../auto_tutorial_source/Data_Augmentation",
    ]
)


sphinx_gallery_conf = {
    "examples_dirs": ["../../auto_tutorial_source"],
    "gallery_dirs": "auto_tutorials",
    "filename_pattern": r"tutorial_",
    "subsection_order": subsection_order,
    "within_subsection_order": FileNameSortKey("./source"),
    "plot_gallery": "True",
    "promote_jupyter_magic": True,
    "backreferences_dir": "generated/backreferences",
    "doc_module": ("torch_uncertainty",),
    "first_notebook_cell": (
        "# For tips on running notebooks in Google Colab, see\n"
        "# https://pytorch.org/tutorials/beginner/colab\n"
        "%matplotlib inline"
    ),
    "reference_url": {
        "sphinx_gallery": None,
    },
}

# Use both the docstrings of the init and the class
autoclass_content = "init"

autosummary_generate = True
napoleon_use_ivar = True


# Disable docstring inheritance
autodoc_inherit_docstrings = False

# Disable displaying type annotations
autodoc_typehints = "none"

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# avoid re-run of notebooks
nbsphinx_execute = "never"


html_theme = "pydata_sphinx_theme"


sass_src_dir = "_sass"
sass_out_dir = "_static"
sass_targets = {"index.scss": "index.css"}

html_static_path = ["_static"]


html_theme_options = {
    "logo": {
        "text": "",
        "image_light": "_static/logo.png",  # Path to the logo for light mode
        "image_dark": "_static/logo_dark.png",  # Path to the logo for dark mode
        "alt_text": "TorchUncertainty Logo",  # Alternative text for accessibility
    },
    "navbar_start": ["navbar-logo"],
    "github_url": "https://github.com/torch-uncertainty/torch-uncertainty",
    "footer_end": [],
    "footer_start": [],
    "footer_center": ["copyright"],
}

html_additional_pages = {"index": "custom_index.html"}

html_sidebars = {
    "cli_guide": [],
    "contributing": [],
    "installation": [],
    "quickstart": [],
    "references": [],
    "index": [],
    "**": ["sidebar-nav-bs", "sidebar-ethical-ads"],
}


def add_custom_css(app, pagename, _templatename, _context, _doctree) -> None:
    if pagename == "index":
        app.add_css_file("index.css")


def setup(app) -> None:
    app.connect("html-page-context", add_custom_css)
