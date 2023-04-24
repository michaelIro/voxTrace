# Use breathe extension to sphinx
extensions = [ "breathe" ]

master_doc = 'index'
html_theme = 'sphinx_rtd_theme'

# Breathe Configuration
breathe_projects = {"voxTrace": "../build/doc/xml/",}
breathe_default_project = "voxTrace"

# Project information
project = "voxTrace"
author = "Michael Iro"
copyright = '2023, Michael Iro'
version = '1.0'

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}