# Use breathe extension to sphinx
extensions = [ "breathe" ]

master_doc = 'rst/index'
html_theme = 'sphinx_rtd_theme'

# Breathe Configuration
breathe_projects = {"voxTrace": "../build/docs/xml/",}
breathe_default_project = "voxTrace"
# sakldkl
project = "voxTrace"
author = "Michael Iro"
copyright = '2022, Michael Iro'
version = '1.0'

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}