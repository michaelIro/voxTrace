# Use breathe extension to sphinx
extensions = [ "breathe" ]

master_doc = 'rst/index'
html_theme = 'sphinx_rtd_theme'

# Breathe Configuration
breathe_projects = {"VoxTrace++": "../build/docs/xml/",}
breathe_default_project = "VoxTrace++"
# sakldkl
project = "VoxTrace++"
author = "Michael Iro"
copyright = '2021, Michael Iro'
version = '1.0'

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}