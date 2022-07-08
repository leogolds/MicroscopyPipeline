from setuptools import setup

setup(
    name="microscopy-pipeline",
    version="0.1",
    py_modules=["cli"],
    install_requires=[
        "Click",
    ],
    entry_points="""
        [console_scripts]
        microscopy-pipeline=cli:app
    """,
)
