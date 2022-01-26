from distutils.core import setup

setup(
    name="neurodmd",
    version="0.1.0",
    author="Tiberiu Tesileanu",
    author_email="ttesileanu@flatironinstitute.org",
    packages=["neurodmd"],
    install_requires=[
        "numpy",
        "scipy",
        "torch",
        "setuptools",
        "matplotlib",
        "seaborn",
        "pydove",
        "tqdm",
    ]
)
