from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

REPO_NAME = "guidesturkiye"
AUTHOR_USER_NAME = "muhammedaliuyanik"
SRC_REPO = "src"
LIST_OF_REQUIREMENTS = [
    'pandas',
    'numpy',
    'scikit-learn',
    'nltk',
    'streamlit',
    'requests',
    'torch',
    'sklearn'
]

setup(
    name=SRC_REPO,
    version="0.0.1",
    author=AUTHOR_USER_NAME,
    description="A small package for Travel Recommender System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    author_email="muhammedaliuyank16@gmail.com",
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.7",
    install_requires=LIST_OF_REQUIREMENTS
)
