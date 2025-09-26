from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements/base.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sentiment-mlops",
    version="1.0.0",
    author="Cynthia Ukawu",
    author_email="cynthia.ukawu@cynscode.com",
    description="Production-ready sentiment analysis with MLOps",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github/cynthiaukawu/sentiment-mlops",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "flake8>6.1.0",
            "mypy>=1.7.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "sentiment-train=scripts.train:main",
            "sentiment-serve=app.gradio_app:main",
            "sentiment-deploy=scripts.deploy:main",
        ],
    },
)
