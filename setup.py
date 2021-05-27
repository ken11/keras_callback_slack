import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="keras_callback_slack",
    version="0.0.5",
    author="ken",
    author_email="kent.adachi@adachi-honten.net",
    description="keras custom callback for slack",
    keywords='keras callback slack notification',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ken11/keras_callback_slack",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "keras>=2.4", "matplotlib>=3", "slack-sdk>=3.5"],
)
