import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="upscaler",
    version="0.0.6",
    author="Morphogen",
    author_email="vipermu97@gmail.com",
    description="Let's upscale!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Morphogens/upscaler",
    project_urls={},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch==1.7.1",
        "pillow==8.4.0",
        "torchvision==0.8.2",
        "tqdm==4.62.3",
    ],
)