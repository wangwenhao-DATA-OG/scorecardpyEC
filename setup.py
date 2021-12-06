import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scorecardpyEC",
    version="1.1.1",
    author="王文皓(wangwenhao)",
    author_email="DATA-OG@139.com",
    description="为评分卡项目https://github.com/ShichenXie/scorecardpy提供常用的额外的工具组件。可以帮助评分卡开发人员提高开发效率",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wangwenhao-DATA-OG/scorecardpyEC",
    packages=setuptools.find_packages(),
    install_requires = ['scorecardpy >= 0.1.8','pandas >= 0.25.1','numpy >= 1.17.4'],
    keywords='score card,scorecardpy',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.4',
)