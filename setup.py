from setuptools import setup, find_packages


# tbd
setup(
    name='pandas_agent',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
    ],
    author='Karim Dabbouz',
    author_email='hey+pandas@karim.ooo',
    description='An agent to work with pandas using LLMs',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/karimdabbouz/pandas_agent',
    python_requires='>=3.6',
)
