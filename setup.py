import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='BubbleBox',  
     version='0.1.4',
     author="Audun Skau Hansen",
     author_email="a.s.hansen@kjemi.uio.no",
     description="A molecular dynamics educational tool for Jupyter Notebooks",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.uio.no/audunsh/bubblebox",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
