from setuptools import setup, find_packages

setup(
    name='baby_cry_mlflow',
    version='0.1',
    description='Classification of signals to detect baby cry',
    url="https://github.com/giulbia/baby_cry_mlflow.git",
    author='Giulia Bianchi',
    author_email="gbianchi@xebia.fr",
    license='new BSD',
    packages=find_packages(),
    install_requires=['numpy', 'librosa'],
    tests_require=['pytest', "unittest2"],
    scripts=[],
    py_modules=["baby_cry_mlflow"],
    include_package_data=True,
    zip_safe=False
)
