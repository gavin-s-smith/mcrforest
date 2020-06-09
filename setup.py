from setuptools import setup
#from numpy.distutils.core import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy 
import os


libraries = []
if os.name == 'posix':
    libraries.append('m')

extensions = [
    Extension(
        "mcrforest._tree",
        ["mcrforest/_tree.pyx"],
        include_dirs=[numpy.get_include()], # not needed for fftw unless it is installed in an unusual place
        libraries=libraries,
        extra_compile_args=["-O3"], # not needed for fftw unless it is installed in an unusual place
    ),
    Extension(
        "mcrforest._splitter",
        ["mcrforest/_splitter.pyx"],
        include_dirs=[numpy.get_include()], # not needed for fftw unless it is installed in an unusual place
        libraries=libraries,
        extra_compile_args=["-O3"], # not needed for fftw unless it is installed in an unusual place
    ),
    Extension(
        "mcrforest._criterion",
        ["mcrforest/_criterion.pyx"],
        include_dirs=[numpy.get_include()], # not needed for fftw unless it is installed in an unusual place
        libraries=libraries,
        extra_compile_args=["-O3"], # not needed for fftw unless it is installed in an unusual place
    ),
    Extension(
        "mcrforest._utils",
        ["mcrforest/_utils.pyx"],
        include_dirs=[numpy.get_include()], # not needed for fftw unless it is installed in an unusual place
        libraries=libraries,
        extra_compile_args=["-O3"], # not needed for fftw unless it is installed in an unusual place
    ),
    Extension(
        "mcrforest._quad_tree",
        ["mcrforest/_quad_tree.pyx"],
        include_dirs=[numpy.get_include()], # not needed for fftw unless it is installed in an unusual place
        libraries=libraries,
        extra_compile_args=["-O3"], # not needed for fftw unless it is installed in an unusual place
    ),
    Extension(
        "mcrforest._random",
        ["mcrforest/_random.pyx"],
        include_dirs=[numpy.get_include()], # not needed for fftw unless it is installed in an unusual place
        libraries=libraries,
        extra_compile_args=["-O3"], # not needed for fftw unless it is installed in an unusual place
    )
]


package_data = {'mcrforest': ['*.pxd']}


setup(
    name = "mcrforest",
    version='0.9.0',
    description='Random Forest implementation with VIM',
    url='http://github.com/gavin-s-smith/mcrforest',
    author='Gavin Smith',
    author_email='gavin.smith@nottingham.ac.uk',
    license='MIT',
    packages = ["mcrforest"],
    zip_safe=False,
    ext_modules = cythonize(extensions),
    include_package_data=True,
    package_data=package_data
)


#if __name__ == "__main__":
#    from setuptools import setup, find_packages
#    from setuptools.extension import Extension
#    from Cython.Build import cythonize
#    from numpy.distutils.core import setup
#    setup(
#    name = "mcrforest",
#    packages = ['mcrforest'],
#    ext_modules = cythonize(extensions)
#	)
