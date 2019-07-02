"""
Adaptive Design Optimization on Experimental Tasks
"""
import os
import sys
import subprocess
from codecs import open as codecs_open
from setuptools import setup, find_packages

if sys.version_info[:2] < (3, 5):
    raise RuntimeError("Python version >= 3.5 required.")

MAJOR = 0
MINOR = 3
MICRO = 1
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


def git_version():
    """
    Return the git revision as a string
    """
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def get_version_info():
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of adopy.version messes up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('adopy/version.py'):
        # must be a source distribution, use existing version file
        try:
            from adopy.version import git_revision as GIT_REVISION
        except ImportError:
            raise ImportError("Unable to import git_revision. Try removing "
                              "adopy/version.py and the build directory "
                              "before building.")
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION


def write_version_py(filename='adopy/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM adopy SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s
if not release:
    version = full_version
"""
    FULLVERSION, GIT_REVISION = get_version_info()

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


write_version_py()


# Get the long description from the relevant file
with codecs_open('README.md', encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='adopy',
    version=VERSION,
    url='https://adopy.org/',
    description='Adaptive Design Optimization on Experimental Tasks',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Jaeyeong Yang',
    author_email='jaeyeong.yang1125@gmail.com',
    license='GPL-3',
    python_requires='>=3.5',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
    ],
    zip_safe=False,
)
