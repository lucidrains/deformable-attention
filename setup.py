from setuptools import setup, find_packages

setup(
  name = 'deformable-attention',
  packages = find_packages(exclude=[]),
  version = '0.0.18',
  license='MIT',
  description = 'Deformable Attention - from the paper "Vision Transformer with Deformable Attention"',
  long_description_content_type = 'text/markdown',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/deformable-attention',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism'
  ],
  install_requires=[
    'einops>=0.4',
    'torch>=1.10'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
