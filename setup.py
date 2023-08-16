# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/6/16 17:21


from setuptools import setup, find_packages

install_requires = [
    # 'deep_training>=0.1.13,<=0.1.15',
    'deep_training ~= 0.1.20',
    'numpy_io>=0.0.8',
]

if __name__ == '__main__':
    setup(
        name='aigc_zoo',
        version='0.1.20',
        description='AIGC zoo',
        long_description='torch_training: https://github.com/ssbuild/aigc_zoo.git',
        license='Apache License 2.0',
        url='https://github.com/ssbuild/aigc_zoo',
        author='ssbuild',
        author_email='9727464@qq.com',
        install_requires=install_requires,
        package_dir={"": "src"},
        packages=find_packages("src"),
        include_package_data=True,
        package_data={"": ["**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/*.pyx"]},
    )