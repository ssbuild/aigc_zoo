# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/6/16 17:21


from setuptools import setup, find_packages

packge_list = find_packages('src')
package_dir= {'aigc_zoo.' + k : 'src/' + k.replace('.','/') for k in packge_list }
package_dir.update({'aigc_zoo': 'src'})


if __name__ == '__main__':
    setup(
        name='aigc_zoo',
        version='0.1.11.post0',
        description='AIGC zoo',
        long_description='torch_training: https://github.com/ssbuild/aigc_zoo.git',
        license='Apache License 2.0',
        url='https://github.com/ssbuild/aigc_zoo',
        author='ssbuild',
        author_email='9727464@qq.com',
        install_requires=[
            'deep_training>=0.1.11,<=0.1.13',
            'numpy_io>=0.0.7',
        ],
        packages= list(package_dir.keys()),
        package_dir= package_dir,
    )