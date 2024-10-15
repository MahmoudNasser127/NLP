rom setuptools import setup, find_packages

def get_requirements(file_name):
    with open(file_name) as file_obj:
        requirements = file_obj.readlines()
        return [req.strip() for req in requirements]

setup(
    name='SentimentAnalysisApp',
    version='0.1.0',
    author='Your Name',
    author_email='your_email@example.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    include_package_data=True,  # Include templates and static files
    entry_points={
        'console_scripts': [
            'run-app=your_module_name:app.run',  # Replace 'your_module_name' with the actual module name where your Flask app is defined
        ],
    },
)