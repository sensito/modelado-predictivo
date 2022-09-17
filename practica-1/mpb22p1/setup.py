import setuptools 
setuptools.setup(
    name='mpb22p1',  
    version='0.1',
    author='Daniel Ivan Medina Barreras',
    author_email='danielivanhola@gmail.com',
    description="Tarea 1 de Modelado Predictivo",
    url="#",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
#comando para instalar el paquete en el entorno virtual de python: pip install -e .