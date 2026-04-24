# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:35:52 2024

@author: seguret_j
"""

from cx_Freeze import setup, Executable
import numpy

# Remplacez 'Tomo_metrics.py' par le nom de votre script Python
script = 'Transfer_impact.py'

# Ajoutez toutes les dépendances supplémentaires ici
build_exe_options = {
    'packages': ['pydicom', 'numpy'],
    'includes': [],
    'excludes': [],
    'include_files': [(numpy.get_include(), 'numpy')]  # Inclure le répertoire numpy
}

# Configurez cx_Freeze
setup(
    name='Transfer_Error',
    version='0.1',
    description='Your application description',
    options={'build_exe': build_exe_options},
    executables=[Executable(script)]
)
